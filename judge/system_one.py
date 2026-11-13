"""Step 3 — System One (TypeSafe Jev). The ONLY file that knows the TypeSafe API.

Design (per the TypeSafe skill): Jev makes small, observable judgments; this
file's code makes the keep/fix/discard decision from them.

  state  = 3 keyframes (start / middle / end) + measured local checks + Gemini's task label
  Jev    = one Noul per judgment ("Are the hands in view in `frames.middle`?" ...)
  code   = truth table: each failure mode has a probability; keep = all checks pass,
           otherwise the mass goes to fix or discard depending on whether the
           failure can be repaired. Routing (>= 0.80 or human_review) is in pipeline.py.

Key: TYPESAFE_API_KEY (SYSTEM_ONE_API_KEY also accepted). Model: TYPESAFE_MODEL, default jev-latest.
Images are a TypeSafe "dreamer-ready" feature, enabled per request with a header.
"""
from __future__ import annotations

import os
from pathlib import Path

VERDICTS = ["keep", "fix", "discard"]
REASONS = ["good", "blurry", "dark", "shaky", "hands_out_of_frame",
           "object_hidden", "task_failed", "multiple_tasks", "privacy"]

MODEL = os.environ.get("TYPESAFE_MODEL", "jev-latest")
IMAGE_HEADERS = {"TypeSafe-Danger-Zone-Images": "true"}

# Failure modes that can usually be repaired (relight, stabilise, trim, blur faces)
FIXABLE = {"dark", "shaky", "multiple_tasks", "privacy"}
# Noul bands from the TypeSafe guidance: below LOW reads as no, above HIGH as yes
LOW, HIGH = 0.3, 0.7

# ---------------------------------------------------------------- all question wording lives here
# Each entry: reason -> (question text, criteria for yes/no, answer that means FAILURE)
QUESTIONS = {
    "blurry": (
        "Is `frames.middle` too blurry to make out the edges of the object in the person's hand?",
        {"true": "Edges and textures are smeared; the object's outline can't be traced.",
         "false": "The object's outline and the fingers around it are crisp."},
        True,
    ),
    "dark": (
        "Is `frames.middle` too dark to tell what the hands are holding?",
        {"true": "The scene is mostly shadow and the held object can't be identified.",
         "false": "The hands and the object are clearly lit."},
        True,
    ),
    "hands_out_of_frame": (
        "Are the person's hands inside the picture in `frames.middle`?",
        {"true": "At least one hand is visible touching or reaching for the object.",
         "false": "No hand is visible, or only a fingertip at the very edge."},
        False,
    ),
    "object_hidden": (
        "Before the person touches it, is the object named in `task` clearly visible in `frames.start`?",
        {"true": "Most of the object can be seen in the picture.",
         "false": "The object is out of frame, or blocked by something other than the person's own hand."},
        False,
    ),
    "task_failed": (
        "Does `frames.end` show the task in `task` finished?",
        {"true": "The object is at its destination and the hand has let go or is letting go.",
         "false": "The object is still in hand, dropped, or back where it started."},
        False,
    ),
    "multiple_tasks": (
        "Do `frames.start` and `frames.end` show the person working on two different objects?",
        {"true": "The hands handle one object at the start and an unrelated object later.",
         "false": "The same object is handled throughout."},
        True,
    ),
    "privacy": (
        "Is a human face or a readable phone, computer or TV screen visible in any of `frames`?",
        {"true": "A face or legible screen content appears in at least one frame.",
         "false": "Only hands, objects and surfaces appear."},
        True,
    ),
}

# Local measurements are better done in code than by a model; they enter the
# same table as failure likelihoods derived from the measured values.
def _ramp(x: float, bad: float, good: float) -> float:
    t = (x - good) / (bad - good)
    return max(0.0, min(1.0, t))


def local_failure_probs(local: dict) -> dict[str, float]:
    return {
        "shaky": _ramp(local.get("shake_score", 0), bad=15, good=6),
        "blurry_measured": _ramp(local.get("blur_score", 999), bad=30, good=120),
        "dark_measured": _ramp(local.get("brightness_mean", 128), bad=25, good=70),
    }


# ---------------------------------------------------------------- request
def _api_key() -> str | None:
    return os.environ.get("TYPESAFE_API_KEY") or os.environ.get("SYSTEM_ONE_API_KEY")


def build_request(state: dict):
    """Return (state_payload, questions) for client.system_one. Exposed for logging/tests."""
    from typesafe_sdk import Image, Noul

    frames = state.get("frames") or {}
    facts = state.get("facts") or {}
    payload = {
        "frames": {k: Image(Path(p)) for k, p in frames.items()},
        "task": facts.get("task_label") or "pick up the apple and put it down somewhere else",
        "context": ("Frames from a first-person video recorded for robot training. "
                    "`frames.start`, `frames.middle` and `frames.end` are taken at 10%, 50% and 90% of the clip."),
    }
    questions = {k: Noul(instructions=q, criteria=c) for k, (q, c, _) in QUESTIONS.items()}
    return payload, questions


def ask(state: dict) -> dict[str, float]:
    """Send one request with every question; return {reason: P(failure)}."""
    key = _api_key()
    if not key:
        raise NotImplementedError("TYPESAFE_API_KEY not set — System One unavailable")
    if not state.get("frames"):
        raise ValueError("System One needs keyframes in state['frames']")
    from typesafe_sdk import TypeSafeClient

    payload, questions = build_request(state)
    with TypeSafeClient(api_key=key, model=MODEL) as client:
        resp = client.system_one(state=payload, questions=questions, extra_headers=IMAGE_HEADERS)
    out = {}
    for reason, (_, _, fail_if) in QUESTIONS.items():
        p_yes = float(resp.answers[reason].noul)
        out[reason] = p_yes if fail_if else 1.0 - p_yes
    return out


# ---------------------------------------------------------------- decision (code, not model)
def combine(fail: dict[str, float], local: dict) -> dict:
    """Truth table over failure probabilities -> verdict + main reason distributions."""
    loc = local_failure_probs(local)
    p = dict(fail)
    # measurement and judgment agree -> trust the stronger signal
    p["blurry"] = max(p.get("blurry", 0.0), loc["blurry_measured"])
    p["dark"] = max(p.get("dark", 0.0), loc["dark_measured"])
    p["shaky"] = loc["shaky"]

    # Truth table with the TypeSafe-recommended bands: < LOW = pass, > HIGH = fail, between = unsettled.
    fails = {r: v for r, v in p.items() if v >= HIGH}
    unsure = {r: v for r, v in p.items() if LOW < v < HIGH}
    unfixable = {r: v for r, v in fails.items() if r not in FIXABLE}
    if unfixable:                                  # any unrepairable failure
        choice, top = "discard", max(unfixable, key=unfixable.get)
        conf = p[top]
    elif fails:                                    # only repairable failures
        choice, top = "fix", max(fails, key=fails.get)
        conf = p[top]
    elif unsure:                                   # a judgment isn't settled: lean, but below the bar
        top = max(unsure, key=unsure.get)
        choice = "keep" if p[top] < 0.5 else ("fix" if top in FIXABLE else "discard")
        conf = max(p[top], 1 - p[top])             # <= HIGH, so pipeline routes to human_review
    else:                                          # everything passes
        choice, top = "keep", max(p, key=p.get)
        conf = 1 - p[top]                          # as sure as the weakest pass
    alt = "fix" if top in FIXABLE else "discard"
    if choice == alt or choice != "keep":
        alt = "keep"
    other = next(v for v in VERDICTS if v not in (choice, alt))
    rest = 1 - conf
    v_probs = {choice: conf, alt: rest * 0.8, other: rest * 0.2}

    keep = v_probs["keep"]
    weights = {r: p.get(r, 0.0) for r in REASONS if r != "good"}
    weights["good"] = keep
    zr = sum(weights.values()) or 1.0
    r_probs = {r: weights[r] / zr for r in REASONS}

    return {
        "verdict": {"choice": max(v_probs, key=v_probs.get),
                    "probs": {k: round(v, 4) for k, v in v_probs.items()}},
        "main_reason": {"choice": max(r_probs, key=r_probs.get),
                        "probs": {k: round(v, 4) for k, v in r_probs.items()}},
        "failure_probs": {k: round(v, 4) for k, v in p.items()},
    }


def decide(state: dict) -> dict:
    return combine(ask(state), state.get("local") or {})
