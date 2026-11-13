"""Transparent rule-based stand-in for System One.

Used ONLY while judge/system_one.py raises NotImplementedError (or when
--decider rules is passed). Output has the exact same shape as
system_one.decide(), and every record it produces is tagged
decider="rules_fallback" so it is never mistaken for a System One verdict.

Each failure mode gets a severity in [0, 1]. Severities become logits
(x SHARPNESS) and a softmax turns them into probabilities, so borderline
clips naturally land below the 0.80 routing threshold.
"""
from __future__ import annotations

import math

from .system_one import REASONS, VERDICTS

SHARPNESS = 5.0

# Reasons whose clips can usually be repaired rather than thrown away,
# as long as the problem is not severe (severity < SEVERE).
FIXABLE = {"dark", "shaky", "multiple_tasks", "privacy"}
SEVERE = 0.8


def _ramp(x: float, bad: float, good: float) -> float:
    """1.0 at/beyond `bad`, 0.0 at/beyond `good`, linear between."""
    if bad == good:
        return float(x <= bad)
    t = (x - good) / (bad - good)
    return max(0.0, min(1.0, t))


def _softmax(logits: dict[str, float]) -> dict[str, float]:
    m = max(logits.values())
    ex = {k: math.exp(v - m) for k, v in logits.items()}
    z = sum(ex.values())
    return {k: round(v / z, 4) for k, v in ex.items()}


def severities(state: dict) -> dict[str, float]:
    loc = state.get("local") or {}
    facts = state.get("facts") or {}
    sev = {
        "blurry": _ramp(loc.get("blur_score", 999), bad=30, good=120),
        "dark": max(_ramp(loc.get("brightness_mean", 128), bad=25, good=70),
                    _ramp(loc.get("dark_frame_ratio", 0), bad=0.6, good=0.1)),
        "shaky": _ramp(loc.get("shake_score", 0), bad=15, good=6),
    }
    if facts:
        tc = facts.get("task_completed")
        sev.update({
            "hands_out_of_frame": _ramp(facts.get("hands_visible_ratio", 1), bad=0.3, good=0.8),
            "object_hidden": _ramp(facts.get("object_visible_ratio", 1), bad=0.3, good=0.8),
            "task_failed": 1.0 if tc is False else 0.5 if tc == "unclear" else 0.0,
            "multiple_tasks": 0.0 if facts.get("single_task", True) else 0.7,
            "privacy": 0.75 if facts.get("faces_or_screens_visible") else 0.0,
        })
    return sev


def decide(state: dict) -> dict:
    sev = severities(state)
    worst = max(sev.values(), default=0.0)

    fix_mass = max((s for r, s in sev.items() if r in FIXABLE and s < SEVERE), default=0.0)
    discard_mass = max((s for r, s in sev.items() if r not in FIXABLE or s >= SEVERE), default=0.0)
    v_probs = _softmax({
        "keep": SHARPNESS * (1 - worst),
        "fix": SHARPNESS * fix_mass,
        "discard": SHARPNESS * discard_mass,
    })
    r_logits = {r: SHARPNESS * sev.get(r, 0.0) for r in REASONS if r != "good"}
    r_logits["good"] = SHARPNESS * (1 - worst)
    r_probs = _softmax({r: r_logits[r] for r in REASONS})

    return {
        "verdict": {"choice": max(VERDICTS, key=v_probs.get), "probs": v_probs},
        "main_reason": {"choice": max(REASONS, key=r_probs.get), "probs": r_probs},
        "severities": {k: round(v, 3) for k, v in sev.items()},
    }
