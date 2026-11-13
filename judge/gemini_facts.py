"""Step 2 — Gemini fact sheet (video understanding -> strict JSON).

Needs GEMINI_API_KEY. Model can be overridden with GEMINI_MODEL.
The response is validated against FACT_SCHEMA; on failure we retry once,
telling the model what was wrong. A second failure raises FactSheetError.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

ACTIONS = ["move_to", "grasp", "lift", "place", "release"]
DEFAULT_MODEL = "gemini-4-hydrogen"

# JSON schema sent to Gemini (response_json_schema). task_completed is a string
# enum here because "unclear" is not a boolean; normalise() converts it.
FACT_SCHEMA = {
    "type": "object",
    "properties": {
        "task_label": {"type": "string"},
        "hands_visible_ratio": {"type": "number", "minimum": 0, "maximum": 1},
        "object_visible_ratio": {"type": "number", "minimum": 0, "maximum": 1},
        "task_completed": {"type": "string", "enum": ["true", "false", "unclear"]},
        "single_task": {"type": "boolean"},
        "faces_or_screens_visible": {"type": "boolean"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "t_start": {"type": "number"},
                    "t_end": {"type": "number"},
                    "action": {"type": "string", "enum": ACTIONS},
                    "object": {"type": "string"},
                },
                "required": ["t_start", "t_end", "action", "object"],
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["task_label", "hands_visible_ratio", "object_visible_ratio",
                 "task_completed", "single_task", "faces_or_screens_visible",
                 "steps", "notes"],
}

PROMPT = """You are auditing an egocentric (first-person) video clip that may be used to train a
robot manipulation policy. Watch the whole clip and return ONLY a JSON object with:

- task_label: short imperative description of the manipulation task, e.g. "pick up cup and place on plate"
- hands_visible_ratio: fraction of the clip (0-1) in which the operator's hand(s) are in frame
- object_visible_ratio: fraction of the clip (0-1) in which the main manipulated object is visible and not occluded
- task_completed: "true" if the task is clearly finished on camera, "false" if it clearly fails or stops, "unclear" otherwise
- single_task: true if the clip contains exactly one task, false if several distinct tasks
- faces_or_screens_visible: true if any human face or readable screen/monitor/phone display appears
- steps: ordered list of {t_start, t_end (seconds), action, object}; action is one of
  move_to | grasp | lift | place | release
- notes: one sentence on anything that would affect training quality

Be strict and literal. Do not guess hidden content."""


class FactSheetError(RuntimeError):
    pass


class GeminiUnavailable(RuntimeError):
    """Raised when GEMINI_API_KEY is not set."""


def validate(d: object, duration_s: float | None = None) -> list[str]:
    """Return a list of problems; empty list means valid (after normalise())."""
    errs: list[str] = []
    if not isinstance(d, dict):
        return ["top level is not an object"]
    for k in FACT_SCHEMA["required"]:
        if k not in d:
            errs.append(f"missing key {k}")
    if errs:
        return errs
    if not isinstance(d["task_label"], str) or not d["task_label"].strip():
        errs.append("task_label must be a non-empty string")
    for k in ("hands_visible_ratio", "object_visible_ratio"):
        v = d[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1:
            errs.append(f"{k} must be a number in [0,1]")
    if d["task_completed"] not in (True, False, "unclear"):
        errs.append("task_completed must be true/false/unclear")
    for k in ("single_task", "faces_or_screens_visible"):
        if not isinstance(d[k], bool):
            errs.append(f"{k} must be boolean")
    if not isinstance(d["notes"], str):
        errs.append("notes must be a string")
    if not isinstance(d["steps"], list):
        errs.append("steps must be a list")
    else:
        slack = 1.0
        for i, s in enumerate(d["steps"]):
            if not isinstance(s, dict):
                errs.append(f"steps[{i}] is not an object")
                continue
            if s.get("action") not in ACTIONS:
                errs.append(f"steps[{i}].action {s.get('action')!r} not in {ACTIONS}")
            ts, te = s.get("t_start"), s.get("t_end")
            if not all(isinstance(t, (int, float)) and not isinstance(t, bool) for t in (ts, te)):
                errs.append(f"steps[{i}] t_start/t_end must be numbers")
            elif ts < 0 or te < ts:
                errs.append(f"steps[{i}] needs 0 <= t_start <= t_end")
            elif duration_s and te > duration_s + slack:
                errs.append(f"steps[{i}].t_end {te} is past clip end {duration_s}")
            if not isinstance(s.get("object"), str):
                errs.append(f"steps[{i}].object must be a string")
    return errs


def normalise(d: dict) -> dict:
    d = dict(d)
    tc = d.get("task_completed")
    if isinstance(tc, str):
        d["task_completed"] = {"true": True, "false": False}.get(tc.strip().lower(), tc.strip().lower())
    return d


def use_vertex() -> bool:
    return os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true", "yes")


def _client():
    """AI Studio (GEMINI_API_KEY) or Vertex AI (GOOGLE_GENAI_USE_VERTEXAI=true; bills the
    Google Cloud project, so Cloud trial credits apply). Vertex auth: GOOGLE_CLOUD_API_KEY,
    or GOOGLE_CLOUD_PROJECT + GOOGLE_CLOUD_LOCATION with gcloud application-default login."""
    from google import genai
    if use_vertex():
        key = os.environ.get("GOOGLE_CLOUD_API_KEY")
        if key:
            return genai.Client(vertexai=True, api_key=key)
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise GeminiUnavailable("Vertex AI selected but GOOGLE_CLOUD_API_KEY / GOOGLE_CLOUD_PROJECT not set")
        return genai.Client(vertexai=True, project=project,
                            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"))
    keys = [k for k in (os.environ.get("GEMINI_API_KEY"), os.environ.get("GEMINI_API_KEY_1"), os.environ.get("GEMINI_API_KEY_2")) if k]
    if not keys:
        raise GeminiUnavailable("GEMINI_API_KEY is not set")
    global _WORKING_KEY
    if _WORKING_KEY in keys:
        return genai.Client(api_key=_WORKING_KEY)
    last = None
    for k in keys:                      # use the first key that can actually generate
        c = genai.Client(api_key=k)
        try:
            c.models.count_tokens(model=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL), contents="ok")
            c.models.generate_content(model=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL), contents="ok")
            _WORKING_KEY = k
            return c
        except Exception as e:          # 402 no credit / 403 blocked -> try the next key
            last = e
    raise last


_WORKING_KEY = None


INLINE_LIMIT = 18 * 1024 * 1024   # inline request cap is ~20 MB


class _Inline:
    """Video sent inline as bytes (works on AI Studio and Vertex); nothing to delete afterwards."""
    name = None

    def __init__(self, path: Path):
        from google.genai import types
        mime = "video/quicktime" if path.suffix.lower() == ".mov" else "video/mp4"
        self.part = types.Part.from_bytes(data=path.read_bytes(), mime_type=mime)


def upload_video(client, path: Path, timeout_s: float = 300):
    if path.stat().st_size <= INLINE_LIMIT or use_vertex():
        return _Inline(path)
    f = client.files.upload(file=str(path))
    t0 = time.time()
    while getattr(f.state, "name", str(f.state)) == "PROCESSING":
        if time.time() - t0 > timeout_s:
            raise FactSheetError(f"Gemini file processing timed out for {path.name}")
        time.sleep(2)
        f = client.files.get(name=f.name)
    if getattr(f.state, "name", str(f.state)) == "FAILED":
        raise FactSheetError(f"Gemini could not process {path.name}")
    return f


def generate_json(client, video_file, prompt: str, schema: dict) -> str:
    from google.genai import types
    resp = client.models.generate_content(
        model=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL),
        contents=[getattr(video_file, "part", video_file), prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=schema,
            temperature=0,
        ),
    )
    return resp.text or ""


def get_facts(path: str | Path, duration_s: float | None = None) -> dict:
    """Upload the clip, ask for the fact sheet, validate, retry once."""
    path = Path(path)
    client = _client()
    video = upload_video(client, path)
    try:
        prompt = PROMPT
        last_errs: list[str] = []
        for attempt in range(2):
            raw = generate_json(client, video, prompt, FACT_SCHEMA)
            try:
                d = normalise(json.loads(raw))
                last_errs = validate(d, duration_s)
            except json.JSONDecodeError as e:
                last_errs = [f"invalid JSON: {e}"]
            if not last_errs:
                return d
            prompt = (PROMPT + "\n\nYour previous answer was rejected for these reasons:\n- "
                      + "\n- ".join(last_errs) + "\nReturn corrected JSON only.")
        raise FactSheetError(f"{path.name}: fact sheet invalid after retry: {last_errs}")
    finally:
        if video.name:
            try:
                client.files.delete(name=video.name)
            except Exception:
                pass
