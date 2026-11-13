"""Baseline: the same keep/fix/discard decision made by Gemini alone, straight
from the video, with no local checks and no System One. Used for the
speed/agreement comparison on the dashboard.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .gemini_facts import FactSheetError, _client, generate_json, upload_video
from .system_one import REASONS, VERDICTS

SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": VERDICTS},
        "main_reason": {"type": "string", "enum": REASONS},
        "probs": {
            "type": "object",
            "properties": {v: {"type": "number"} for v in VERDICTS},
            "required": VERDICTS,
        },
    },
    "required": ["verdict", "main_reason", "probs"],
}

PROMPT = f"""You are the quality gate for egocentric video used to train robot manipulation policies.
Watch this clip and decide whether it should be kept, fixed, or discarded.
Consider blur, darkness, camera shake, whether hands and the manipulated object stay in frame,
whether the task is completed, whether there is exactly one task, and privacy (faces/screens).

Return JSON: verdict in {VERDICTS}, main_reason in {REASONS},
and probs: your probability for each verdict (they must sum to 1)."""


def decide(path: str | Path) -> dict:
    """Returns {"verdict": {...}, "main_reason": {...}, "seconds": float}."""
    path = Path(path)
    client = _client()
    video = upload_video(client, path)
    t0 = time.perf_counter()   # time the decision only, not the upload
    try:
        for _ in range(2):
            try:
                d = json.loads(generate_json(client, video, PROMPT, SCHEMA))
            except json.JSONDecodeError:
                continue
            probs = d.get("probs") or {}
            if d.get("verdict") in VERDICTS and d.get("main_reason") in REASONS and \
                    all(isinstance(probs.get(v), (int, float)) for v in VERDICTS):
                z = sum(probs[v] for v in VERDICTS) or 1.0
                return {
                    "verdict": {"choice": d["verdict"],
                                "probs": {v: round(probs[v] / z, 4) for v in VERDICTS}},
                    "main_reason": {"choice": d["main_reason"]},
                    "seconds": round(time.perf_counter() - t0, 3),
                }
        raise FactSheetError(f"{path.name}: baseline answer invalid after retry")
    finally:
        if video.name:
            try:
                client.files.delete(name=video.name)
            except Exception:
                pass
