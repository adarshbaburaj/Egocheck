"""Live webcam mode: judge what the camera sees, every few seconds.

    python -m judge.live [--camera 0] [--window 3] [--record out/live.mp4] [--task "pick up the apple ..."]

Each window of frames gets the same local checks as the batch pipeline (blur,
brightness, shake) plus a System One (Jev) request on 3 frames, answered in the
background so the video stays smooth. The verdict overlay uses the same
decision code (judge/system_one.combine) and the same 0.80 routing rule.
Gemini is not used live (it takes ~8 s per clip); run the pipeline for fact sheets.
Keys: q / Esc quit, space pause judging.
"""
from __future__ import annotations

import argparse
import statistics
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from . import pipeline, rules_decider, system_one
from .local_checks import ANALYSIS_WIDTH, DARK_FRAME_MEAN, FLOW_WIDTH, _resize

ROUTE_BGR = {"keep": (89, 124, 74), "fix": (44, 125, 176), "discard": (63, 83, 180), "human_review": (166, 98, 106)}
ROUTE_LABEL = {"keep": "KEEP", "fix": "FIX", "discard": "DISCARD", "human_review": "HUMAN REVIEW"}
REASON = {"good": "Looks good", "blurry": "Too blurry", "dark": "Too dark", "shaky": "Too shaky",
          "hands_out_of_frame": "Hands out of frame", "object_hidden": "Apple hidden",
          "task_failed": "Task not finished", "multiple_tasks": "More than one task",
          "privacy": "Face or screen visible"}


def window_metrics(frames: list[np.ndarray]) -> dict:
    """Same metrics as local_checks.run, on an in-memory list of BGR frames."""
    blur, bright, flows, prev = [], [], [], None
    for f in frames[::3]:
        g = cv2.cvtColor(_resize(f, ANALYSIS_WIDTH), cv2.COLOR_BGR2GRAY)
        blur.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
        bright.append(float(g.mean()))
        s = _resize(g, FLOW_WIDTH)
        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(prev, s, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flows.append(float(np.linalg.norm(flow, axis=2).mean()))
        prev = s
    return {"blur_score": round(statistics.median(blur), 1), "brightness_mean": round(statistics.fmean(bright), 1),
            "dark_frame_ratio": round(sum(b < DARK_FRAME_MEAN for b in bright) / len(bright), 3),
            "shake_score": round(statistics.fmean(flows), 2) if flows else 0.0}


class Judge:
    """Runs one judgment at a time in a background thread."""

    def __init__(self, task: str, use_jev: bool):
        self.task, self.use_jev = task, use_jev
        self.result, self.busy, self.error = None, False, None
        self.tmp = Path(tempfile.mkdtemp(prefix="clipjudge_live_"))

    def submit(self, frames: list[np.ndarray]):
        if self.busy:
            return
        self.busy = True
        threading.Thread(target=self._run, args=(list(frames),), daemon=True).start()

    def _run(self, frames):
        t0 = time.perf_counter()
        try:
            local = window_metrics(frames)
            n = len(frames)
            paths = {}
            for name, frac in (("start", 0.1), ("middle", 0.5), ("end", 0.9)):
                f = frames[min(n - 1, int(n * frac))]
                h, w = f.shape[:2]
                s = min(1.0, 1280 / max(h, w))
                if s < 1:
                    f = cv2.resize(f, (round(w * s), round(h * s)))
                p = self.tmp / f"{name}.jpg"
                cv2.imwrite(str(p), f, [cv2.IMWRITE_JPEG_QUALITY, 85])
                paths[name] = str(p)
            state = {"local": local, "facts": {"task_label": self.task}, "frames": paths}
            if self.use_jev:
                fail = system_one.ask(state)
                d = system_one.combine(fail, local)
                checks, decider = d["failure_probs"], "system_one"
            else:
                d = rules_decider.decide({"local": local, "facts": None})
                checks, decider = d["severities"], "rules_fallback"
            probs = d["verdict"]["probs"]
            self.result = {"route": pipeline.route(probs), "choice": d["verdict"]["choice"],
                           "conf": max(probs.values()), "reason": d["main_reason"]["choice"],
                           "checks": checks, "local": local, "decider": decider,
                           "ms": round((time.perf_counter() - t0) * 1000)}
            self.error = None
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"[:90]
            pipeline.report_error(e, "live")
        finally:
            self.busy = False


CHECK_ROWS = [("Sharp", "blurry"), ("Well lit", "dark"), ("Steady", "shaky"), ("Hands in view", "hands_out_of_frame"),
              ("Apple in view", "object_hidden"), ("Task finished", "task_failed"), ("No faces/screens", "privacy")]


def draw(frame, judge: Judge, paused: bool, fps: float):
    h, w = frame.shape[:2]
    over = frame.copy()
    pw = 330
    cv2.rectangle(over, (w - pw - 20, 20), (w - 20, 20 + 360), (250, 250, 248), -1)
    cv2.addWeighted(over, 0.88, frame, 0.12, 0, frame)
    x, y = w - pw, 52
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "CLIP JUDGE  live", (x, y), font, 0.55, (110, 110, 105), 1, cv2.LINE_AA)
    r = judge.result
    if r:
        col = ROUTE_BGR[r["route"]]
        cv2.circle(frame, (x + 8, y + 36), 8, col, -1)
        cv2.putText(frame, ROUTE_LABEL[r["route"]], (x + 24, y + 44), font, 0.85, col, 2, cv2.LINE_AA)
        sub = f'{r["choice"]} {round(r["conf"] * 100)}%  -  {REASON.get(r["reason"], r["reason"])}'
        cv2.putText(frame, sub, (x, y + 72), font, 0.48, (60, 60, 58), 1, cv2.LINE_AA)
        yy = y + 104
        for label, key in CHECK_ROWS:
            sev = r["checks"].get(key)
            st = (180, 180, 175) if sev is None else (89, 124, 74) if sev < 0.34 else (44, 125, 176) if sev < 0.67 else (63, 83, 180)
            cv2.circle(frame, (x + 6, yy - 5), 6, st, -1)
            cv2.putText(frame, label, (x + 20, yy), font, 0.48, (40, 40, 38), 1, cv2.LINE_AA)
            val = "not checked" if sev is None else "ok" if sev < 0.34 else "unsure" if sev < 0.67 else "problem"
            cv2.putText(frame, val, (x + pw - 100, yy), font, 0.45, (110, 110, 105), 1, cv2.LINE_AA)
            yy += 26
        foot = f'{"Jev" if r["decider"] == "system_one" else "rules"} {r["ms"]} ms  |  {fps:.0f} fps'
        cv2.putText(frame, foot, (x, 20 + 345), font, 0.42, (130, 130, 125), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "watching...", (x, y + 44), font, 0.7, (110, 110, 105), 1, cv2.LINE_AA)
    if judge.busy:
        cv2.circle(frame, (w - 40, 40), 5, (110, 110, 105), -1)
    if judge.error:
        cv2.putText(frame, judge.error, (24, h - 24), font, 0.45, (63, 83, 180), 1, cv2.LINE_AA)
    if paused:
        cv2.putText(frame, "PAUSED (space)", (24, 40), font, 0.6, (63, 83, 180), 2, cv2.LINE_AA)
    return frame


def main(argv=None):
    ap = argparse.ArgumentParser(prog="judge.live")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--window", type=float, default=3.0, help="seconds per judgment")
    ap.add_argument("--task", default="pick up the apple and put it on the plate")
    ap.add_argument("--record", help="also save the annotated stream to this .mp4")
    ap.add_argument("--rules", action="store_true", help="skip Jev, use local rules only")
    a = ap.parse_args(argv)
    pipeline.init_sentry()

    cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera. On macOS allow camera access for your terminal / VS Code "
                         "(System Settings > Privacy & Security > Camera), then retry.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    use_jev = not a.rules and bool(system_one._api_key())
    judge = Judge(a.task, use_jev)
    buf: deque = deque(maxlen=int(a.window * 30))
    last, paused, writer = time.time(), False, None
    tick, fps = time.time(), 30.0
    print(f"live: judging every {a.window}s with {'Jev' if use_jev else 'local rules'} - q to quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        buf.append(frame.copy())
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(1e-3, now - tick)
        tick = now
        if not paused and now - last >= a.window and len(buf) > 10:
            judge.submit(buf)
            last = now
        out = draw(frame, judge, paused, fps)
        if a.record:
            if writer is None:
                h, w = out.shape[:2]
                Path(a.record).parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(a.record, cv2.VideoWriter_fourcc(*"avc1"), 30, (w, h))
            writer.write(out)
        cv2.imshow("Clip Judge - live", out)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord(" "):
            paused = not paused
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
