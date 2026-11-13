"""Step 1 — local checks. OpenCV only, no AI, no network.

All image metrics are computed on frames resized to ANALYSIS_WIDTH so that
thresholds mean the same thing for 720p phone clips and 4K head-cam clips.
"""
from __future__ import annotations

import statistics
from pathlib import Path

import cv2
import numpy as np

SAMPLE_EVERY = 10          # analyse every 10th frame
ANALYSIS_WIDTH = 640       # px; frames are downscaled to this width first
DARK_FRAME_MEAN = 40.0     # a sampled frame with mean gray below this is "dark"
FLOW_WIDTH = 320           # optical flow runs on an even smaller frame (speed)


def _resize(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == width:
        return frame
    return cv2.resize(frame, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)


def run(path: str | Path) -> dict:
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {path.name}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    blur, bright, flows = [], [], []
    prev_small = None
    n_frames = 0
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        n_frames += 1
        if idx % SAMPLE_EVERY == 0:
            ok, frame = cap.retrieve()
            if ok and frame is not None:
                gray = cv2.cvtColor(_resize(frame, ANALYSIS_WIDTH), cv2.COLOR_BGR2GRAY)
                blur.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
                bright.append(float(gray.mean()))
                small = _resize(gray, FLOW_WIDTH)
                if prev_small is not None and prev_small.shape == small.shape:
                    flow = cv2.calcOpticalFlowFarneback(prev_small, small, None,
                                                        0.5, 3, 15, 3, 5, 1.2, 0)
                    mag = np.linalg.norm(flow, axis=2)
                    flows.append(float(mag.mean()))
                prev_small = small
        idx += 1
    cap.release()

    if not blur:
        raise RuntimeError(f"no decodable frames in {path.name}")

    duration = n_frames / fps if fps else 0.0
    return {
        "duration_s": round(duration, 2),
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "frames": n_frames,
        "blur_score": round(statistics.median(blur), 1),
        "brightness_mean": round(statistics.fmean(bright), 1),
        "dark_frame_ratio": round(sum(b < DARK_FRAME_MEAN for b in bright) / len(bright), 3),
        "shake_score": round(statistics.fmean(flows), 2) if flows else 0.0,
    }


if __name__ == "__main__":
    import json
    import sys

    for p in sys.argv[1:]:
        print(p, json.dumps(run(p)))


def keyframes(path: str | Path, out_dir: str | Path, max_side: int = 1280) -> dict[str, str]:
    """Save JPEG frames at 10% / 50% / 90% of the clip for System One. Sides kept within 280..max_side."""
    path, out_dir = Path(path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    out = {}
    for name, frac in (("start", 0.1), ("middle", 0.5), ("end", 0.9)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(n - 1, int(n * frac))))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        s = min(1.0, max_side / max(h, w))
        if min(h, w) * s < 280:
            s = 280 / min(h, w)
        if s != 1.0:
            frame = cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        p = out_dir / f"{path.stem}_{name}.jpg"
        cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        out[name] = str(p)
    cap.release()
    return out
