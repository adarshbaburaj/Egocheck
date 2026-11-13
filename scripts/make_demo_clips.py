"""Generate synthetic stand-in clips (H.264 .mp4) for demos and tests.

A crude top-down "egocentric" scene: textured table, a hand blob moves to a
cup, carries it to a plate, and pulls back. Variants degrade it in known ways.
These are placeholders — real clips go in clips/.

    python scripts/make_demo_clips.py clips/ [--labels labels.csv]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

W, H, FPS, SECONDS = 640, 360, 30, 4


def _table(rng: np.random.Generator) -> np.ndarray:
    base = np.full((H, W, 3), (70, 110, 150), np.uint8)
    yy, xx = np.mgrid[0:H, 0:W]
    checker = (((xx // 40) + (yy // 40)) % 2).astype(np.uint8) * 30
    base = cv2.add(base, cv2.merge([checker] * 3))
    noise = rng.integers(0, 50, (H, W, 1), dtype=np.uint8).repeat(3, axis=2)
    return cv2.add(base, noise)


def _frame(t: float, table: np.ndarray) -> np.ndarray:
    img = table.copy()
    plate, cup0 = (470, 200), (170, 210)
    cv2.circle(img, plate, 55, (235, 235, 235), -1)
    cv2.circle(img, plate, 55, (60, 60, 60), 2)
    # phases: 0-1s reach, 1-1.4 grasp, 1.4-3 carry, 3-4 release+retreat
    home = np.array([320, 380])
    if t < 1:
        hand = home + (np.array(cup0) - home) * t
        cup = np.array(cup0)
    elif t < 1.4:
        hand = cup = np.array(cup0)
    elif t < 3:
        a = (t - 1.4) / 1.6
        cup = np.array(cup0) + (np.array(plate) - np.array(cup0)) * a
        hand = cup
    else:
        cup = np.array(plate)
        hand = cup + (home - cup) * (t - 3)
    cv2.circle(img, tuple(int(v) for v in cup), 28, (40, 40, 200), -1)
    cv2.circle(img, tuple(int(v) for v in cup), 28, (10, 10, 60), 3)
    hx, hy = (int(v) for v in hand)
    cv2.ellipse(img, (hx, hy + 45), (40, 60), 0, 0, 360, (120, 160, 210), -1)
    for k in range(4):
        cv2.line(img, (hx - 30 + 20 * k, hy + 10), (hx - 30 + 20 * k, hy - 25), (110, 150, 200), 10)
    return img


def make(path: Path, variant: str = "good", seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    table = _table(rng)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), FPS, (W, H))
    if not w.isOpened():
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(FPS * SECONDS):
        img = _frame(i / FPS, table)
        if variant == "blurry":
            img = cv2.GaussianBlur(img, (0, 0), 6)
        elif variant == "soft":                       # borderline blur
            img = cv2.GaussianBlur(img, (0, 0), 1.0)
        elif variant == "dark":
            img = (img * 0.3).astype(np.uint8)
        elif variant == "shaky":
            dx, dy = rng.integers(-60, 61, 2)
            m = np.float32([[1, 0, dx], [0, 1, dy]])
            img = cv2.warpAffine(img, m, (W, H), borderMode=cv2.BORDER_REFLECT)
        w.write(img)
    w.release()


DEMO = [  # file, variant, seed, hand label
    ("demo_01_good.mp4", "good", 1, "keep"),
    ("demo_02_good.mp4", "good", 2, "keep"),
    ("demo_03_blurry.mp4", "blurry", 3, "discard"),
    ("demo_04_dark.mp4", "dark", 4, "fix"),
    ("demo_05_shaky.mp4", "shaky", 5, "discard"),
    ("demo_06_soft.mp4", "soft", 6, "keep"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="clips")
    ap.add_argument("--labels", default=None, help="also write a labels.csv here")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, variant, seed, _ in DEMO:
        make(out / name, variant, seed)
        print("wrote", out / name)
    if a.labels:
        with open(a.labels, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["clip", "my_verdict"])
            wr.writerows((n, lab) for n, _, _, lab in DEMO)
        print("wrote", a.labels)


if __name__ == "__main__":
    main()
