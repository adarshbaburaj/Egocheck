"""Replay a kept clip's manipulation steps on a simulated Shadow Hand.

    python -m sim.replay <clip-name-or-card.json>   [--out out/sim/<clip>.mp4] [--live]

Steps come from the clip's Gemini fact sheet (move_to / grasp / lift / place /
release with timestamps). The scene layout is fixed (apple, plate); only the
order and timing come from the video. Grasping uses a weld "assist" once the
fingers close. --live opens the MuJoCo viewer (on macOS run with `mjpython`).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message=".*empty prefix.*")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mujoco  # noqa: E402
import scene   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PALM_OFFSET = np.array([0.0, 0.01, -0.247])        # mocap base -> palm (from model)
A = np.array(scene.APPLE_POS)
P = np.array(scene.PLATE_POS)
APPLE_ON_PLATE_Z = P[2] + 0.006 + scene.APPLE_R
GRASP_DX, GRASP_DY, GRASP_DZ = -0.025, -0.052, 0.112   # palm relative to apple centre (measured)
PRESHAPE = 0.45                                          # half-curled while approaching (clears the table)

# Side grasp (default for replays), like a person's right hand in the egocentric clips: hand on its side,
# thumb up, fingers pointing along +y toward the plate, palm facing -x (away from the camera), wrapping
# the apple's far side; it reaches in from the camera side (+x). Orientation found by searching axis-aligned
# rotations; palm offset by a physics grid search (no table contact, apple moves < 1.5 cm on close).
SIDE_QUAT = np.array([-0.5, -0.5, 0.5, -0.5])
SIDE_PALM_OFFSET = np.array([-0.01, 0.247, 0.0])           # mocap base -> palm at SIDE_QUAT
SIDE_GRASP = np.array([0.060, -0.075, 0.0])            # palm relative to apple centre

OPEN = {}
CLOSED = {  # actuator targets (radians) for a wrap grasp
    "rh_A_THJ5": 0.55, "rh_A_THJ4": 1.15, "rh_A_THJ2": 0.30, "rh_A_THJ1": 0.85,
    "rh_A_FFJ3": 1.05, "rh_A_FFJ0": 1.6, "rh_A_MFJ3": 1.05, "rh_A_MFJ0": 1.6,
    "rh_A_RFJ3": 1.05, "rh_A_RFJ0": 1.6, "rh_A_LFJ3": 1.05, "rh_A_LFJ0": 1.6, "rh_A_LFJ5": 0.3,
}

DEFAULT_STEPS = [
    {"t_start": 0.0, "t_end": 1.4, "action": "move_to", "object": "apple"},
    {"t_start": 1.4, "t_end": 2.1, "action": "grasp", "object": "apple"},
    {"t_start": 2.1, "t_end": 2.9, "action": "lift", "object": "apple"},
    {"t_start": 2.9, "t_end": 4.3, "action": "place", "object": "plate"},
    {"t_start": 4.3, "t_end": 5.0, "action": "release", "object": "apple"},
]


def palm(apple_xyz) -> np.ndarray:
    return np.asarray(apple_xyz) + np.array([GRASP_DX, GRASP_DY, GRASP_DZ])


def keyframes(steps: list[dict]) -> list[tuple[float, np.ndarray, str, str]]:
    """Turn fact-sheet steps into (time, palm target, finger state, label) keyframes."""
    home = palm(A) + np.array([0, -0.12, 0.16])
    above_apple = palm(A) + np.array([0, -0.02, 0.10])
    lifted = palm(A) + np.array([0, 0, 0.14])
    above_plate = palm([P[0], P[1], APPLE_ON_PLATE_Z]) + np.array([0, 0, 0.12])
    on_plate = palm([P[0], P[1], APPLE_ON_PLATE_Z + 0.004])
    retreat = on_plate + np.array([0, -0.10, 0.16])

    seq = [(0.0, home, "pre", "start")]
    t = 0.6
    for s in steps:
        dur = max(0.5, min(2.5, float(s["t_end"]) - float(s["t_start"])))
        a = s["action"]
        if a == "move_to" and len(seq) < 3:
            seq += [(t + dur * 0.6, above_apple, "pre", "move to apple"), (t + dur, palm(A), "pre", "move to apple")]
        elif a == "move_to":
            seq += [(t + dur, above_plate, "closed", "move to plate")]
        elif a == "grasp":
            seq += [(t + dur, palm(A), "closed", "grasp")]
        elif a == "lift":
            seq += [(t + dur, lifted, "closed", "lift")]
        elif a == "place":
            seq += [(t + dur * 0.6, above_plate, "closed", "carry"), (t + dur, on_plate, "closed", "place")]
        elif a == "release":
            seq += [(t + dur * 0.5, on_plate, "open", "release")]
        t += dur
    seq += [(t + 1.0, retreat, "open", "done"), (t + 1.8, retreat, "open", "done")]
    return seq


def side_palm(apple_xyz) -> np.ndarray:
    return np.asarray(apple_xyz) + SIDE_GRASP


def keyframes_side(steps: list[dict]) -> list[tuple[float, np.ndarray, str, str]]:
    """Sideways pick: slide in from the side, wrap, lift, carry along the fingers to the plate, set down, slide out."""
    g = side_palm(A)
    home = g + np.array([0.18, -0.04, 0.10])
    approach = g + np.array([0.06, 0.0, 0.02])
    lifted = g + np.array([0, 0, 0.12])
    over_plate = side_palm([P[0], P[1], APPLE_ON_PLATE_Z]) + np.array([0, 0, 0.12])
    on_plate = side_palm([P[0], P[1], APPLE_ON_PLATE_Z + 0.004])
    retreat = on_plate + np.array([0.04, -0.24, 0.14])
    seq = [(0.0, home, "pre", "start")]
    t = 0.4
    for st in steps:
        dur = max(0.5, min(1.6, float(st["t_end"]) - float(st["t_start"])))
        a = st["action"]
        if a == "move_to" and len(seq) < 3:
            seq += [(t + dur * 0.6, approach, "pre", "move to apple"), (t + dur, g, "pre", "move to apple")]
        elif a == "grasp":
            seq += [(t + dur, g, "closed", "grasp")]
        elif a == "lift":
            seq += [(t + dur, lifted, "closed", "lift")]
        elif a in ("place", "move_to"):
            seq += [(t + dur * 0.55, over_plate, "closed", "carry"), (t + dur, on_plate, "closed", "place")]
        elif a == "release":
            seq += [(t + dur * 0.6, on_plate, "open", "release")]
        t += dur
    seq += [(t + 0.7, retreat, "open", "done"), (t + 1.2, retreat, "open", "done")]
    return seq


def _interp(seq, t):
    for (t0, p0, f0, l0), (t1, p1, f1, l1) in zip(seq, seq[1:]):
        if t0 <= t <= t1:
            k = (t - t0) / max(1e-6, t1 - t0)
            k = k * k * (3 - 2 * k)                 # smoothstep
            return p0 + (p1 - p0) * k, f1, l1
    return seq[-1][1], seq[-1][2], seq[-1][3]


class Replayer:
    def __init__(self, steps, side: bool = True):
        self.m, self.d = scene.build()
        self.side = side
        self.quat = SIDE_QUAT if side else np.array([1.0, 0, 0, 0])
        self.off = SIDE_PALM_OFFSET if side else PALM_OFFSET
        self.seq = (keyframes_side if side else keyframes)(steps or DEFAULT_STEPS)
        self.T = self.seq[-1][0]
        self.held = None
        self.apple = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "apple")
        self.palm = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "rh_palm")
        self.act = {self.m.actuator(i).name: i for i in range(self.m.nu)}
        self.grip = 0.0
        self.label = "start"
        self.d.mocap_quat[0] = self.quat
        self.d.mocap_pos[0] = self.seq[0][1] - self.off
        mujoco.mj_forward(self.m, self.d)

    def _hold(self, on: bool):
        """Grasp assist: while the hand is closed, the apple keeps the pose it had
        relative to the palm at the moment of grasp (kinematic attach)."""
        jnt = self.m.body_jntadr[self.apple]
        qadr, vadr = self.m.jnt_qposadr[jnt], self.m.jnt_dofadr[jnt]
        pp, qp = self.d.xpos[self.palm].copy(), self.d.xquat[self.palm].copy()
        if on and self.held is None:
            qp_inv = np.zeros(4); mujoco.mju_negQuat(qp_inv, qp)
            rel_p = np.zeros(3); mujoco.mju_rotVecQuat(rel_p, self.d.xpos[self.apple] - pp, qp_inv)
            rel_q = np.zeros(4); mujoco.mju_mulQuat(rel_q, qp_inv, self.d.xquat[self.apple])
            self.held = (rel_p, rel_q)
        elif not on:
            self.held = None
        if self.held is not None:
            rel_p, rel_q = self.held
            wp = np.zeros(3); mujoco.mju_rotVecQuat(wp, rel_p, qp)
            wq = np.zeros(4); mujoco.mju_mulQuat(wq, qp, rel_q)
            self.d.qpos[qadr:qadr + 3] = pp + wp
            self.d.qpos[qadr + 3:qadr + 7] = wq
            self.d.qvel[vadr:vadr + 6] = 0

    def step(self):
        t = self.d.time
        target, fingers, self.label = _interp(self.seq, t)
        self.d.mocap_pos[0] = target - self.off
        goal = 1.0 if fingers == "closed" else PRESHAPE if fingers == "pre" else 0.0
        self.grip += np.clip(goal - self.grip, -0.008, 0.008)          # smooth finger motion
        for name, q in CLOSED.items():
            self.d.ctrl[self.act[name]] = q * self.grip
        self._hold(fingers == "closed" and self.grip > 0.8)
        mujoco.mj_step(self.m, self.d)

    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        self.grip, self.held, self.label = 0.0, None, "start"
        self.d.mocap_quat[0] = self.quat
        self.d.mocap_pos[0] = self.seq[0][1] - self.off
        mujoco.mj_forward(self.m, self.d)

    @property
    def done(self):
        return self.d.time >= self.T


def render(steps, out: Path, width=1920, height=1080, fps=30, title: str | None = None):
    import cv2
    r = Replayer(steps)
    rend = mujoco.Renderer(r.m, height, width)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [-0.01, 0.03, 0.02]; cam.distance = 0.70; cam.azimuth = 180; cam.elevation = -50   # first-person, like the clips
    out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))
    steps_per_frame = int(round(1 / fps / r.m.opt.timestep))
    while not r.done:
        for _ in range(steps_per_frame):
            r.step()
        cam.azimuth = 180 + 2.0 * np.sin(1.3 * r.d.time)                   # slight head sway
        rend.update_scene(r.d, cam)
        frame = rend.render()[:, :, ::-1].copy()
        _overlay(frame, r.label, title)
        vw.write(frame)
    vw.release()
    on_plate = np.linalg.norm(r.d.xpos[r.apple][:2] - P[:2]) < 0.07
    return {"video": str(out), "seconds": round(r.T, 2), "apple_on_plate": bool(on_plate),
            "apple_final": r.d.xpos[r.apple].round(3).tolist()}


def _overlay(frame, label, title):
    import cv2
    h, w = frame.shape[:2]
    font, sc, th = cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
    (tw, tht), _ = cv2.getTextSize(label, font, sc, th)
    cv2.rectangle(frame, (32, h - 48 - tht - 16), (32 + tw + 32, h - 48), (255, 255, 255), -1)
    cv2.putText(frame, label, (48, h - 48 - 12), font, sc, (35, 35, 33), th, cv2.LINE_AA)
    if title:
        cv2.putText(frame, title, (36, 60), font, 0.8, (90, 90, 86), 2, cv2.LINE_AA)


def live(steps):
    import mujoco.viewer
    r = Replayer(steps)
    with mujoco.viewer.launch_passive(r.m, r.d) as v:
        v.cam.lookat[:] = [0.0, 0.02, 0.09]; v.cam.distance = 0.78; v.cam.azimuth = 200; v.cam.elevation = -22
        while v.is_running():
            t0 = time.time()
            r.step()
            if r.done:                       # loop: restart the replay on the same model
                r.reset()
            v.sync()
            time.sleep(max(0, r.m.opt.timestep - (time.time() - t0)))


def steps_for(arg: str | None) -> tuple[list[dict], str]:
    if not arg:
        return DEFAULT_STEPS, "default"
    p = Path(arg)
    card = p if p.suffix == ".json" else ROOT / "out" / "cards" / f"{Path(arg).stem}.json"
    if card.exists():
        facts = json.loads(card.read_text()).get("facts") or {}
        if facts.get("steps"):
            return facts["steps"], card.stem
    return DEFAULT_STEPS, Path(arg).stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", nargs="?", help="clip name or out/cards/<clip>.json")
    ap.add_argument("--out", help="mp4 path (default out/sim/<clip>.mp4)")
    ap.add_argument("--live", action="store_true", help="open the MuJoCo viewer (macOS: run with mjpython)")
    a = ap.parse_args()
    steps, name = steps_for(a.clip)
    if a.live:
        live(steps)
        return
    out = Path(a.out) if a.out else ROOT / "out" / "sim" / f"{name}.mp4"
    print(json.dumps(render(steps, out, title=f"Sim replay · {name}"), indent=1))


if __name__ == "__main__":
    main()
