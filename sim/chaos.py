"""The "garbage out" replay: a Shadow Hand whose policy learned from bad clips.

    .venv/bin/python -m sim.chaos [--out out/sim/chaos.mp4]

For the film, not for the pipeline. Same scene as sim/replay.py, but the hand twitches on its
way to the apple, grabs it, winds up and throws it straight at the camera. The throw is real
physics: at release the apple gets the ballistic velocity that reaches the camera lens, then
MuJoCo flies it. The clip ends a frame before the apple hits the lens; the film adds the crack.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mujoco  # noqa: E402
import replay as R  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CAM = dict(lookat=np.array([0.0, -0.06, 0.24]), distance=1.0, azimuth=250.0, elevation=-10.0)
FLIGHT = 0.30          # seconds from release to the lens


def cam_pos():
    a, e = np.radians(CAM["azimuth"]), np.radians(CAM["elevation"])
    fwd = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    return CAM["lookat"] - CAM["distance"] * fwd


class Chaos(R.Replayer):
    def __init__(self):
        super().__init__(R.DEFAULT_STEPS, side=False)
        apple, p = R.A, R.palm
        cam = cam_pos()
        toward = cam[:2] - R.A[:2]; toward = np.r_[toward / np.linalg.norm(toward), 0]
        home = p(apple) + np.array([0, -0.12, 0.16])
        lifted = p(apple) + np.array([0, 0, 0.16])
        windup = lifted - 0.13 * toward + np.array([0, 0, 0.07])
        follow = lifted + 0.14 * toward + np.array([0, 0, 0.02])
        self.seq = [
            (0.0, home, "pre", "policy: trained on bad data"),
            (0.9, p(apple) + np.array([0.05, -0.03, 0.09]), "pre", "move to apple?"),
            (1.3, p(apple) + np.array([-0.04, 0.02, 0.06]), "pre", "move to apple??"),
            (1.7, p(apple), "pre", "move to apple"),
            (2.2, p(apple), "closed", "grasp"),
            (2.7, lifted, "closed", "lift"),
            (3.2, windup, "closed", "wind up"),
            (3.36, follow, "open", "THROW"),
            (4.2, follow, "open", "THROW"),
        ]
        self.T = self.seq[-1][0]
        self.release_t = 3.27
        self.thrown = False
        self.cam = cam
        self.apple_geoms = [g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == self.apple]

    def step(self):
        t = self.d.time
        if not self.thrown and t >= self.release_t:
            self._throw()
        target, fingers, self.label = R._interp(self.seq, t)
        if t < 1.75:  # twitchy approach: the policy has seen shaky, half-finished clips
            target = target + 0.014 * np.array([np.sin(41 * t), np.sin(29 * t + 1), 0.6 * np.sin(53 * t + 2)])
        self.d.mocap_pos[0] = target - R.PALM_OFFSET
        goal = 1.0 if fingers == "closed" else R.PRESHAPE if fingers == "pre" else 0.0
        self.grip += np.clip(goal - self.grip, -0.012, 0.012)
        for name, q in R.CLOSED.items():
            self.d.ctrl[self.act[name]] = q * self.grip
        self._hold(fingers == "closed" and self.grip > 0.8)
        mujoco.mj_step(self.m, self.d)

    def _throw(self):
        self.thrown = True
        self.held = None
        for g in self.apple_geoms:          # nothing in its way now: fly
            self.m.geom_contype[g] = 0; self.m.geom_conaffinity[g] = 0
        jnt = self.m.body_jntadr[self.apple]
        vadr = self.m.jnt_dofadr[jnt]
        p0 = self.d.xpos[self.apple].copy()
        aim = self.cam + np.array([0.0, 0.0, -0.01])
        v = (aim - p0) / FLIGHT + np.array([0, 0, 0.5 * 9.81 * FLIGHT])
        self.d.qvel[vadr:vadr + 3] = v
        self.d.qvel[vadr + 3:vadr + 6] = [14.0, -9.0, 6.0]   # tumble

    def _hold(self, on):
        super()._hold(on and not self.thrown)

    @property
    def done(self):
        if self.thrown and np.linalg.norm(self.d.xpos[self.apple] - self.cam) < R.scene.APPLE_R + 0.05:
            return True
        return self.d.time >= self.T



def render(out: Path, width=1920, height=1080, fps=30):
    import cv2
    r = Chaos()
    rend = mujoco.Renderer(r.m, height, width)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = CAM["lookat"]; cam.distance = CAM["distance"]; cam.azimuth = CAM["azimuth"]; cam.elevation = CAM["elevation"]
    out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))
    spf = int(round(1 / fps / r.m.opt.timestep))
    frames = 0
    while not r.done:
        for _ in range(spf):
            r.step()
            if r.done:
                break
        rend.update_scene(r.d, cam)
        frame = rend.render()[:, :, ::-1].copy()
        R._overlay(frame, r.label, None)
        vw.write(frame); frames += 1
    vw.release()
    return {"video": str(out), "seconds": round(frames / fps, 2), "thrown": r.thrown,
            "apple_to_lens_m": round(float(np.linalg.norm(r.d.xpos[r.apple] - r.cam)), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "out" / "sim" / "chaos.mp4"))
    a = ap.parse_args()
    print(json.dumps(render(Path(a.out)), indent=1))


if __name__ == "__main__":
    main()
