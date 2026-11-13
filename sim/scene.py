"""MuJoCo scene: a Shadow Hand (DeepMind Menagerie) on a kinematic forearm,
a table, an apple and a plate.

The forearm is attached to a mocap body, so we drive the hand along a scripted
path without arm inverse kinematics. Fingers are physical position actuators.
A weld constraint ("grasp assist") holds the apple once the fingers close —
an honest replay aid, not learned grasping.
"""
from __future__ import annotations

from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parent
HAND_XML = ROOT / "menagerie" / "shadow_hand" / "right_hand.xml"

APPLE_POS = (0.0, -0.12, 0.045)   # on the table, z = radius
PLATE_POS = (0.0, 0.16, 0.006)
APPLE_R = 0.040

SCENE = f"""
<mujoco model="clip-judge apple replay">
  <option timestep="0.002" cone="elliptic" impratio="10" integrator="implicitfast"/>
  <visual>
    <global azimuth="135" elevation="-28" offwidth="3840" offheight="2160"/>
    <quality shadowsize="4096" offsamples="8"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.55 0.55 0.55" specular="0.1 0.1 0.1"/>
    <rgba haze="0.96 0.96 0.95 1"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="flat" rgb1="0.97 0.97 0.96" rgb2="0.97 0.97 0.96" width="32" height="32"/>
    <texture name="wood" type="2d" builtin="flat" rgb1="0.86 0.80 0.71" width="64" height="64"/>
    <material name="table" texture="wood" rgba="0.86 0.80 0.71 1" reflectance="0.05"/>
    <material name="apple" rgba="0.78 0.10 0.10 1" specular="0.6" shininess="0.6"/>
    <material name="stem" rgba="0.35 0.22 0.10 1"/>
    <material name="leaf" rgba="0.25 0.55 0.20 1"/>
    <material name="plate" rgba="0.98 0.98 0.97 1" specular="0.4" shininess="0.8"/>
    <material name="floor" rgba="0.93 0.93 0.92 1"/>
  </asset>
  <worldbody>
    <light pos="0.4 -0.4 1.4" dir="-0.3 0.3 -1" directional="true" castshadow="true" diffuse="0.6 0.6 0.6"/>
    <geom name="floor" type="plane" pos="0 0 -0.75" size="3 3 0.1" material="floor"/>
    <geom name="table" type="box" pos="0 0 -0.02" size="0.45 0.45 0.02" material="table" friction="1 0.01 0.001"/>
    <body name="plate" pos="{PLATE_POS[0]} {PLATE_POS[1]} {PLATE_POS[2]}">
      <geom type="cylinder" size="0.09 0.006" material="plate"/>
      <geom type="cylinder" size="0.10 0.003" pos="0 0 0.004" material="plate" contype="0" conaffinity="0"/>
    </body>
    <body name="apple" pos="{APPLE_POS[0]} {APPLE_POS[1]} {APPLE_POS[2]}">
      <freejoint name="apple"/>
      <geom name="apple" type="sphere" size="{APPLE_R}" mass="0.18" material="apple"
            condim="6" friction="1.2 0.02 0.002" priority="1"/>
      <geom type="capsule" fromto="0 0 0.035 0.004 0 0.055" size="0.003" material="stem" contype="0" conaffinity="0"/>
      <geom type="ellipsoid" pos="0.014 0 0.05" size="0.012 0.005 0.002" euler="0 0.4 0" material="leaf" contype="0" conaffinity="0"/>
    </body>
    <body name="hand_base" mocap="true" pos="0 -0.45 0.35"/>
  </worldbody>
</mujoco>
"""


def build() -> tuple[mujoco.MjModel, mujoco.MjData]:
    spec = mujoco.MjSpec.from_string(SCENE)
    hand = mujoco.MjSpec.from_file(str(HAND_XML))
    base = spec.body("hand_base")
    frame = base.add_frame()
    # forearm points along -y toward the table, palm facing down
    frame.quat = [0.5, -0.5, 0.5, 0.5]
    frame.attach_body(hand.body("rh_forearm"), "", "")
    model = spec.compile()
    return model, mujoco.MjData(model)
