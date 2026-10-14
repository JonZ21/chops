"""Build the 'training_grip' keyframe for models/training_chopsticks.xml.

  1. Work out the finger joint angles for TARGET: run the hand with the chopsticks' collisions
     switched off, so the fingers can move freely into position.
  2. Restart with the hand already in that pose and the chopsticks closed (hinge at 0).
  3. The hinge spring opens the upper stick until it meets the index and middle fingertips.
     Let it settle, then save the pose + finger targets as a keyframe.

Change TARGET, rerun, then check with test_grip.py.
Run:  python make_grip.py
"""
import os
import numpy as np
import mujoco

MODEL = "models/training_chopsticks.xml"
KEYFILE = "models/grip_keyframe.xml"

# Finger targets in radians, by actuator name (without the "rh_A_" prefix). Anything missing is 0.
# FFJ0/MFJ0/RFJ0/LFJ0 drive a finger's two outer joints together (0 to 3.14).
TARGET = {
    "WRJ2": 0.0,  # wrist side-to-side
    "WRJ1": 0.0,  # wrist up/down
    "THJ5": 0.0308,  # thumb
    "THJ4": 1.08,
    "THJ3": 0.1604,
    "THJ2": 0.5839,
    "THJ1": 0.3069,
    "FFJ4": -0.3013,  # index (presses the upper stick)
    "FFJ3": 0.9,
    "FFJ0": 1.0,
    "MFJ3": 0.6,
    "MFJ0": 0.6,
    "RFJ3": 0.9,
    "RFJ0": 0.8,
    "LFJ5": 0.3,  # little (curled out of the way)
    "LFJ3": 1.1,
    "LFJ0": 1.1,
}

if not os.path.exists(KEYFILE):              # the scene includes this file, so it must exist
    open(KEYFILE, "w").write("<mujoco>\n  <keyframe/>\n</mujoco>\n")

m = mujoco.MjModel.from_xml_path(MODEL)
d = mujoco.MjData(m)
names = [m.actuator(i).name.replace("rh_A_", "") for i in range(m.nu)]
assert not set(TARGET) - set(names), f"Unknown actuator names: {set(TARGET) - set(names)}"
target = np.array([TARGET.get(n, 0.0) for n in names])
hand_joints = [j for j in range(m.njnt) if m.joint(j).name.startswith("rh_")]
hand_qpos = [m.jnt_qposadr[j] for j in hand_joints]

# 1. Fingers into position with the chopsticks "ghosted"
stick_geoms = [m.geom("chopstick_a").id, m.geom("chopstick_b").id]
saved = [(m.geom_contype[g], m.geom_conaffinity[g]) for g in stick_geoms]
for g in stick_geoms:
    m.geom_contype[g] = m.geom_conaffinity[g] = 0
mujoco.mj_resetData(m, d)
d.ctrl[:] = target
while d.time < 1.0:
    mujoco.mj_step(m, d)
finger_pose = d.qpos[hand_qpos].copy()
for g, (ct, ca) in zip(stick_geoms, saved):
    m.geom_contype[g], m.geom_conaffinity[g] = ct, ca

# 2-3. Restart in that pose with the sticks closed; the spring opens the upper stick into the fingers
mujoco.mj_resetData(m, d)
d.qpos[hand_qpos] = finger_pose
d.ctrl[:] = target
while d.time < 1.5:
    mujoco.mj_step(m, d)

qpos = " ".join(f"{x:.6g}" for x in d.qpos)
ctrl = " ".join(f"{x:.6g}" for x in d.ctrl)
with open(KEYFILE, "w") as f:
    f.write(f'<mujoco>\n  <keyframe>\n    <key name="training_grip"\n      qpos="{qpos}"\n      ctrl="{ctrl}"/>\n  </keyframe>\n</mujoco>\n')

gap = 1000 * np.linalg.norm(d.site("chopstick_a_tip").xpos - d.site("chopstick_b_tip").xpos)
touching = sorted({m.body(m.geom_bodyid[c.geom1 if m.geom_bodyid[c.geom2] == m.body("chopstick_a").id else c.geom2]).name
                   for c in d.contact[: d.ncon]
                   if m.body("chopstick_a").id in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2])})
print(f"Saved {KEYFILE}")
print(f"Hinge angle {d.joint('hinge').qpos[0]:.3f} rad, tip gap {gap:.1f} mm")
print("Upper stick is touching:", ", ".join(touching) or "nothing")
