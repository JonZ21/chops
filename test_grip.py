"""Check the training-chopstick grip:
  1. Hold: from the keyframe, freeze the finger targets for 3 s. The tips should barely move.
  2. Open/close: extend, then flex, the index + middle fingers together and report the tip gap.
     Extending should OPEN the chopsticks (bigger gap), flexing should CLOSE them.
Saves test_grip.mp4 from the "demo" camera.

Run:  python test_grip.py            (or --no-video to skip the video)
"""
import sys
import numpy as np
import mujoco

m = mujoco.MjModel.from_xml_path("models/training_chopsticks.xml")
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, m.key("training_grip").id)
mujoco.mj_forward(m, d)

video = "--no-video" not in sys.argv
frames, fps = [], 30
if video:
    renderer = mujoco.Renderer(m, 480, 640)

def step_for(seconds):
    t_end = d.time + seconds
    while d.time < t_end:
        mujoco.mj_step(m, d)
        if video and len(frames) < d.time * fps:
            renderer.update_scene(d, camera="demo")
            frames.append(renderer.render())

def tip(name):
    return d.site(f"{name}_tip").xpos.copy()

def gap_mm():
    return 1000 * np.linalg.norm(tip("chopstick_a") - tip("chopstick_b"))

# 1. Hold
a0, b0, g0 = tip("chopstick_a"), tip("chopstick_b"), gap_mm()
step_for(3.0)
print(f"Hold 3 s: gap {g0:.1f} -> {gap_mm():.1f} mm | upper tip moved {1000*np.linalg.norm(tip('chopstick_a')-a0):.1f} mm,"
      f" lower tip moved {1000*np.linalg.norm(tip('chopstick_b')-b0):.1f} mm")

# 2. Open / close with index + middle together
fingers = [m.actuator(f"rh_A_{n}").id for n in ("FFJ3", "FFJ0", "MFJ3", "MFJ0")]
base = d.ctrl.copy()
for label, delta in [("extend 0.4 rad", -0.4), ("back to grip", 0.0), ("flex 0.2 rad", 0.2), ("flex 0.4 rad", 0.4)]:
    d.ctrl[:] = base
    d.ctrl[fingers] += delta
    step_for(1.0)
    print(f"{label:15s} -> gap {gap_mm():5.1f} mm  (hinge {d.joint('hinge').qpos[0]:.2f} rad)")

if video:
    import imageio
    imageio.mimsave("test_grip.mp4", frames, fps=fps)
    print("Saved test_grip.mp4")
