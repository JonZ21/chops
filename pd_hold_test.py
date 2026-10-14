"""PD-hold test: start from the 'tripod_grip' keyframe, keep the finger targets frozen, and measure how
far the chopstick tips move in 5 s. Pass = both tips move less than ~5 mm.

Run:  python pd_hold_test.py            (prints results, saves pd_hold.mp4)
      python pd_hold_test.py --no-video (faster)
"""
import sys
import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path("models/chopsticks.xml")
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, m.key("tripod_grip").id)  # sets pose AND finger targets
mujoco.mj_forward(m, d)

sticks = ["chopstick_a", "chopstick_b"]
tip0 = {s: d.site(f"{s}_tip").xpos.copy() for s in sticks}
axis0 = {s: d.geom(s).xmat.reshape(3, 3)[:, 2].copy() for s in sticks}

video = "--no-video" not in sys.argv
if video:
    renderer = mujoco.Renderer(m, 480, 640)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.37, 0, 0.05]; cam.distance = 0.45; cam.azimuth = 215; cam.elevation = -30
    frames, fps = [], 30

while d.time < 5.0:
    mujoco.mj_step(m, d)
    if video and len(frames) < d.time * fps:
        renderer.update_scene(d, cam)
        frames.append(renderer.render())

for s in sticks:
    moved = 1000 * np.linalg.norm(d.site(f"{s}_tip").xpos - tip0[s])
    ax = d.geom(s).xmat.reshape(3, 3)[:, 2]
    tilt = np.degrees(np.arccos(np.clip(abs(ax @ axis0[s]), 0, 1)))
    print(f"{s}: tip moved {moved:5.1f} mm, tilted {tilt:4.1f} deg  ->  {'PASS' if moved < 5 else 'FAIL'}")

if video:
    import imageio
    imageio.mimsave("pd_hold.mp4", frames, fps=fps)
    print("Saved pd_hold.mp4")
