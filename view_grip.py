"""Open the interactive viewer already in the training grip.
Try the Control panel sliders rh_A_FFJ3 / rh_A_FFJ0 (index) and rh_A_MFJ3 / rh_A_MFJ0 (middle).
Run:  python view_grip.py
"""
import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path("models/training_chopsticks.xml")
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, m.key("training_grip").id)
mujoco.viewer.launch(m, d)
