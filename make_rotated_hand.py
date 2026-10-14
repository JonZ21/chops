"""Make models/right_hand_rotated.xml: the Menagerie Shadow Hand turned 90 deg about the forearm,
so the thumb is on top and the pinky is at the bottom (the usual way you hold chopsticks).

The ONLY change is the orientation of the hand's root body, rh_forearm:
    quat="0 1 0 1"  (original: palm up)  ->  quat="0.5 0.5 0.5 0.5"  (thumb up, pinky down)
i.e. the original rotation followed by -90 deg about the forearm's long axis (world x).
Run once:  python make_rotated_hand.py
"""
SRC = "mujoco_menagerie/shadow_hand/right_hand.xml"
DST = "models/right_hand_rotated.xml"
OLD = '<body name="rh_forearm" childclass="right_hand" quat="0 1 0 1">'
NEW = '<body name="rh_forearm" childclass="right_hand" quat="0.5 0.5 0.5 0.5">'

text = open(SRC).read()
assert OLD in text, "Couldn't find the forearm line - has Menagerie's right_hand.xml changed?"
open(DST, "w").write(text.replace(OLD, NEW, 1))
print(f"Wrote {DST}")
