"""Combine the exact upstream rigid-body inertias, with a configurable fixed mount."""
import json
import pathlib
import subprocess
import xml.etree.ElementTree as ET
import copy
import numpy as np
import pinocchio as pin

ROOT = pathlib.Path(__file__).resolve().parents[1]
cfg = json.loads((ROOT / "config/model.json").read_text())
repo = ROOT / "third_party/unitree_ros"
b2 = ET.parse(repo / "robots/b2_description/urdf/b2_description.urdf").getroot()
z1 = ET.parse(repo / "robots/z1_description/xacro/z1.urdf").getroot()
robot = ET.Element("robot", name="b2_z1")
for source, prefix in [(b2, ""), (z1, "z1_")]:
    for element in source:
        if element.tag not in ("link", "joint"):
            continue
        if prefix and element.get("name") in ("world", "base_static_joint"):
            continue
        element.set("name", prefix + element.get("name"))
        for node in element.findall("parent") + element.findall("child"):
            node.set("link", prefix + node.get("link"))
        # Multiple material tags on upstream B2 visuals violate URDF schema.
        # Keep the first; inertias, joints and collision shapes remain unchanged.
        for visual in element.findall("visual"):
            for extra in visual.findall("material")[1:]:
                visual.remove(extra)
        for mesh in element.iter("mesh"):
            mesh.set("filename", str(repo / "robots" / mesh.get("filename").removeprefix("package://")))
        robot.append(element)
mount = ET.SubElement(robot, "joint", name="z1_mount", type="fixed")
ET.SubElement(mount, "parent", link="base_link")
ET.SubElement(mount, "child", link="z1_link00")
ET.SubElement(mount, "origin", xyz=" ".join(map(str, cfg["mount_xyz"])),
              rpy=" ".join(map(str, cfg["mount_rpy"])))
tip = ET.SubElement(robot, "link", name="tool_tip")
j = ET.SubElement(robot, "joint", name="tool_tip_joint", type="fixed")
ET.SubElement(j, "parent", link="z1_link06")
ET.SubElement(j, "child", link="tool_tip")
ET.SubElement(j, "origin", xyz="0.051 0 0", rpy="0 0 0")
out = ROOT / "models"
out.mkdir(exist_ok=True)
ET.indent(robot)
path = out / "b2_z1.urdf"
ET.ElementTree(robot).write(path, encoding="utf-8", xml_declaration=True)
m = pin.buildModelFromUrdf(str(path), pin.JointModelFreeFlyer())
d = m.createData()
q = pin.neutral(m)
for leg in ("FL", "FR", "RL", "RR"):
    for kind, value in zip(("hip", "thigh", "calf"), cfg["leg_home"]):
        q[m.joints[m.getJointId(f"{leg}_{kind}_joint")].idx_q] = value
for i, value in enumerate(cfg["arm_home"], 1):
    q[m.joints[m.getJointId(f"z1_joint{i}")].idx_q] = value
pin.framesForwardKinematics(m, d, q)
feet = [m.getFrameId(f"{leg}_foot") for leg in ("FL", "FR", "RL", "RR")]
q[2] = -np.mean([d.oMf[f].translation[2] for f in feet])
assert m.nq == 25 and m.nv == 24
assert np.all(q[7:] >= m.lowerPositionLimit[7:]) and np.all(q[7:] <= m.upperPositionLimit[7:])
np.savetxt(out / "q_home.txt", q)
manifest = {"source_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
            "nq": m.nq, "nv": m.nv, "actuated_joints": m.nv - 6,
            "mass_kg": pin.computeTotalMass(m), "mount": cfg,
            "joint_names": list(m.names), "base_height": float(q[2]),
            "tool_tip": "virtual point 0.051 m along z1_link06 local x; no gripper mass"}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))

# Physical decomposition at a common base frame: front legs, rear legs, arm.
# Split trunk inertia equally across front/rear; assign each articulated branch
# exactly once. This preserves the total inertia, rather than cloning the trunk.
parents = {j.find("child").get("link"): j.find("parent").get("link") for j in robot.findall("joint")}
def owner(name):
    if name.startswith(("FL_", "FR_")): return "front"
    if name.startswith(("RL_", "RR_")): return "rear"
    if name.startswith("z1_") or name == "tool_tip": return "arm"
    return "torso"
for part in ("front", "rear", "arm"):
    local = ET.Element("robot", name=f"b2_z1_{part}")
    keep = {e.get("name") for e in robot.findall("link")
            if owner(e.get("name")) == part or (part != "arm" and owner(e.get("name")) == "torso")}
    keep.add("base_link")
    for element in robot:
        e = copy.deepcopy(element)
        if e.tag == "link" and e.get("name") in keep:
            if owner(e.get("name")) == "torso":
                inertial = e.find("inertial")
                if inertial is not None:
                    if part == "arm": e.remove(inertial)
                    else:
                        mass = inertial.find("mass")
                        mass.set("value", str(0.5*float(mass.get("value"))))
                        tensor = inertial.find("inertia")
                        for key,value in list(tensor.attrib.items()): tensor.set(key,str(0.5*float(value)))
            local.append(e)
        elif e.tag == "joint" and e.find("parent").get("link") in keep and e.find("child").get("link") in keep:
            local.append(e)
    ET.indent(local)
    ET.ElementTree(local).write(out/f"{part}.urdf",encoding="utf-8",xml_declaration=True)
