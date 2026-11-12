"""Export a collision-only MuJoCo scene and compare independent rigid-body models.

This checks physics model consistency; it is not a closed-loop control test.
"""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pinocchio as pin
import mujoco as mj

root = Path(__file__).resolve().parents[1]
urdf = root / "models/b2_z1.urdf"
tree = ET.parse(urdf)
omitted = []
for link in tree.getroot().findall("link"):
    for visual in link.findall("visual"):
        link.remove(visual)
    inertia = link.find("inertial")
    if inertia is not None and float(inertia.find("mass").get("value")) < 1e-10:
        # Upstream dummy sensor inertias (~1e-27) are below MuJoCo's eigenvalue
        # threshold. Drop only these effectively massless, fixed attachments.
        omitted.append(link.get("name"))
        link.remove(inertia)
ext = ET.SubElement(tree.getroot(), "mujoco")
ET.SubElement(ext, "compiler", fusestatic="false", discardvisual="true", inertiafromgeom="false")
intermediate = root / "models/mujoco_import.urdf"
tree.write(intermediate, encoding="utf-8", xml_declaration=True)
fixed = mj.MjModel.from_xml_path(str(intermediate))
scene = root / "models/b2_z1.xml"
mj.mj_saveLastXML(str(scene), fixed)
tree = ET.parse(scene)
r = tree.getroot()
# mj_saveLastXML rounds some inertial values. Restore full URDF precision rather
# than weakening the model-equivalence tolerance to hide serialization error.
for link in ET.parse(intermediate).getroot().findall("link"):
    body = r.find(f".//body[@name='{link.get('name')}']")
    inertial = link.find("inertial")
    if body is None or inertial is None:
        continue
    previous = body.find("inertial")
    if previous is not None:
        body.remove(previous)
    origin = inertial.find("origin")
    pos = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
    rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
    ri = pin.rpy.rpyToMatrix(rpy)
    a = inertial.find("inertia").attrib
    tensor = np.array([[float(a["ixx"]),float(a["ixy"]),float(a["ixz"])],
                       [float(a["ixy"]),float(a["iyy"]),float(a["iyz"])],
                       [float(a["ixz"]),float(a["iyz"]),float(a["izz"])]])
    tensor = ri @ tensor @ ri.T
    full = [tensor[0,0],tensor[1,1],tensor[2,2],tensor[0,1],tensor[0,2],tensor[1,2]]
    ET.SubElement(body,"inertial",pos=pos,mass=inertial.find("mass").get("value"),
                  fullinertia=" ".join(format(x,".17g") for x in full))
base = r.find("worldbody/body[@name='base_link']")
assert base is not None
ET.SubElement(base, "freejoint", name="floating_base")
ET.SubElement(r, "option", timestep="0.001", gravity="0 0 -9.81")
ET.SubElement(r.find("worldbody"), "geom", name="floor", type="plane", size="5 5 0.1", rgba="0.25 0.3 0.35 1", friction="0.6 0.005 0.0001", pos="0 0 -0.032")
actuators = ET.SubElement(r, "actuator")
for j in ET.parse(urdf).getroot().findall("joint"):
    if j.get("type") == "revolute":
        lim = j.find("limit").get("effort")
        ET.SubElement(actuators, "motor", name=j.get("name")+"_motor", joint=j.get("name"),
                      ctrllimited="true", ctrlrange=f"-{lim} {lim}")
ET.indent(tree)
tree.write(scene, encoding="utf-8", xml_declaration=True)
m = mj.MjModel.from_xml_path(str(scene)); d = mj.MjData(m)
pm = pin.buildModelFromUrdf(str(urdf), pin.JointModelFreeFlyer()); pd = pm.createData()
assert (m.nq, m.nv, m.nu) == (25, 24, 18)
q0 = np.loadtxt(root / "models/q_home.txt")
errors = []
for trial in range(5):
    # Nonidentity base rotations expose world/body velocity convention errors.
    delta = np.random.default_rng(trial).normal(0, 0.04, pm.nv) if trial else np.zeros(pm.nv)
    q = pin.integrate(pm, q0, delta)
    R = pin.Quaternion(q[3:7]).matrix()
    d.qpos[:3] = q[:3]; d.qpos[3:7] = q[[6,3,4,5]]
    T = np.zeros((pm.nv, m.nv)); T[:3,:3] = R.T; T[3:6,3:6] = np.eye(3)
    for joint in list(pm.names)[2:]:
        pj = pm.joints[pm.getJointId(joint)]; mjid = mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,joint)
        assert mjid >= 0
        d.qpos[m.jnt_qposadr[mjid]] = q[pj.idx_q]
        T[pj.idx_v, m.jnt_dofadr[mjid]] = 1
    d.qvel[:] = 0
    if trial == 0:
        home_qpos = d.qpos.copy()
    mj.mj_forward(m,d)
    M = np.zeros((m.nv,m.nv))
    # MuJoCo 3.4+ uses (model, data, destination).
    mj.mj_fullM(m,d,M)
    Mp = pin.crba(pm,pd,q); Mp = np.triu(Mp)+np.triu(Mp,1).T
    h = pin.nonLinearEffects(pm,pd,q,np.zeros(pm.nv))
    pin.framesForwardKinematics(pm,pd,q)
    tip_id = mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,"tool_tip")
    errors.append({"trial":trial,"mass_matrix_max_abs":float(np.max(np.abs(M-T.T@Mp@T))),
                   "gravity_max_abs":float(np.max(np.abs(d.qfrc_bias-T.T@h))),
                   "tip_position_error":float(np.linalg.norm(d.xpos[tip_id]-pd.oMf[pm.getFrameId("tool_tip")].translation))})
    assert errors[-1]["mass_matrix_max_abs"] < 1e-7, errors[-1]
    assert errors[-1]["gravity_max_abs"] < 1e-7, errors[-1]
    assert errors[-1]["tip_position_error"] < 1e-8, errors[-1]
report = {"mujoco_version":mj.__version__, "pinocchio_version":pin.__version__,
          "omitted_negligible_fixed_inertias":omitted,
          "scope":"model equivalence at five poses, zero velocity; no closed-loop validation", "checks":errors}
(root / "results").mkdir(exist_ok=True)
(root / "results/mujoco_model_check.json").write_text(json.dumps(report,indent=2))
keyframes = ET.SubElement(r, "keyframe")
ET.SubElement(keyframes, "key", name="home", qpos=" ".join(format(x,".17g") for x in home_qpos))
ET.indent(tree)
tree.write(scene, encoding="utf-8", xml_declaration=True)
scene_model = mj.MjModel.from_xml_path(str(scene))
scene_data = mj.MjData(scene_model)
mj.mj_resetDataKeyframe(scene_model, scene_data, 0)
assert np.max(np.abs(scene_data.qpos-home_qpos)) < 1e-14
print(json.dumps(report,indent=2))
