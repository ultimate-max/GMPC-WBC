import copy,xml.etree.ElementTree as ET
import importlib.metadata
import quadprog

def kernel_inputs(root):
    # The native kernel binds the exported C routine of this pinned package;
    # reject an unverified ABI instead of silently using another installation.
    if importlib.metadata.version('quadprog')!='0.1.13':raise RuntimeError('Native WBC requires quadprog 0.1.13')
    tree=ET.parse(root/'models/b2_z1.urdf').getroot()
    joints={j.get('name'):j for j in tree.findall('joint')}
    mount=joints['z1_mount'];assert mount.get('type')=='fixed'
    arm={mount.find('child').get('link')}
    while True:
        before=len(arm)
        for j in joints.values():
            if j.find('parent').get('link') in arm:arm.add(j.find('child').get('link'))
        if len(arm)==before:break
    def xml(keep_arm):
        tr=copy.deepcopy(tree)
        for link in tr.findall('link'):
            inertial=link.find('inertial')
            if (link.get('name') in arm)!=keep_arm and inertial is not None:link.remove(inertial)
        return ET.tostring(tr,encoding='unicode')
    return xml(False),xml(True),quadprog.__file__

def publish(runtime,plan,sequence):
    import numpy as np
    runtime.publish(sequence,plan.t,plan.dt,np.array([s.q0 for s in plan.segments]),
        np.vstack([s.c for s in plan.segments]),plan.torques,plan.forces)
