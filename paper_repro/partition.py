"""Redistribute only the shared torso inertia; total full-body inertia is unchanged."""
from pathlib import Path
from copy import deepcopy
import xml.etree.ElementTree as ET

def prepare(root):
    out=root/'paper_repro/models';out.mkdir(exist_ok=True)
    complete=ET.parse(root/'models/b2_z1.urdf').getroot()
    trunk=complete.find("link[@name='base_link']/inertial")
    for name in ['front','rear','arm']:
        tree=ET.parse(root/'models'/f'{name}.urdf');base=tree.getroot().find("link[@name='base_link']")
        old=base.find('inertial')
        if old is not None: base.remove(old)
        inertia=deepcopy(trunk)
        mass=inertia.find('mass');mass.set('value',str(float(mass.get('value'))/3))
        tensor=inertia.find('inertia')
        for key,value in list(tensor.attrib.items()): tensor.set(key,str(float(value)/3))
        base.append(inertia)
        tree.write(out/f'{name}.urdf',encoding='utf-8',xml_declaration=True)

if __name__=='__main__': prepare(Path(__file__).resolve().parents[1])
