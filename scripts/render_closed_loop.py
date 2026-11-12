"""Render actual recorded states, never the MPC's predicted configurations."""
import os
os.environ.setdefault("MUJOCO_GL","egl")
from pathlib import Path
import csv
import json
import numpy as np
import mujoco as mj
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

root=Path(__file__).resolve().parents[1]
out=root/"results/closed_loop"
names=json.loads((root/"models/manifest.json").read_text())["joint_names"][2:]
models=[mj.MjModel.from_xml_path(str(root/"models/b2_z1.xml")) for _ in range(2)]
data=[mj.MjData(m) for m in models]
traces=[]
for filename in ("bench_central_1.csv","bench_admm10_1.csv"):
    with (out/filename).open() as f:
        traces.append(list(csv.DictReader(f)))
renderers=[]
for m in models:
    m.vis.global_.offwidth=768;m.vis.global_.offheight=576
    m.vis.headlight.active=1
    m.vis.headlight.ambient[:]=[0.5,0.5,0.5]
    m.vis.headlight.diffuse[:]=[0.8,0.8,0.8]
    m.vis.headlight.specular[:]=[0.15,0.15,0.15]
    for g in range(m.ngeom):
        name=mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,m.geom_bodyid[g]) or ""
        m.geom_rgba[g]=([0.95,0.55,0.15,1] if "z1_" in name else [0.25,0.48,0.7,1])
        if m.geom_bodyid[g]==0:
            m.geom_rgba[g]=[0.18,0.21,0.25,1]
    renderers.append(mj.Renderer(m,height=576,width=768))
camera=mj.MjvCamera();camera.lookat[:]=[0.02,0,0.65];camera.distance=2.55;camera.azimuth=125;camera.elevation=-17
font=ImageFont.truetype(font_manager.findfont("DejaVu Sans"),21)
small=ImageFont.truetype(font_manager.findfont("DejaVu Sans"),17)
def marker(scene,position,color,radius=.009):
    geom=scene.geoms[scene.ngeom]
    mj.mjv_initGeom(geom,mj.mjtGeom.mjGEOM_SPHERE,np.array([radius,0,0]),np.asarray(position),np.eye(3).ravel(),np.asarray(color,dtype=np.float32))
    scene.ngeom+=1
writer=imageio.get_writer(out/"closed_loop.mp4",fps=25,codec="libx264",quality=8,macro_block_size=16)
for frame in range(min(map(len,traces))):
    panels=[]
    for i in range(2):
        m,d,renderer=models[i],data[i],renderers[i];row=traces[i][frame]
        q=np.array([float(row[f"q{j}"]) for j in range(25)])
        d.qpos[:3]=q[:3];d.qpos[3:7]=q[[6,3,4,5]]
        for j,name in enumerate(names):
            jid=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,name);d.qpos[m.jnt_qposadr[jid]]=q[7+j]
        mj.mj_forward(m,d);renderer.update_scene(d,camera)
        target=np.array([float(row[f"target_{a}"]) for a in "xyz"])
        tip=np.array([float(row[f"tip_{a}"]) for a in "xyz"])
        marker(renderer.scene,target,[1,0.2,0.15,.85],.012)
        marker(renderer.scene,tip,[.15,1,.45,1],.007)
        for old in traces[i][:frame:5]:
            marker(renderer.scene,[float(old[f"target_{a}"]) for a in "xyz"],[1,.4,.25,.6],.0025)
        img=Image.fromarray(renderer.render());draw=ImageDraw.Draw(img)
        draw.rectangle((0,0,768,85),fill=(17,24,34))
        title="Central OSQP (500 iterations max)" if i==0 else "Subsystem ADMM (10 x 100 max)"
        draw.text((18,10),title,font=font,fill="white")
        draw.text((18,43),f"t = {float(row['time']):.2f} s   |   tip error = {1000*float(row['error_m']):.2f} mm",font=small,fill=(200,220,240))
        draw.text((18,535),"Red: target   Green: actual tip   |   Collision geometry",font=small,fill="white")
        panels.append(np.asarray(img))
    combined=np.concatenate(panels,axis=1);writer.append_data(combined)
    if frame==75: Image.fromarray(combined).save(out/"simulation_preview.png")
writer.close()
for renderer in renderers: renderer.close()
print(out/"closed_loop.mp4")
