import argparse,csv,json,time,pathlib,hashlib
import numpy as np
import native
from model import Reference,log
from controller import Controller
from relu_solver import torch

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['baseline','hgmpc'],required=True)
    ap.add_argument('--duration',type=float,default=6);ap.add_argument('--iterations',type=int,default=100)
    ap.add_argument('--horizon',type=int,default=10);ap.add_argument('--dt',type=float,default=.03)
    ap.add_argument('--scale',type=float,default=1.);ap.add_argument('--name');args=ap.parse_args()
    if args.duration<=0 or args.iterations<1 or args.horizon<1 or args.dt<=0:raise ValueError('positive time, horizon and iteration budget required')
    root=pathlib.Path(__file__).resolve().parents[1]
    out=root/'results'/'relu_hgmpc';out.mkdir(parents=True,exist_ok=True)
    name=args.name or f'{args.mode}_i{args.iterations}'
    if pathlib.Path(name).name!=name:raise ValueError('name must be a filename stem')
    plant=native.Plant(str(root));s=plant.truth();home=s['q'].copy()
    ref=Reference(plant,home,s['pose'],args.scale)
    # External, preplanned trajectory: IK is offline, identical Cartesian target.
    t0=time.perf_counter()
    if args.mode=='baseline':
        times=sorted(set(round(i*.02+k*args.dt,6) for i in range(round(args.duration/.02)) for k in range(1,args.horizon+1)))
        for t in times:ref.joint(t)
    reference_ms=1000*(time.perf_counter()-t0)
    t0=time.perf_counter();ctrl=Controller(plant,ref,args.mode,args.horizon,args.dt,args.iterations)
    initialization_ms=1000*(time.perf_counter()-t0)
    # CUDA context and allocations warmed outside the measured closed loop.
    for i in range(3):ctrl.control(s,0)
    rows=[];failure=None
    for i in range(round(args.duration/.02)):
        t=i*.02;ts=time.perf_counter();s=plant.truth();snapshot_ms=1000*(time.perf_counter()-ts)
        torque,acc=ctrl.control(s,t)
        err=log(np.linalg.inv(ref.pose_at(t))@s['pose'])
        tilt=np.linalg.norm(plant.difference(home,s['q'])[3:6])
        row=dict(time=t,position_error_mm=float(np.linalg.norm(s['pose'][:3,3]-ref.pose_at(t)[:3,3])*1000),
                 rotation_error_deg=float(np.linalg.norm(err[3:])*180/np.pi),base_z=float(s['q'][2]),tilt_rad=float(tilt),
                 foot_drift_mm=float(np.max(np.linalg.norm(s['ce'].reshape(4,3),axis=1))*1000),contacts=s['contacts'],
                 snapshot_ms=snapshot_ms,feedback_ms=snapshot_ms+ctrl.info['controller_ms'],**ctrl.info)
        row['clips']=0
        if not np.all(np.isfinite(torque)) or s['q'][2]<.35 or tilt>.5 or row['primal']>1e-3 or row['hierarchy']>1e-8:
            failure=dict(time=t,reason='nonfinite/physical/constraint validity check',row=row);rows.append(row);break
        row['clips']=plant.advance(torque,acc,20)
        rows.append(row)
        if i%50==0:print(name,{k:round(row[k],5) for k in ['time','position_error_mm','rotation_error_deg','controller_ms','primal','hierarchy']},flush=True)
    with (out/f'{name}.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in pathlib.Path(__file__).resolve().parent.iterdir() if f.suffix in ('.py','.cpp','.sh')}
    summary=dict(args=vars(args),gpu=torch.cuda.get_device_name(0),torch=torch.__version__,source_sha256=hashes,
                 reference_generation_ms=reference_ms,initialization_ms=initialization_ms,
                 duration_completed=rows[-1]['time']+.02,failed=failure,torque_clips=sum(r.get('clips',0) for r in rows),
                 position_rmse_mm=float(np.sqrt(np.mean([r['position_error_mm']**2 for r in rows]))),
                 rotation_rmse_deg=float(np.sqrt(np.mean([r['rotation_error_deg']**2 for r in rows]))))
    for key in ['feedback_ms','snapshot_ms','controller_ms','setup_ms','qp_ms','primal','primary_primal','primary_dual','secondary_dual','hierarchy','foot_drift_mm']:
        vals=[r[key] for r in rows];summary[key]=dict(mean=float(np.mean(vals)),p95=float(np.quantile(vals,.95)),max=float(np.max(vals)))
    (out/f'{name}.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
    if failure:raise RuntimeError('Closed-loop validation failed; see saved diagnostics')
if __name__=='__main__':main()
