import argparse,csv,json,time,pathlib,hashlib
import numpy as np
import native
from model import Reference,log
from prioritized import PrioritizedController

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task-dim',type=int,choices=[3,6],default=3)
    ap.add_argument('--duration',type=float,default=12);ap.add_argument('--scale',type=float,default=1.)
    ap.add_argument('--sigma-min',type=float,default=.1);ap.add_argument('--name')
    ap.add_argument('--activation',choices=['threshold','hysteresis','always','arm','smooth_qp','smooth_admm'],default='threshold')
    ap.add_argument('--allocation-iterations',type=int,default=80)
    ap.add_argument('--task-error',choices=['cartesian','se3'],default='cartesian')
    args=ap.parse_args()
    root=pathlib.Path(__file__).resolve().parents[1];out=root/'results'/'prioritized';out.mkdir(parents=True,exist_ok=True)
    name=args.name or f'xie_{args.task_dim}d_12s'
    if pathlib.Path(name).name!=name or args.duration<=0:raise ValueError('invalid name/duration')
    plant=native.Plant(str(root));s=plant.truth();home=s['q'].copy();ref=Reference(plant,home,s['pose'],args.scale)
    ctrl=PrioritizedController(plant,ref,args.task_dim,sigma_min=args.sigma_min,activation_mode=args.activation,allocation_iterations=args.allocation_iterations,task_error=args.task_error)
    rows=[];failure=None
    for i in range(round(args.duration/.02)):
        t=i*.02;ts=time.perf_counter();s=plant.truth();snapshot_ms=1000*(time.perf_counter()-ts)
        try:torque,acc=ctrl.control(s,t)
        except Exception as e:
            failure=dict(time=t,reason=str(e));break
        err=log(np.linalg.inv(ref.pose_at(t))@s['pose']);tilt=np.linalg.norm(plant.difference(home,s['q'])[3:6])
        row=dict(time=t,position_error_mm=float(np.linalg.norm(s['pose'][:3,3]-ref.pose_at(t)[:3,3])*1000),
            rotation_error_deg=float(np.linalg.norm(err[3:])*180/np.pi),base_z=float(s['q'][2]),tilt_rad=float(tilt),
            foot_drift_mm=float(np.max(np.linalg.norm(s['ce'].reshape(4,3),axis=1))*1000),contacts=s['contacts'],
            snapshot_ms=snapshot_ms,feedback_ms=snapshot_ms+ctrl.info['controller_ms'],**ctrl.info,clips=0)
        row['measured_joint_position_violation']=float(max(0,np.max(s['q_min']-s['q'][7:]),np.max(s['q'][7:]-s['q_max'])))
        row['measured_joint_velocity_violation']=float(max(0,np.max(np.abs(s['v'][6:])-s['v_max'])))
        if not np.all(np.isfinite(torque)) or s['q'][2]<.35 or tilt>.5 or row['primal']>1e-3 or row['hierarchy']>1e-7 or row['measured_joint_position_violation']>1e-5 or row['measured_joint_velocity_violation']>1e-4:
            failure=dict(time=t,reason='physical/dynamics/constraint validity check',row=row);rows.append(row);break
        row['clips']=plant.advance_reference(torque,ctrl.q_reference,ctrl.v_reference,20);rows.append(row)
        if i%50==0:print(name,{k:round(row[k],5) for k in ['time','position_error_mm','rotation_error_deg','controller_ms','primal','base_active','sigma_min']},flush=True)
    if rows:
        with (out/f'{name}.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(args=vars(args),backend='CPU NumPy/SciPy/quadprog; C++ Pinocchio/MuJoCo',
        source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (pathlib.Path(__file__).parent/'prioritized.py',pathlib.Path(__file__))},
        duration_completed=rows[-1]['time']+.02 if rows else 0.,failed=failure,torque_clips=sum(r['clips'] for r in rows))
    if rows:
        summary.update(position_rmse_mm=float(np.sqrt(np.mean([r['position_error_mm']**2 for r in rows]))),
            rotation_rmse_deg=float(np.sqrt(np.mean([r['rotation_error_deg']**2 for r in rows]))),
            base_active_fraction=float(np.mean([r['base_active'] for r in rows])),
            min_phi=float(min(r['phi'] for r in rows)),min_psi=float(min(r['psi'] for r in rows)),
            min_sigma=float(min(r['sigma_min'] for r in rows)),total_saturations=sum(r['saturations'] for r in rows),
            scaling_lp_calls=sum(r['scaling_lp'] for r in rows))
        for key in ['feedback_ms','snapshot_ms','controller_ms','distribution_ms','projection_ms','qp_ms','primal',
                    'primary_primal','primary_dual','hierarchy','foot_drift_mm','wbc_contact_acc_residual',
                    'wbc_float_dynamics_residual','distribution_task_residual','tertiary_task_change','velocity_violation','qp_iterations',
                    'measured_joint_position_violation','measured_joint_velocity_violation']:
            val=[r[key] for r in rows];summary[key]=dict(mean=float(np.mean(val)),p95=float(np.quantile(val,.95)),max=float(np.max(val)))
    (out/f'{name}.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
    if failure:raise RuntimeError('Prioritized WBC validation failed: '+failure['reason'])
if __name__=='__main__':main()
