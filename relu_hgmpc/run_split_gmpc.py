import argparse,pathlib,json,time,csv,hashlib
import numpy as np
import pinocchio as pin
import native
from model import Reference
from split_gmpc import SplitGMPC

ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['central','distributed'],default='distributed')
ap.add_argument('--duration',type=float,default=12);ap.add_argument('--name')
ap.add_argument('--iterations',type=int,default=100);ap.add_argument('--rho',type=float,default=30.)
ap.add_argument('--nodes',type=int,default=10);ap.add_argument('--step',type=float,default=.03)
ap.add_argument('--scale',type=float,default=1.);ap.add_argument('--parallel',action='store_true')
ap.add_argument('--backend',choices=['osqp','reluqp'],default='osqp')
ap.add_argument('--qp-tol',type=float,default=2e-7);ap.add_argument('--qp-max-iter',type=int,default=3000)
args=ap.parse_args();root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/split_gmpc';out.mkdir(exist_ok=True,parents=True)
name=args.name or args.mode
if pathlib.Path(name).name!=name or args.duration<=0 or args.step<.02:raise ValueError('Invalid run configuration')
p=native.Plant(str(root));s=p.truth();ref=Reference(p,s['q'],s['pose'],args.scale)
c=SplitGMPC(p,ref,root,args.mode,args.nodes,args.step,args.iterations,args.rho,args.parallel,
            args.backend,args.qp_tol,args.qp_max_iter);p.instrument=True
if p.take_trace().shape[1]!=129:raise RuntimeError('Rebuild native module for 1 kHz actual velocities')
rows=[];details=[];failure=None;simulated_cycles=0
for k in range(round(args.duration/.02)):
    t=k*.02;s=p.truth()
    try:tau,acc=c.control(s,t)
    except Exception as e:failure=dict(time=t,reason=str(e));break
    desired=ref.pose_at(t)
    row=dict(time=t,position_error_mm=1000*float(np.linalg.norm(s['pose'][:3,3]-desired[:3,3])),
        rotation_error_deg=float(np.linalg.norm(pin.log3(desired[:3,:3].T@s['pose'][:3,:3]))*180/np.pi),
        contacts=s['contacts'],foot_center_mm=1000*float(np.max(np.linalg.norm(s['ce'].reshape(4,3),axis=1))),
        base_z=float(s['q'][2]),joint_position_violation=float(max(0,np.max(s['q_min']-s['q'][7:]),np.max(s['q'][7:]-s['q_max']))),
        joint_velocity_violation=float(max(0,np.max(np.abs(s['v'][6:])-s['v_max']))),**c.info,clips=0)
    details.append(dict(q=s['q'].copy(),v=s['v'].copy(),**c.debug))
    if not np.all(np.isfinite(tau)) or s['q'][2]<.35 or row['joint_position_violation']>1e-5 or row['joint_velocity_violation']>1e-4 or row['local_primal']>1e-2 or row['interface_scaled']>.01 or row['first_acceleration_defect']>.1:
        failure=dict(time=t,reason='Physical/local constraint gate',row=row);rows.append(row);break
    row['clips']=p.advance(tau,acc,20);simulated_cycles+=1;rows.append(row)
    if k%50==0:print(name,{key:round(row[key],6) for key in ['time','position_error_mm','interface_scaled','full_dynamics_max','controller_ms']},flush=True)
c.close();trace=p.take_trace()
summary=dict(args=vars(args),failed=failure,cycles=len(rows),
    source_sha256={f:hashlib.sha256((pathlib.Path(__file__).parent/f).read_bytes()).hexdigest() for f in ['split_gmpc.py','run_split_gmpc.py','central_relu.py','relu_solver.py','native.cpp']})
if rows:
    with (out/f'{name}.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary.update(position_rmse_mm=float(np.sqrt(np.mean([r['position_error_mm']**2 for r in rows]))),
        rotation_rmse_deg=float(np.sqrt(np.mean([r['rotation_error_deg']**2 for r in rows]))),torque_clips=sum(r['clips'] for r in rows))
    for key in ['controller_ms','model_ms','setup_ms','solve_ms','iterations','local_primal','interface_scaled','dual_residual','full_dynamics_max','first_acceleration_defect','wrench_force_max','wrench_moment_max','contact_acceleration_residual','interface_pose_increment_max','nonlinear_base_force_max','nonlinear_base_moment_max','nonlinear_joint_torque_max','admm_converged']:
        a=np.array([r[key] for r in rows]);summary[key]=dict(mean=float(a.mean()),max=float(a.max()),p95=float(np.quantile(a,.95)))
    for key in ['qp_primal_raw','qp_primal_relative','qp_dual_raw','qp_dual_relative','qp_normal_relative','qp_iterations','qp_converged','preprocessing_ms','kernel_setup_ms']:
        if key in rows[0]:
            a=np.array([r[key] for r in rows]);summary[key]=dict(mean=float(a.mean()),max=float(a.max()),p95=float(np.quantile(a,.95)))
if len(trace)!=20*simulated_cycles:raise RuntimeError('Incomplete 1 kHz instrumentation')
if len(trace):
    mask=trace[:,1:5]>0;slip=trace[:,21:33].reshape(-1,4,3).copy();slip[~mask]=0
    summary.update(min_contacts=int(mask.sum(axis=1).min()),missing_contact_ms=int(np.sum(mask.sum(axis=1)<4)),
        max_slip_path_mm=float(np.linalg.norm(slip[:,:,:2],axis=2).sum(axis=0).max()))
    np.savez_compressed(out/f'{name}_trace.npz',trace=trace,**{key:np.array([d[key] for d in details]) for key in details[0]})
(out/f'{name}.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
if failure:raise RuntimeError(failure)
