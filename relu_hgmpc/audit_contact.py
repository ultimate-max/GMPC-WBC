import argparse,pathlib,json,time,hashlib
import numpy as np
import pinocchio as pin
import native
from model import Reference,contact
from prioritized import PrioritizedController

ap=argparse.ArgumentParser();ap.add_argument('--activation',default='threshold',choices=['threshold','always','arm','hysteresis','smooth_qp','smooth_admm'])
ap.add_argument('--allocation-iterations',type=int,default=80)
ap.add_argument('--scale',type=float,default=1.)
ap.add_argument('--task-error',choices=['cartesian','se3'],default='cartesian')
ap.add_argument('--output-dir',default='results/contact_audit')
ap.add_argument('--duration',type=float,default=12);ap.add_argument('--name');args=ap.parse_args()
root=pathlib.Path(__file__).resolve().parents[1];out=root/args.output_dir;out.mkdir(exist_ok=True,parents=True)
name=args.name or args.activation
if pathlib.Path(name).name!=name or args.duration<=0:raise ValueError('Invalid run name or duration')
p=native.Plant(str(root));s=p.truth();home=s['q'];origin=s['feet'].copy();ref=Reference(p,home,s['pose'],args.scale)
c=PrioritizedController(p,ref,6,activation_mode=args.activation,allocation_iterations=args.allocation_iterations,task_error=args.task_error);p.instrument=True
records=[];detailed=[];failure=None
for i in range(round(args.duration/.02)):
    s=p.truth();t=i*.02
    try:tau,acc=c.control(s,t)
    except Exception as e:failure=dict(time=t,error=str(e));break
    debug=c.debug;T,a0,F,f0=contact(s);fc=F@tau+f0
    pose=ref.pose_at(t)
    row=dict(time=t,center_drift_mm=1000*float(np.max(np.linalg.norm(s['ce'].reshape(4,3),axis=1))),
        base_active=c.info['base_active'],sigma=c.info['sigma_min'],contacts=s['contacts'],
        position_error_mm=1000*float(np.linalg.norm(s['pose'][:3,3]-pose[:3,3])),
        rotation_error_deg=float(np.linalg.norm(pin.log3(pose[:3,:3].T@s['pose'][:3,:3]))*180/np.pi),
        controller_ms=c.info['controller_ms'],distribution_ms=c.info['distribution_ms'],
        task_residual=c.info['distribution_task_residual'],
        slew_violation=float(max(0,np.max(np.abs(debug['desired_acc'])-np.r_[np.full(3,2.),np.full(3,3.),np.full(6,8.)]))),
        allocation_iterations=c.info.get('allocation_iterations',0),
        allocation_dual_residual=c.info.get('allocation_dual_residual',0.),
        allocation_primal_residual=c.info.get('allocation_primal_residual',0.),
        allocation_converged=c.info.get('allocation_converged',1),
        measured_joint_position_violation=float(max(0,np.max(s['q_min']-s['q'][7:]),np.max(s['q'][7:]-s['q_max']))),
        measured_joint_velocity_violation=float(max(0,np.max(np.abs(s['v'][6:])-s['v_max']))),
        primal=c.info['primal'],velocity_violation=c.info['velocity_violation'],
        acc_jump_max=float(np.max(np.abs(debug['desired_acc']))),base_acc_jump=float(np.linalg.norm(debug['desired_acc'][:6])),
        qp_delta_acc=float(np.linalg.norm(debug['delta_a'])),contact_pre=float(np.max(np.abs(debug['contact_pre']))),
        contact_post=float(np.max(np.abs(debug['contact_post']))),pred_fz_min=float(np.min(debug['force'][2::3])),
        rigid_fz_min=float(np.min(fc[2::3])),force_inconsistency=float(np.linalg.norm(fc-debug['force'])))
    detailed.append(dict(q=s['q'].copy(),v=s['v'].copy(),tau=tau.copy(),**debug))
    row['clips']=p.advance_reference(tau,c.q_reference,c.v_reference,20);records.append(row)
    if not np.isfinite(s['q']).all() or s['q'][2]<.35 or row['primal']>1e-3 or row['measured_joint_position_violation']>1e-5 or row['measured_joint_velocity_violation']>1e-4:
        failure=dict(time=t,error='physical or constraint failure');break
if not records:
    (out/f'{name}.json').write_text(json.dumps(dict(name=name,failed=failure,control_cycles=0),indent=2))
    raise RuntimeError(failure)
trace=p.take_trace();np.savez_compressed(out/f'{name}_trace.npz',trace=trace,
    **{k:np.array([d[k] for d in detailed]) for k in detailed[0]})
keys=list(records[0]);arr=np.array([[r[k] for k in keys] for r in records]);np.savetxt(out/f'{name}.csv',arr,delimiter=',',header=','.join(keys),comments='')
mask=trace[:,1:5]>0;fz=trace[:,5:9];center=trace[:,9:21].reshape(-1,4,3)
vc=trace[:,21:33].reshape(-1,4,3);vcenter=trace[:,33:45].reshape(-1,4,3)
omega=trace[:,45:57].reshape(-1,4,3)
# Exact velocity decomposition at each measured evaluation point. Contact point
# motion of the material and center motion due to rotation must not be conflated.
slip_vel=vc.copy();slip_vel[~mask]=0
rolling_vel=(vcenter-vc);rolling_vel[~mask]=0
air_vel=vcenter.copy();air_vel[mask]=0
dt=.001
slip_disp=np.cumsum(slip_vel[:,:,:2],axis=0)*dt
roll_disp=np.cumsum(rolling_vel[:,:,:2],axis=0)*dt
air_disp=np.cumsum(air_vel[:,:,:2],axis=0)*dt
center_disp=center[:,:,:2]-center[0,:,:2]
slip_path=np.sum(np.linalg.norm(slip_vel[:,:,:2],axis=2),axis=0)*dt
support=fz>1
switches=np.flatnonzero(np.diff(arr[:,keys.index('base_active')])!=0)+1
events=[{k:records[j][k] for k in ['time','base_active','sigma','acc_jump_max','contact_post','pred_fz_min','rigid_fz_min']} for j in switches]
no_contact=np.flatnonzero(np.sum(mask,axis=1)<4)
summary=dict(name=name,activation=args.activation,failed=failure,control_cycles=len(records),
    source_sha256={f:hashlib.sha256((pathlib.Path(__file__).parent/f).read_bytes()).hexdigest()
        for f in ['prioritized.py','smooth_distribution.py','audit_contact.py','model.py']},
    scale=args.scale,task_error=args.task_error,allocation_iteration_budget=args.allocation_iterations,
    position_rmse_mm=float(np.sqrt(np.mean(arr[:,keys.index('position_error_mm')]**2))),
    rotation_rmse_deg=float(np.sqrt(np.mean(arr[:,keys.index('rotation_error_deg')]**2))),
    controller_ms_mean=float(np.mean(arr[:,keys.index('controller_ms')])),
    controller_ms_p95=float(np.quantile(arr[:,keys.index('controller_ms')],.95)),
    distribution_ms_mean=float(np.mean(arr[:,keys.index('distribution_ms')])),
    max_task_residual=float(np.max(arr[:,keys.index('task_residual')])),
    max_slew_violation=float(np.max(arr[:,keys.index('slew_violation')])),
    converged_fraction=float(np.mean(arr[:,keys.index('allocation_converged')])),
    mean_allocation_iterations=float(np.mean(arr[:,keys.index('allocation_iterations')])),
    max_allocation_primal=float(np.max(arr[:,keys.index('allocation_primal_residual')])),
    max_measured_position_violation=float(np.max(arr[:,keys.index('measured_joint_position_violation')])),
    max_measured_velocity_violation=float(np.max(arr[:,keys.index('measured_joint_velocity_violation')])),
    max_primal=float(np.max(arr[:,keys.index('primal')])),
    torque_clips=int(np.sum(arr[:,keys.index('clips')])),
    max_center_drift_mm=float(np.max(np.linalg.norm(center-center[0],axis=2))*1000),
    contacts_min=int(np.min(np.sum(mask,axis=1))),missing_contact_steps=int(np.sum(np.sum(mask,axis=1)<4)),
    first_missing_contact_s=float(trace[no_contact[0],0]) if len(no_contact) else None,
    low_load_steps=int(np.sum(np.sum(support,axis=1)<4)),
    switch_count=len(switches),switch_events=events,
    peak_command_acceleration=float(np.max(arr[:,keys.index('acc_jump_max')])),
    max_contact_pre=float(np.max(arr[:,keys.index('contact_pre')])),max_contact_post=float(np.max(arr[:,keys.index('contact_post')])),
    peak_slip_speed_mps=float(np.max(np.linalg.norm(slip_vel[:,:,:2],axis=2))),
    per_foot_final_center_xy_mm=(center_disp[-1]*1000).tolist(),
    per_foot_final_slip_xy_mm=(slip_disp[-1]*1000).tolist(),per_foot_final_roll_xy_mm=(roll_disp[-1]*1000).tolist(),
    per_foot_final_air_xy_mm=(air_disp[-1]*1000).tolist(),per_foot_slip_path_mm=(slip_path*1000).tolist(),
    velocity_decomposition_error_mm=float(np.max(np.abs(center_disp-(slip_disp+roll_disp+air_disp)))*1000),
    max_pd_torque_nm=float(np.max(np.abs(trace[:,75:93]))),max_friction_torque_nm=float(np.max(np.abs(trace[:,93:111]))))
(out/f'{name}.json').write_text(json.dumps(summary,indent=2))
brief={k:v for k,v in summary.items() if k not in ['switch_events']};print(json.dumps(brief,indent=2),flush=True)
if failure:raise RuntimeError(failure)
