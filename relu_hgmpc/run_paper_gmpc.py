"""Matched frozen / geometric-kinematics / complete paper-task comparison."""
import argparse,pathlib,json,time,csv,hashlib
import numpy as np
import pinocchio as pin
import native
from model import Reference
from split_gmpc import SplitGMPC
from paper_gmpc import PaperGMPC,ReferenceCache

class CanonicalReference(Reference):
    def pose_at(self,t):return super().pose_at(round(t,9))
    def twist(self,t):return super().twist(round(t,9))

ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['frozen','geometry','paper'],required=True)
ap.add_argument('--cached',action='store_true');ap.add_argument('--duration',type=float,default=12.)
ap.add_argument('--scale',type=float,default=1.);ap.add_argument('--name',required=True)
ap.add_argument('--backend',choices=['osqp','reluqp'],default='osqp')
args=ap.parse_args()
if pathlib.Path(args.name).name!=args.name or args.duration<=0:raise ValueError('Invalid run')
root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/paper_gmpc';out.mkdir(parents=True,exist_ok=True)
p=native.Plant(str(root));s=p.truth();raw=CanonicalReference(p,s['q'],s['pose'],args.scale)
ref=ReferenceCache(raw,args.duration,.02,10,.03,geometric=args.model!='frozen') if args.cached else raw
common=dict(plant=p,ref=ref,root=root,mode='central',N=10,dt=.03,backend=args.backend,qp_tol=2e-7)
c=SplitGMPC(**common) if args.model=='frozen' else PaperGMPC(**common,geometry_only=args.model=='geometry',reuse_static=args.cached)
startup_ms=getattr(ref,'precompute_ms',0.)+getattr(c,'static_precompute_ms',0.)
p.instrument=True;rows=[];details=[];failure=None;simulated=0
for k in range(round(args.duration/.02)):
    t=k*.02;feedback_start=time.perf_counter();s=p.truth();state_read_ms=1000*(time.perf_counter()-feedback_start)
    try:tau,acc=c.control(s,t)
    except Exception as e:failure=dict(time=t,reason=str(e));break
    feedback_ms=1000*(time.perf_counter()-feedback_start)
    desired=ref.pose_at(t);info=c.info.copy()
    info.setdefault('affine_dynamics_max',info['full_dynamics_max']);info.setdefault('derivatives_ms',0.)
    row=dict(time=t,position_error_mm=1000*float(np.linalg.norm(s['pose'][:3,3]-desired[:3,3])),
        rotation_error_deg=float(np.linalg.norm(pin.log3(desired[:3,:3].T@s['pose'][:3,:3]))*180/np.pi),
        contacts=s['contacts'],foot_center_mm=1000*float(np.max(np.linalg.norm(s['ce'].reshape(4,3),axis=1))),
        base_z=float(s['q'][2]),joint_position_violation=float(max(0,np.max(s['q_min']-s['q'][7:]),np.max(s['q'][7:]-s['q_max']))),
        joint_velocity_violation=float(max(0,np.max(np.abs(s['v'][6:])-s['v_max']))),
        state_read_ms=state_read_ms,feedback_ms=feedback_ms,**info,clips=0)
    details.append(dict(q=s['q'].copy(),v=s['v'].copy(),**c.debug))
    if not np.all(np.isfinite(tau)) or s['q'][2]<.35 or row['joint_position_violation']>1e-5 or row['joint_velocity_violation']>1e-4 or row['local_primal']>1e-2 or row['interface_scaled']>.01 or row['first_acceleration_defect']>.1:
        failure=dict(time=t,reason='Physical/local gate',row=row);rows.append(row);break
    row['clips']=p.advance(tau,acc,20);simulated+=1;rows.append(row)
    if k%100==0:print(args.name,{key:round(row[key],6) for key in ['time','position_error_mm','controller_ms']},flush=True)
c.close();trace=p.take_trace()
assert len(trace)==20*simulated
summary=dict(args=vars(args),failed=failure,cycles=len(rows),precompute_ms=startup_ms,
    reference_cache_misses=getattr(ref,'misses',0),reference_cache_bytes=getattr(ref,'bytes',0),
    source_sha256={f:hashlib.sha256((pathlib.Path(__file__).parent/f).read_bytes()).hexdigest() for f in ['split_gmpc.py','paper_gmpc.py','run_paper_gmpc.py','model.py','native.cpp']},
    timing_scope='feedback_ms includes MuJoCo truth read/native model evaluation, model/reference evaluation, matrix assembly, QP setup/solve, recovery and diagnostics; precompute separately and amortized. Simulation advance excluded; no computation delay injected.')
if rows:
    with (out/f'{args.name}.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary.update(position_rmse_mm=float(np.sqrt(np.mean([r['position_error_mm']**2 for r in rows]))),
        rotation_rmse_deg=float(np.sqrt(np.mean([r['rotation_error_deg']**2 for r in rows]))),torque_clips=sum(r['clips'] for r in rows),
        amortized_controller_ms=(sum(r['controller_ms'] for r in rows)+startup_ms)/len(rows),
        amortized_feedback_ms=(sum(r['feedback_ms'] for r in rows)+startup_ms)/len(rows))
    for key in ['feedback_ms','state_read_ms','controller_ms','model_ms','setup_ms','solve_ms','derivatives_ms','local_primal','interface_scaled','affine_dynamics_max','first_acceleration_defect','qp_primal_relative','qp_dual_relative','qp_normal_relative','qp_iterations','qp_converged']:
        a=np.array([r[key] for r in rows]);summary[key]=dict(mean=float(a.mean()),max=float(a.max()),p95=float(np.quantile(a,.95)),p99=float(np.quantile(a,.99)))
if len(trace):
    mask=trace[:,1:5]>0;slip=trace[:,21:33].reshape(-1,4,3).copy();slip[~mask]=0
    summary.update(min_contacts=int(mask.sum(axis=1).min()),missing_contact_ms=int(np.sum(mask.sum(axis=1)<4)),
        max_slip_path_mm=float(np.linalg.norm(slip[:,:,:2],axis=2).sum(axis=0).max()))
    np.savez_compressed(out/f'{args.name}_trace.npz',trace=trace,**{key:np.array([d[key] for d in details]) for key in details[0]})
(out/f'{args.name}.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2),flush=True)
if failure:raise RuntimeError(failure)
