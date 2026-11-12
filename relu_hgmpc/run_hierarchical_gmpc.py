"""Matched weighted versus lexicographic GMPC closed-loop runs."""
import argparse,pathlib,json,time,csv,hashlib
import numpy as np
import pinocchio as pin
import native
from model import Reference
from paper_gmpc import ReferenceCache
from osqp_acceleration import AcceleratedGMPC
from hierarchical_gmpc import HierarchicalGMPC

class CanonicalReference(Reference):
    def pose_at(self,t):return super().pose_at(round(t,9))
    def twist(self,t):return super().twist(round(t,9))

ap=argparse.ArgumentParser()
ap.add_argument('--controller',choices=['weighted','hierarchical'],default='hierarchical')
ap.add_argument('--duration',type=float,default=12.)
ap.add_argument('--scale',type=float,default=1.)
ap.add_argument('--fresh',action='store_true')
ap.add_argument('--locks',choices=['nullspace','rate','all'],default='nullspace')
ap.add_argument('--algebra',choices=['builtin','cuda'],default='builtin')
ap.add_argument('--results-dir',default='hierarchical_gmpc')
ap.add_argument('--progress-every',type=int,default=100)
ap.add_argument('--name',required=True)
args=ap.parse_args()
if pathlib.Path(args.name).name!=args.name or args.duration<=0:raise ValueError('Invalid run')
if pathlib.Path(args.results_dir).name!=args.results_dir:raise ValueError('Invalid results directory')
root=pathlib.Path(__file__).resolve().parents[1];out=root/'results'/args.results_dir;out.mkdir(parents=True,exist_ok=True)
p=native.Plant(str(root));s=p.truth()
ref=ReferenceCache(CanonicalReference(p,s['q'],s['pose'],args.scale),args.duration,.02,10,.03)
common=dict(plant=p,ref=ref,root=root,mode='central',N=10,dt=.03,backend='osqp',qp_tol=2e-7,reuse_static=True)
init_start=time.perf_counter()
c=(HierarchicalGMPC(**common,reuse=not args.fresh,lock_mode=args.locks,algebra=args.algebra) if args.controller=='hierarchical'
   else AcceleratedGMPC(**common,optimization='eliminate' if args.fresh else 'combined',algebra=args.algebra))
initialization_ms=1000*(time.perf_counter()-init_start)
startup=ref.precompute_ms+initialization_ms
p.instrument=True;rows=[];details=[];failure=None;simulated=0
for k in range(round(args.duration/.02)):
    t=k*.02;start=time.perf_counter();s=p.truth()
    try:tau,acc=c.control(s,t)
    except Exception as e:failure=dict(time=t,reason=str(e));break
    feedback_ms=1000*(time.perf_counter()-start);desired=ref.pose_at(t)
    row=dict(time=t,position_error_mm=1000*float(np.linalg.norm(s['pose'][:3,3]-desired[:3,3])),
        rotation_error_deg=float(np.linalg.norm(pin.log3(desired[:3,:3].T@s['pose'][:3,:3]))*180/np.pi),
        contacts=s['contacts'],foot_center_mm=1000*float(np.max(np.linalg.norm(s['ce'].reshape(4,3),axis=1))),
        base_z=float(s['q'][2]),joint_position_violation=float(max(0,np.max(s['q_min']-s['q'][7:]),np.max(s['q'][7:]-s['q_max']))),
        joint_velocity_violation=float(max(0,np.max(np.abs(s['v'][6:])-s['v_max']))),
        feedback_ms=feedback_ms,**c.info,clips=0)
    details.append(dict(q=s['q'].copy(),v=s['v'].copy(),**c.debug))
    if (not np.all(np.isfinite(tau)) or s['q'][2]<.35 or row['joint_position_violation']>1e-5
        or row['joint_velocity_violation']>1e-4 or row['local_primal']>1e-2
        or row['interface_scaled']>.01 or row['first_acceleration_defect']>.1):
        failure=dict(time=t,reason='Physical gate',row=row);rows.append(row);break
    row['clips']=p.advance(tau,acc,20);simulated+=1;rows.append(row)
    if k%max(1,args.progress_every)==0:print(args.name,{key:round(row[key],8) for key in
        ['time','position_error_mm','feedback_ms','qp_iterations']+
        (['task_lock_relative'] if args.controller=='hierarchical' else [])},flush=True)
c.close();trace=p.take_trace();assert len(trace)==20*simulated
summary=dict(args=vars(args),failed=failure,cycles=len(rows),simulated_cycles=simulated,
    precompute_ms=startup,initialization_ms=initialization_ms,reference_cache_misses=ref.misses,
    source_sha256={f:hashlib.sha256((pathlib.Path(__file__).parent/f).read_bytes()).hexdigest() for f in
        ['split_gmpc.py','paper_gmpc.py','osqp_acceleration.py','hierarchical_gmpc.py',
         'run_hierarchical_gmpc.py','model.py','native.cpp']},
    timing_scope='state read + model/matrices + contact reduction + both levels + multiplier recovery + original-coordinate KKT/lock/physics checks; reference/controller initialization amortized. CUDA prepare/solve explicitly synchronized; transfer included. Simulator advance excluded. No computation-delay injection.')
if rows:
    with (out/f'{args.name}.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary.update(position_rmse_mm=float(np.sqrt(np.mean([r['position_error_mm']**2 for r in rows]))),
        rotation_rmse_deg=float(np.sqrt(np.mean([r['rotation_error_deg']**2 for r in rows]))),
        torque_clips=sum(r['clips'] for r in rows),osqp_setups=sum(r['osqp_rebuilt'] for r in rows),
        pattern_growths=sum(r['pattern_growth'] for r in rows),
        amortized_feedback_ms=(sum(r['feedback_ms'] for r in rows)+startup)/len(rows))
    for key in rows[0]:
        a=np.array([r[key] for r in rows],dtype=float)
        summary[key]=dict(mean=float(a.mean()),max=float(a.max()),min=float(a.min()),
                          p95=float(np.quantile(a,.95)),p99=float(np.quantile(a,.99)))
if len(trace):
    mask=trace[:,1:5]>0;slip=trace[:,21:33].reshape(-1,4,3).copy();slip[~mask]=0
    summary.update(min_contacts=int(mask.sum(axis=1).min()),missing_contact_ms=int(np.sum(mask.sum(axis=1)<4)),
        max_slip_path_mm=float(np.linalg.norm(slip[:,:,:2],axis=2).sum(axis=0).max()))
    np.savez_compressed(out/f'{args.name}_trace.npz',trace=trace,**{key:np.array([d[key] for d in details]) for key in details[0]})
(out/f'{args.name}.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({k:summary[k] for k in ['args','failed','cycles','position_rmse_mm','rotation_rmse_deg','amortized_feedback_ms'] if k in summary}),flush=True)
if failure:raise RuntimeError(failure)
