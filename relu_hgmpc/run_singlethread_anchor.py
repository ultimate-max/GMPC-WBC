"""Single-thread WBC-clock scheduler. WBC executes BEFORE each MPC job.

Scheduled 20 ms planning delay is simulated, with C2 plan splicing. Compute
latencies are measured including construction and recovery. This test does
not certify real-time operation: simulation waits for every solve to finish.
"""
import argparse,csv,hashlib,json,pathlib,time,traceback
import numpy as np
import pinocchio as pin
import native
from geometric_wbc import pose,rolling_contact
from anchored_geometry import mount_transform,AnchoredWBC
from anchored_mpc import AnchoredMPC
from paper_anchor_baseline import PaperUpper,PaperRateController,EEReference
from online_wbc_mpc import ReferenceMux
from run_online_wbc_mpc import Target
from synchronous_schedule import SynchronousSchedule

class Goal:
    def __init__(self,p,root,case):
        self.raw=Target(p,'double' if case=='double' else 'normal',12.,root/'results/online_wbc_mpc_50hz')
        self.case=case;self.fixed=self.raw.at(0.)['ee_pose'].copy()
        self.hash=hashlib.sha256((self.raw.hash+case).encode()).hexdigest()
    def at(self,t):
        goal=dict(self.raw.at(t))
        if self.case=='hold':
            goal.update(ee_pose=self.fixed,ee_twist=np.zeros(6),ee_acc=np.zeros(6))
        return goal

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--method',choices=['paper','paper_shared','anchor','direct'],required=True)
    ap.add_argument('--case',choices=['normal','double','hold','push'],default='normal')
    ap.add_argument('--duration',type=float,default=12.)
    ap.add_argument('--name',required=True)
    ap.add_argument('--wbc-hz',type=int,choices=[100,200],default=200)
    ap.add_argument('--schedule',choices=['ratio','budget'],default='budget')
    args=ap.parse_args();root=pathlib.Path(__file__).resolve().parents[1]
    out=root/'results/anchored_mpc';out.mkdir(parents=True,exist_ok=True)
    (root/'results/prioritized').mkdir(exist_ok=True)
    if pathlib.Path(args.name).name!=args.name:raise ValueError(args.name)
    p=native.Plant(str(root));p.fast_reference=True;s=p.truth();home=s['q'].copy()
    target=Goal(p,root,args.case);mux=ReferenceMux(p,s)
    model=pin.buildModelFromUrdf(str(root/'models/b2_z1.urdf'),pin.JointModelFreeFlyer())
    wbc=AnchoredWBC(p,mount_transform(model,home))
    schedule=SynchronousSchedule(args.wbc_hz,50.,50.,100.,args.schedule)
    dt=schedule.dt;wbc.dt=dt;substeps=round(1000*dt)
    upper=None;paper=None
    if args.method in ('anchor','direct'):upper=AnchoredMPC(p,root,target,error_mode=args.method)
    elif args.method=='paper_shared':upper=PaperUpper(p,target)
    else:paper=PaperRateController(p,EEReference(target),task_dim=6,dt=.02)
    pending=None;rows=[];plans=[];states=[];failure=None;advanced=0
    p.instrument=True;p.instrument_base=True
    for k in range(round(args.duration/dt)):
        t=k*dt;start=time.perf_counter();cpu=time.thread_time();planner_ms=0.;continuity=0.
        s=p.truth();goal=target.at(t)
        try:
            if pending is not None and k>=pending[1]:
                continuity=mux.install(pending[0],t);pending=None
            wr=time.perf_counter()
            if paper is not None:
                tau,acc=paper.control(rolling_contact(s),t)
                ref=dict(q=paper.q_reference,v=paper.v_reference,a=acc)
                info=dict(controller_ms=paper.info['controller_ms'],constraint_violation=paper.info['primal'],
                    contact_residual=paper.info['wbc_contact_acc_residual'],dynamics_residual=paper.info['wbc_float_dynamics_residual'],
                    task_lock_max=paper.info['hierarchy'],qp_kkt=paper.info['primary_dual'])
                if round(t/.005)%4==0:plans.append(dict(time=t,wall_ms=paper.info['distribution_ms'],**paper.allocation_info))
            else:
                ref=mux.reference(t)
                # Execute the coherent planned EE/base/arm reference. All
                # reported task errors are against the ORIGINAL external
                # target below, never against the planner's own forward FK.
                tau,acc=wbc.control(s,ref)
                info={key:wbc.info[key] for key in ('controller_ms','constraint_violation','contact_residual','dynamics_residual','task_lock_max')}
                info['qp_kkt']=max(wbc.info[key] for key in ('qp_primal_relative','qp_dual_relative','qp_normal_relative'))
            wbc_ms=1000*(time.perf_counter()-wr)
            wbc_response_ms=1000*(time.perf_counter()-start)
            if upper is not None and pending is None and schedule.due(k):
                stamp=time.perf_counter();delay=dt*schedule.divisor;anchor=mux.state(t+delay)
                if args.method=='paper_shared':
                    plan=upper.solve(s,t,anchor,tau,ref['force'],release_delay=delay)
                else:plan=upper.solve(s,t+delay,tau,anchor[2],anchor=anchor,first_slew_time=delay)
                planner_ms=1000*(time.perf_counter()-stamp)
                pending=(plan,k+schedule.divisor);schedule.submitted(k)
                plans.append(dict(time=t,wall_ms=planner_ms,divisor=schedule.divisor,
                    release_delay_ms=1000*delay,**upper.last_info))
            schedule.observe(k,wbc_ms,planner_ms if planner_ms else None)
            row=dict(time=t,total_ms=1000*(time.perf_counter()-start),cpu_ms=1000*(time.thread_time()-cpu),
                     planner_ms=planner_ms,wbc_reference_ms=wbc_ms,reference_continuity=continuity,
                     wbc_response_ms=wbc_response_ms,mpc_divisor=schedule.divisor,
                     requested_mpc_hz=args.wbc_hz/schedule.divisor,budget_overload=int(schedule.over_budget),**info)
            row['cycle_overrun']=int(row['total_ms']>1000*dt)
            row['estimated_blocked_wbc_slots']=max(0,int(np.ceil(row['total_ms']/(1000*dt)))-1)
            for frame,T in [('base',pose(s['q'])),('ee',pin.SE3(s['pose']))]:
                g=goal[frame+'_pose']
                row[frame+'_position_error_mm']=float(np.linalg.norm(T.translation-g[:3,3])*1000)
                row[frame+'_rotation_error_deg']=float(np.linalg.norm(pin.log3(g[:3,:3].T@T.rotation))*180/np.pi)
            row.update(sigma_min=float(np.linalg.svd(s['J'][:,18:],compute_uv=False)[-1]),
                       contacts=int(s['contacts']),base_height=float(s['q'][2]),
                       joint_position_violation=float(max(0,np.max(s['q_min']-s['q'][7:]),np.max(s['q'][7:]-s['q_max']))),
                       joint_velocity_violation=float(max(0,np.max(np.abs(s['v'][6:])-s['v_max']))))
            if not np.all(np.isfinite(tau)) or s['q'][2]<.35 or row['joint_position_violation']>1e-5 or row['joint_velocity_violation']>1e-4:
                raise RuntimeError('Measured physical state gate')
            if args.case=='push':p.set_base_wrench(np.array([120.,0.,0.]) if 6.<=t<6.15 else np.zeros(3),np.zeros(3))
            states.append(dict(q=s['q'],v=s['v'],tau=tau,acc=acc,reference_q=ref['q'],reference_v=ref['v']))
            row['clips']=p.advance_reference(tau,ref['q'],ref['v'],substeps);advanced+=1;rows.append(row)
            if k%round(2/dt)==0:print(args.name,round(t,3),'EE mm',round(row['ee_position_error_mm'],3),'base mm',round(row['base_position_error_mm'],3),'ms',round(row['total_ms'],3),'MPC Hz',args.wbc_hz/schedule.divisor,flush=True)
        except Exception as exc:
            failure=dict(time=t,reason=str(exc),traceback=traceback.format_exc());print(failure,flush=True)
            np.savez_compressed(out/f'{args.name}_failure_state.npz',q=s['q'],v=s['v'])
            break
    trace=p.take_trace();base=p.take_base_trace()
    assert trace.shape==(substeps*advanced,129)
    np.savez_compressed(out/f'{args.name}_trace.npz',trace=trace,base=base,home=home,
                        **({key:np.array([st[key] for st in states]) for key in states[0]} if states else {}))
    if rows:
        with (out/f'{args.name}.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=dict(args=vars(args),completed_seconds=advanced*dt,failure=failure,target_hash=target.hash,
        clock='single-thread WBC-first controlled simulation; integer-tick MPC; not real-time',
        wbc_reference='coherent planned EE/base/arm; tracking metrics use external goal',
        rates_hz=dict(upper_initial=50,upper_minimum=50,wbc=args.wbc_hz,servo=1000),
        schedule_events=schedule.events,scheduler_mode=args.schedule,
        horizon=dict(nodes=10,step_seconds=.04) if args.method in ('anchor','direct') else None,
        plans=plans,total_controller_cpu_ms=sum(r['cpu_ms'] for r in rows),
        source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in pathlib.Path(__file__).parent.glob('*.py')})
    if rows:
        summary['singlethread_timing']=dict(cycle_overrun_fraction=float(np.mean([r['cycle_overrun'] for r in rows])),
            estimated_blocked_slots=sum(r['estimated_blocked_wbc_slots'] for r in rows),
            budget_overload_fraction=float(np.mean([r['budget_overload'] for r in rows])),
            actual_simulation_replans_hz=len(plans)/(advanced*dt),
            wbc_response_p95_ms=float(np.quantile([r['wbc_response_ms'] for r in rows],.95)),
            warning='Blocked slots are timing estimates; simulator waits during solves, so these are not hardware delay experiments.')
        summary.update(torque_clips=sum(r['clips'] for r in rows),minimum_contacts=min(r['contacts'] for r in rows),
            contact_loss_fraction=float(np.mean(trace[:,1:5]<.5)),minimum_sigma=min(r['sigma_min'] for r in rows))
        for key in ['ee_position_error_mm','ee_rotation_error_deg','base_position_error_mm','base_rotation_error_deg']:
            summary[key]=dict(rms=float(np.sqrt(np.mean([r[key]**2 for r in rows]))),max=max(r[key] for r in rows))
        for key in ['total_ms','cpu_ms','wbc_reference_ms','constraint_violation','contact_residual','dynamics_residual','task_lock_max','qp_kkt']:
            v=np.array([r[key] for r in rows]);summary[key]=dict(mean=float(v.mean()),p95=float(np.quantile(v,.95)),max=float(v.max()))
        lat=np.array([x['wall_ms'] for x in plans]);summary['upper_total_ms']=dict(mean=float(lat.mean()),p95=float(np.quantile(lat,.95)),max=float(lat.max()),over20_fraction=float(np.mean(lat>20)))
        from scipy.signal import butter,sosfiltfilt
        for name,columns in [('leg_velocity',slice(111,123)),('arm_velocity',slice(123,129)),('leg_torque',slice(57,69)),('arm_torque',slice(69,75))]:
            x=trace[:,columns];sel=trace[:,0]>=min(1.,advanced*dt/2)
            values=dict(step_rms=float(np.sqrt(np.mean(np.diff(x[sel],axis=0)**2))))
            for low in (5,10):
                filtered=sosfiltfilt(butter(4,[low,100],fs=1000,btype='bandpass',output='sos'),x,axis=0)
                values[f'hf_{low}_100_rms']=float(np.sqrt(np.mean(filtered[sel]**2)))
            summary[name]=values
        summary['foot_velocity_rms']=float(np.sqrt(np.mean(trace[:,21:33]**2)))
    (out/f'{args.name}.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('plans','source_sha256')},indent=2),flush=True)
    if failure:raise SystemExit(2)
if __name__=='__main__':main()
