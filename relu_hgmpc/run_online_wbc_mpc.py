"""Causal online MPC/WBC scheduling benchmark with a declared release delay."""
import argparse,csv,hashlib,json,pathlib,time
import numpy as np
import pinocchio as pin
import native
from common_wbc_reference import CommonReference
from online_wbc_mpc import OnlineMPC,ReferenceMux,FeedbackWBC,Schedule,pose


class Target:
    def __init__(self,plant,case,duration,out):
        begin=time.perf_counter();self.dt=.005
        path=out/f'target_{case}_{duration:g}.npz'
        if not path.exists():
            reference=CommonReference(plant,case,duration+.5)
            samples=[reference.sample(k*self.dt) for k in range(round((duration+.5)/self.dt)+1)]
            data={key:np.array([s[key] for s in samples]) for key in samples[0]}
            np.savez_compressed(path,**data)
        with np.load(path) as data:self.values={key:data[key] for key in data.files}
        self.samples=[{key:val[i] for key,val in self.values.items()} for i in range(len(self.values['q']))]
        self.precompute_ms=1000*(time.perf_counter()-begin)
        digest=hashlib.sha256()
        for key in sorted(self.values):digest.update(self.values[key].tobytes())
        self.hash=digest.hexdigest()
    def at(self,t):
        idx=round(t/self.dt)
        if abs(idx*self.dt-t)>1e-8:raise ValueError(f'Off-grid target time {t}')
        return self.samples[idx]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',choices=['normal','double','offset','push'],default='normal')
    ap.add_argument('--policy',choices=['fixed50','fixed10','fixed5','event'],default='fixed50')
    ap.add_argument('--duration',type=float,default=12.);ap.add_argument('--name',required=True)
    ap.add_argument('--delay',type=float,default=.02)
    ap.add_argument('--output-dir',default='results/online_wbc_mpc_50hz')
    args=ap.parse_args();root=pathlib.Path(__file__).resolve().parents[1];out=root/args.output_dir;out.mkdir(exist_ok=True,parents=True)
    periods={'fixed50':.02,'fixed10':.1,'fixed5':.2}
    if args.delay<0 or args.policy in periods and args.delay>periods[args.policy]+1e-10:
        raise ValueError('Release delay must not throttle the requested MPC frequency')
    if pathlib.Path(args.name).name!=args.name:raise ValueError(args.name)
    p=native.Plant(str(root));s=p.truth();home=s['q'].copy()
    target=Target(p,'normal' if args.case=='push' else args.case,args.duration,out)
    init=time.perf_counter();mpc=OnlineMPC(p,root,target);mux=ReferenceMux(p,s);wbc=FeedbackWBC(p,'se3')
    scheduler=Schedule('event' if args.policy=='event' else 'fixed',periods.get(args.policy,.1))
    initialization_ms=1000*(time.perf_counter()-init)
    rows=[];plans=[];details=[];pending=None;failed=None;advanced=0;last_acc=np.zeros(24)
    p.instrument=True;p.instrument_base=True;dt=.005;delayticks=round(args.delay/dt)
    assert abs(delayticks*dt-args.delay)<1e-9
    for k in range(round(args.duration/dt)):
        t=k*dt;start=time.perf_counter();s=p.truth();planms=0.;continuity=0.;reason='';meta={}
        try:
            if pending is not None and k>=pending[1]:
                continuity=mux.install(pending[0],t);pending=None
            reason=scheduler.due(t,pending is not None)
            if reason:
                begin=time.perf_counter();activation=t+args.delay;anchor=mux.state(activation)
                plan=mpc.solve(s,activation,wbc.prev_tau,anchor[2],anchor=anchor,first_slew_time=max(args.delay,.005))
                planms=1000*(time.perf_counter()-begin);pending=(plan,k+delayticks);scheduler.mark(t)
                plans.append(dict(time=t,release_time=t+args.delay,reason=reason,wall_ms=planms,
                    deadline_miss=bool(planms>1000*args.delay),**mpc.last_info))
                if delayticks==0:continuity=mux.install(plan,t);pending=None
            wr=time.perf_counter();ref=mux.reference(t);tau,acc=wbc.control(s,ref)
            scheduler.observe(wbc.feedback);wbcms=1000*(time.perf_counter()-wr)
            totalms=1000*(time.perf_counter()-start);last_acc=acc.copy()
        except Exception as exc:failed=dict(time=t,reason=str(exc));break
        goal=target.at(t);row=dict(time=t,total_ms=totalms,planner_ms=planms,wbc_reference_ms=wbcms,
            replan=int(bool(reason)),plan_age=t-scheduler.last,reference_continuity=continuity,
            reference_acc_max=float(np.max(np.abs(ref['a'][6:]))),**wbc.info,**wbc.feedback)
        for frame,T in [('base',pose(s['q'])),('ee',pin.SE3(s['pose'][:3,:3],s['pose'][:3,3]))]:
            g=goal[frame+'_pose'];rp=ref[frame+'_pose']
            row[frame+'_position_error_mm']=float(np.linalg.norm(T.translation-g[:3,3])*1000)
            row[frame+'_rotation_error_deg']=float(np.linalg.norm(pin.log3(g[:3,:3].T@T.rotation))*180/np.pi)
            row[frame+'_plan_position_error_mm']=float(np.linalg.norm(T.translation-rp[:3,3])*1000)
            row[frame+'_plan_rotation_error_deg']=float(np.linalg.norm(pin.log3(rp[:3,:3].T@T.rotation))*180/np.pi)
        row.update(contacts=s['contacts'],joint_position_violation=float(max(0,np.max(s['q_min']-s['q'][7:]),np.max(s['q'][7:]-s['q_max']))),
            joint_velocity_violation=float(max(0,np.max(np.abs(s['v'][6:])-s['v_max']))),clips=0)
        details.append(dict(q=s['q'],v=s['v'],tau=tau,acc=acc,reference_q=ref['q'],reference_v=ref['v'],reference_a=ref['a']))
        if not np.all(np.isfinite(tau)) or s['q'][2]<.35 or row['joint_position_violation']>1e-5 or row['joint_velocity_violation']>1e-4:
            failed=dict(time=t,reason='Physical gate',row=row);rows.append(row);break
        if args.case=='push':p.set_base_wrench(np.array([120.,0.,0.]) if 6.<=t<6.15 else np.zeros(3),np.zeros(3))
        row['clips']=p.advance(tau,acc,5);advanced+=1;rows.append(row)
        if k%400==0:print(args.name,{key:round(row[key],6) for key in ['time','base_position_error_mm','ee_position_error_mm','total_ms']},'plans',len(plans),flush=True)
    trace=p.take_trace();base=p.take_base_trace()
    assert trace.shape==(5*advanced,129) and base.shape==(5*advanced,20)
    if advanced:
        assert np.array_equal(trace[:,0],base[:,0])
        assert np.max(abs(base[::5,1:8]-np.array([d['q'][:7] for d in details[:advanced]])))<1e-12
    np.savez_compressed(out/f'{args.name}_trace.npz',trace=trace,base=base,home=home,
        **({key:np.array([d[key] for d in details]) for key in details[0]} if details else {}))
    if rows:
        with (out/f'{args.name}.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(args=vars(args),failed=failed,cycles=len(rows),simulated_cycles=advanced,plans=plans,
        target_sha256=target.hash,target_preparation_ms=target.precompute_ms,initialization_ms=initialization_ms,
        total_control_ms=sum(r['total_ms'] for r in rows),total_planner_ms=sum(r['planner_ms'] for r in rows),
        total_wbc_reference_ms=sum(r['wbc_reference_ms'] for r in rows),torque_clips=sum(r['clips'] for r in rows),
        source_sha256={name:hashlib.sha256((root/'relu_hgmpc'/name).read_bytes()).hexdigest() for name in
            ['online_wbc_mpc.py','run_online_wbc_mpc.py','geometric_wbc.py','common_wbc_reference.py','native.cpp']},
        prediction='N=10, dt=40 ms, 180 torque variables; frozen instantaneous dynamics/contact/Coriolis coefficients, first-order SE3 task costs, same QP for every policy; not converged nonlinear MPC. Reference-generator initial q/v/a equals committed reference at release; measured state updates all physical coefficients and actuator constraints.',
        timing='All online matrices, data movement, solver, packet construction, reference interpolation, WBC and trigger checks counted; no GPU. Serial wall times. Reference geometry precomputed and initialization separately reported.',
        delay='Declared deterministic plan-release delay; logical concurrent planner/WBC simulation executed serially. Not a real OS asynchronous real-time deployment.',
        trigger='Legacy event policy: previous WBC residual EWMA alpha=.2; min50/max200ms, NOT compliant with the new minimum50Hz requirement. fixed50=20ms; fixed10=100ms and fixed5=200ms are historical ablations only.',
        disturbance='push case only: 120 N world +x at floating-base COM during [6,6.15) s; hidden from planner dynamics, identical across policies')
    summary['execution_envelope']='All policies: arm a-a_ref+16(v-v_ref)+64(q-q_ref) in [-64*.08,+64*.08] rad/s2. Model-based moving reference envelope; not a proof of invariance under model mismatch.'
    summary['requested_mpc_period_s']=periods.get(args.policy)
    summary['planner_deadline_misses']=sum(p['deadline_miss'] for p in plans)
    summary['timing_validity']='Fixed logical release is used even for measured overruns, which are explicitly counted. Closed-loop results with overruns are not proof of actual real-time performance.'
    (out/f'{args.name}.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:summary[k] for k in ['args','failed','simulated_cycles','total_control_ms','total_planner_ms']}),flush=True)
    if failed:raise RuntimeError(failed)


if __name__=='__main__':main()
