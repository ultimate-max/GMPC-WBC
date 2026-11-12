"""Same 400 Hz/2 kHz plant and external target as native MPC+WBC.

Preserves the existing paper's 6D engineering adaptation and threshold SNS.
Changes only the multi-rate clock: exactly one 50 Hz allocation per 8 ticks.
This remains Python, so timing is an implementation comparison, not an
algorithmic speedup or a reproduction of the authors' reported runtime.
"""
import argparse,csv,hashlib,json,os,pathlib,time,traceback
import numpy as np
import native
from prioritized import PrioritizedController
from paper_anchor_baseline import EEReference
from geometric_wbc import rolling_contact
from runtime_support import DenseTarget,physical_cores,stats

class Paper50(PrioritizedController):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.bucket=-1;self.allocations=0
    def allocate(self,s,t):
        bucket=round(t/.0025)//8
        self.did_allocate=bucket!=self.bucket
        if self.did_allocate:
            self.cached=super().allocate(s,t);self.bucket=bucket;self.allocations+=1
        _,command,acc,ff,error=self.cached
        return self.coordinates(s),command,acc,ff,error

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',default='paper400_normal');ap.add_argument('--duration',type=float,default=12.);args=ap.parse_args()
    root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/paper_comparison';out.mkdir(parents=True,exist_ok=True)
    os.sched_setaffinity(0,{physical_cores()[0]})
    p=native.Plant(str(root));p.set_timestep(.0005);p.fast_reference=True;p.reference_hold_gil=True
    target=DenseTarget(p,root,'normal');s=p.truth();home=s['q'].copy()
    ctrl=Paper50(p,EEReference(target),task_dim=6,dt=.02)
    p.instrument=True;p.instrument_base=True;rows=[];states=[];failure=None
    origin=time.perf_counter()+.03;dt=.0025
    try:
        for k in range(round(args.duration/dt)):
            t=k*dt;deadline=origin+t;wait=deadline-time.perf_counter()
            if wait>0:time.sleep(wait)
            begin=time.perf_counter();cpu=time.thread_time();s=p.truth()
            tau,acc=ctrl.control(rolling_contact(s),t)
            compute_ms=1000*(time.perf_counter()-begin)
            if not np.all(np.isfinite(tau)) or s['q'][2]<.35 or max(np.max(s['q_min']-s['q'][7:]),np.max(s['q'][7:]-s['q_max']))>1e-5 or np.max(np.abs(s['v'][6:])-s['v_max'])>1e-4:raise RuntimeError('Measured physical state gate')
            response=1000*(time.perf_counter()-deadline);cpu_ms=1000*(time.thread_time()-cpu)
            clips=p.advance_reference(tau,ctrl.q_reference,ctrl.v_reference,5)
            full_ms=1000*(time.perf_counter()-begin)
            rows.append(dict(time=t,control_ms=compute_ms,control_cpu_ms=cpu_ms,response_ms=response,full_loop_ms=full_ms,
                start_lateness_ms=1000*(begin-deadline),contacts=s['contacts'],clips=clips,
                allocated=int(ctrl.did_allocate),allocation_ms=ctrl.allocation_info['distribution_ms'] if ctrl.did_allocate else 0.,
                qp_bound=ctrl.info['primal'],contact_residual=ctrl.info['wbc_contact_acc_residual'],
                dynamics_residual=ctrl.info['wbc_float_dynamics_residual'],hierarchy=ctrl.info['hierarchy']))
            states.append(dict(q=s['q'],v=s['v'],tau=tau,acc=acc,ee_pose=s['pose']))
            if k%800==0:print(args.name,t,'ms',round(compute_ms,3),flush=True)
    except Exception as exc:
        failure=dict(time=len(rows)*dt,reason=str(exc),traceback=traceback.format_exc());print(failure,flush=True)
        np.savez_compressed(out/f'{args.name}_failure_state.npz',q=s['q'],v=s['v'])
    elapsed=time.perf_counter()-origin
    trace=p.take_trace();np.savez_compressed(out/f'{args.name}_trace.npz',trace=trace,base=p.take_base_trace(),home=home,
        **{key:np.array([s[key] for s in states]) for key in states[0]})
    with (out/f'{args.name}.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result=dict(failure=failure,completed_seconds=len(rows)*dt,wall_seconds=elapsed,achieved_wbc_hz=len(rows)/elapsed,
        allocations=ctrl.allocations,expected_allocation_calls=(len(rows)+7)//8+(1 if failure and len(rows)%8==0 else 0),
        target_hash=target.hash,minimum_contacts=min(r['contacts'] for r in rows),torque_clips=sum(r['clips'] for r in rows),
        full_deadline_miss_fraction=float(np.mean([r['start_lateness_ms']+r['full_loop_ms']>2.5 for r in rows])),
        settings=dict(allocator_hz=50,wbc_hz=400,servo_hz=2000,activation_mode=ctrl.activation_mode,task_dim=6),
        scope='Existing B2+Z1 6D paper adaptation, not author code. Original adapted WBC has floating-base acceleration correction; it does not share the GMPC task/constraint set. Same external EE target, plant, actuator and clock. Python baseline versus native WBC: no algorithmic speedup attribution.',
        source_sha256={name:hashlib.sha256((root/'relu_hgmpc'/name).read_bytes()).hexdigest() for name in ('run_paper_matched.py','prioritized.py','paper_anchor_baseline.py','runtime_support.py')})
    assert result['allocations']==result['expected_allocation_calls'],result
    for key in ('control_ms','full_loop_ms','response_ms'):result[key]=stats([r[key] for r in rows])
    (out/f'{args.name}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
