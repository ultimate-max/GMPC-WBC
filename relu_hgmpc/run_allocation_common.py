"""Established 6D allocation+WBC on the online MPC's common tool targets.

No retuning: hysteresis, gains, hierarchy and final correction QP unchanged.
200 Hz allocation AND WBC; compare complete architectures, not the same QP.
"""
import argparse,csv,json,pathlib,time,os,hashlib
import numpy as np
import pinocchio as pin
import native
from prioritized import PrioritizedController
from geometric_wbc import rolling_contact,pose
from run_online_wbc_mpc import Target

class ToolTarget:
    def __init__(self,target):self.target=target
    def pose_at(self,t):return self.target.at(t)['ee_pose']
    def twist(self,t):return self.target.at(t)['ee_twist']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',choices=['normal','double','push'],default='normal')
    ap.add_argument('--contact',choices=['legacy','rolling'],default='rolling');ap.add_argument('--duration',type=float,default=12.)
    ap.add_argument('--name',required=True);args=ap.parse_args()
    root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/allocation_vs_online_mpc';out.mkdir(exist_ok=True,parents=True)
    if pathlib.Path(args.name).name!=args.name:raise ValueError(args.name)
    cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu})
    p=native.Plant(str(root));s=p.truth();home=s['q'].copy()
    target=Target(p,'normal' if args.case=='push' else args.case,args.duration,root/'results/online_wbc_mpc_threads')
    begin=time.perf_counter();ctrl=PrioritizedController(p,ToolTarget(target),task_dim=6,dt=.005,
        sigma_min=.1,activation_mode='hysteresis',task_error='cartesian')
    initialization_ms=1000*(time.perf_counter()-begin)
    p.instrument=True;p.instrument_base=True;rows=[];states=[];failed=None;advanced=0;total_cpu=0.
    origin=time.perf_counter()+.05
    for k in range(round(args.duration/.005)):
        t=k*.005;deadline=origin+t;wait=deadline-time.perf_counter()
        if wait>0:time.sleep(wait)
        start=time.perf_counter();clockcpu=time.thread_time();s=p.truth()
        try:
            tau,acc=ctrl.control(rolling_contact(s) if args.contact=='rolling' else s,t)
        except Exception as exc:
            failed=dict(time=t,error=str(exc));break
        finish=time.perf_counter();cpu_ms=1000*(time.thread_time()-clockcpu);total_cpu+=cpu_ms
        row=dict(time=t,control_cpu_ms=cpu_ms,control_wall_ms=1000*(finish-start),
            control_response_ms=1000*(finish-deadline),wbc_deadline_miss=int(finish>deadline+.005),**ctrl.info)
        goal=target.at(t);R=pose(s['q']).rotation;Rd=pin.rpy.rpyToMatrix(ctrl.desired[3:6])
        row.update(ee_position_error_mm=float(np.linalg.norm(s['pose'][:3,3]-goal['ee_pose'][:3,3])*1000),
            ee_rotation_error_deg=float(np.linalg.norm(pin.log3(goal['ee_pose'][:3,:3].T@s['pose'][:3,:3]))*180/np.pi),
            internal_base_position_error_mm=float(np.linalg.norm(s['q'][:3]-ctrl.desired[:3])*1000),
            internal_base_rotation_error_deg=float(np.linalg.norm(pin.log3(Rd.T@R))*180/np.pi),
            base_offset_mm=float(np.linalg.norm(s['q'][:3]-home[:3])*1000),
            base_rotation_offset_deg=float(np.linalg.norm(pin.log3(pose(home).rotation.T@R))*180/np.pi))
        states.append(dict(q=s['q'],v=s['v'],tau=tau,acc=acc,internal_base_goal=ctrl.desired[:6].copy()))
        if not np.all(np.isfinite(tau)) or s['q'][2]<.35 or max(np.max(s['q_min']-s['q'][7:]),np.max(s['q'][7:]-s['q_max']))>1e-5 or np.max(np.abs(s['v'][6:])-s['v_max'])>1e-4:
            failed=dict(time=t,error='Physical state gate');break
        if args.case=='push':p.set_base_wrench(np.array([120.,0.,0.]) if 6.<=t<6.15 else np.zeros(3),np.zeros(3))
        row['clips']=p.advance_reference(tau,ctrl.q_reference,ctrl.v_reference,5);advanced+=1;rows.append(row)
        if k%400==0:print(args.name,round(t,3),'tool_mm',round(row['ee_position_error_mm'],4),'response_ms',round(row['control_response_ms'],3),flush=True)
    trace=p.take_trace();base=p.take_base_trace()
    np.savez_compressed(out/f'{args.name}_trace.npz',trace=trace,base=base,home=home,
        **({key:np.array([x[key] for x in states]) for key in states[0]} if states else {}))
    if rows:
        with (out/f'{args.name}.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    info=dict(args=vars(args),failed=failed,simulated_cycles=advanced,total_cpu_ms=total_cpu,
        initialization_ms=initialization_ms,target_sha256=target.hash,cpu_affinity=[cpu],
        source_sha256={f:hashlib.sha256((root/'relu_hgmpc'/f).read_bytes()).hexdigest() for f in
            ['run_allocation_common.py','prioritized.py','native.cpp','geometric_wbc.py']},
        scope='6D engineering extension, hysteresis retained. Both allocation and WBC at 200Hz. Only common tool pose/twist supplied; base goals generated internally. Not original-paper 3D experiments or identical objective/constraint set to online MPC.',
        contact_note='legacy: foot sphere center xyz anchored; rolling: only world-normal position stabilized, identical native rolling point Jacobian. Final base-acceleration correction QP unchanged.',
        timing='Wall-paced 5ms loop. CPU includes model/gradient/allocation/hierarchy/QP/checks; targets/init, logs, MuJoCo excluded. No hardware dropped-frame model.')
    (out/f'{args.name}.json').write_text(json.dumps(info,indent=2));print(json.dumps({k:info[k] for k in ['args','failed','simulated_cycles','total_cpu_ms']}),flush=True)
    if failed:raise RuntimeError(failed)

if __name__=='__main__':main()
