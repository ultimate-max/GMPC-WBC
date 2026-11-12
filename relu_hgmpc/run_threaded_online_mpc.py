"""Wall-paced 50 Hz planning / 200 Hz WBC, explicit late-result handling."""
import argparse,csv,hashlib,json,pathlib,sys,time,os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pinocchio as pin
import native
from online_wbc_mpc import ReferenceMux,FeedbackWBC,pose
from run_online_wbc_mpc import Target
from threaded_online_mpc import PlannerWorker,snapshot,clone_plan

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--backend',choices=['quadprog','osqp_screened'],default='osqp_screened')
    ap.add_argument('--execution',choices=['threaded','serial'],default='threaded')
    ap.add_argument('--case',choices=['normal','double','push'],default='normal')
    ap.add_argument('--duration',type=float,default=12.);ap.add_argument('--name',required=True)
    args=ap.parse_args();root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/online_wbc_mpc_threads';out.mkdir(exist_ok=True,parents=True)
    if pathlib.Path(args.name).name!=args.name:raise ValueError(args.name)
    sys.setswitchinterval(.0001)
    allowed=sorted(os.sched_getaffinity(0));cores={}
    for cpu in allowed:
        key=(pathlib.Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/core_id').read_text().strip(),
             pathlib.Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id').read_text().strip())
        cores.setdefault(key,cpu)
    selected=list(cores.values())[:2]
    if len(selected)<2:raise RuntimeError('Two separate CPU cores required for this experiment')
    os.sched_setaffinity(0,{selected[0]})
    p=native.Plant(str(root));p.fast_reference=True;s=p.truth();mux=ReferenceMux(p,s);wbc=FeedbackWBC(p,'se3')
    target=Target(p,'normal' if args.case=='push' else args.case,args.duration,out)
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='mpc')
    initialization=time.perf_counter();worker=pool.submit(PlannerWorker,root,target,args.backend,selected[1]).result()
    # Explicit preflight outside the real-time clock; retain symbolic workspace.
    mux.reference(0.);ss,mm=snapshot(s,mux,0.)
    pool.submit(worker.compute,ss,mm,0.,wbc.prev_tau.copy(),time.perf_counter()).result()
    initialization_ms=1000*(time.perf_counter()-initialization)
    p.instrument=True;p.instrument_base=True
    rows=[];plans=[];states=[];pending=None;ready=None;failed=None;skipped=0;installed=0
    origin=time.perf_counter()+.05;duration=args.duration;main_cpu=0.;advanced=0
    try:
        for k in range(round(duration/.005)):
            t=k*.005;deadline=origin+t
            wait=deadline-time.perf_counter()
            if wait>0:time.sleep(wait)
            begin=time.perf_counter();cpu=time.thread_time();s=p.truth();continuity=0.;install_late=0.
            if pending is not None and pending.done():
                ready=pending.result();pending=None
            if ready is not None and t>=ready[1]['release_time']-1e-9:
                plan,info=ready
                plan=clone_plan(plan,p);continuity=mux.install(plan,t)
                info.update(installed_time=t,install_late_ms=1000*(t-info['release_time']),
                    completion_from_tick_ms=1000*(info['completed_wall']-origin-info['request_time']),
                    deadline_miss=bool(info['completed_wall']>origin+info['release_time']))
                install_late=info['install_late_ms'];plans.append(info);installed+=1;ready=None
            ref=mux.reference(t)
            wt=time.perf_counter();tau,acc=wbc.control(s,ref)
            wbc_ms=1000*(time.perf_counter()-wt)
            submitted=0;serial_planner_cpu=0.
            if k%4==0:
                if pending is None and ready is None:
                    ss,mm=snapshot(s,mux,t);stamp=time.perf_counter();submitted=1
                    if args.execution=='threaded':
                        pending=pool.submit(worker.compute,ss,mm,t,wbc.prev_tau.copy(),stamp)
                    else:
                        ready=worker.compute(ss,mm,t,wbc.prev_tau.copy(),stamp)
                        serial_planner_cpu=ready[1]['worker_cpu_ms']
                else:skipped+=1
            finish=time.perf_counter();cpu_ms=1000*(time.thread_time()-cpu)-serial_planner_cpu;main_cpu+=cpu_ms
            goal=target.at(t);row=dict(time=t,start_lateness_ms=1000*(begin-deadline),
                control_response_ms=1000*(finish-deadline),main_wall_ms=1000*(finish-begin),main_cpu_ms=cpu_ms,
                wbc_ms=wbc_ms,wbc_deadline_miss=int(finish>deadline+.005),submitted=submitted,
                reference_continuity=continuity,install_late_ms=install_late,**wbc.info,**wbc.feedback)
            for frame,T in [('base',pose(s['q'])),('ee',pin.SE3(s['pose'][:3,:3],s['pose'][:3,3]))]:
                g=goal[frame+'_pose']
                row[frame+'_position_error_mm']=float(np.linalg.norm(T.translation-g[:3,3])*1000)
                row[frame+'_rotation_error_deg']=float(np.linalg.norm(pin.log3(g[:3,:3].T@T.rotation))*180/np.pi)
            states.append(dict(q=s['q'],v=s['v'],tau=tau,acc=acc,reference_q=ref['q']))
            if not np.all(np.isfinite(tau)) or s['q'][2]<.35 or max(np.max(s['q_min']-s['q'][7:]),np.max(s['q'][7:]-s['q_max']))>1e-5 or np.max(np.abs(s['v'][6:])-s['v_max'])>1e-4:
                raise RuntimeError('Physical gate')
            if args.case=='push':p.set_base_wrench(np.array([120.,0.,0.]) if 6.<=t<6.15 else np.zeros(3),np.zeros(3))
            row['clips']=p.advance(tau,acc,5);advanced+=1
            row['loop_with_sim_ms']=1000*(time.perf_counter()-deadline);rows.append(row)
            if k%400==0:print(args.name,round(t,3),'installed',installed,'skipped',skipped,
                'response_ms',round(row['control_response_ms'],3),flush=True)
    except Exception as exc:failed=dict(time=advanced*.005,error=str(exc))
    finally:
        # Final packet may be due at/after simulation end; account for its CPU
        # and latency but do not falsely count it as an applied trajectory.
        if pending is not None:
            try:ready=pending.result()
            except Exception as exc:
                if failed is None:failed=dict(time=advanced*.005,error=str(exc))
        if ready is not None:
            info=ready[1];info.update(installed_time=None,install_late_ms=None,
                completion_from_tick_ms=1000*(info['completed_wall']-origin-info['request_time']),
                deadline_miss=bool(info['completed_wall']>origin+info['release_time']));plans.append(info)
        pool.shutdown()
    elapsed=time.perf_counter()-origin
    trace=p.take_trace();base=p.take_base_trace()
    np.savez_compressed(out/f'{args.name}_trace.npz',trace=trace,base=base,
        **({key:np.array([d[key] for d in states]) for key in states[0]} if states else {}))
    if rows:
        with (out/f'{args.name}.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    info=dict(args=vars(args),failed=failed,simulated_cycles=advanced,plans=plans,main_cpu_ms=main_cpu,
        worker_cpu_ms=sum(x['worker_cpu_ms'] for x in plans),wall_duration_s=elapsed,skipped_requests=skipped,
        installed_plans=installed,initialization_ms=initialization_ms,target_sha256=target.hash,cpu_affinity=selected,
        source_sha256={name:hashlib.sha256((root/'relu_hgmpc'/name).read_bytes()).hexdigest() for name in
            ['online_wbc_mpc.py','online_osqp.py','online_assembly.hpp','threaded_online_mpc.py','run_threaded_online_mpc.py','geometric_wbc.py','native.cpp']},
        scope='Actual ThreadPoolExecutor worker with independent native/Pinocchio data. Main loop never waits for a future. Late results are applied at the next tick with C2 splice; busy request slots are counted, not queued.',
        timing='WBC deadline includes actual dispatch delay, reference sample, packet copy/install and controller. Sum thread CPU counts controller work, not overlapping wall time. Preflight/init and common targets separate; Python/WSL simulation, not a hard-real-time or hardware certification.')
    (out/f'{args.name}.json').write_text(json.dumps(info,indent=2))
    print(json.dumps({k:info[k] for k in ['args','failed','simulated_cycles','skipped_requests','main_cpu_ms','worker_cpu_ms','wall_duration_s']}),flush=True)
    if failed:raise RuntimeError(failed)

if __name__=='__main__':main()
