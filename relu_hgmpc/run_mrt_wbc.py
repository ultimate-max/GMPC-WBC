"""Reference-style producer/consumer: Python MPC owner + pure C++ WBC owner.

MPC publishes complete, timestamped nominal policies. WBC never calls Python,
waits for a solve, or consumes a FIFO of old commands. All simulation/control
objects belong to the C++ control thread; MPC owns separate Pinocchio data.
"""
import argparse,hashlib,json,os,pathlib,time,traceback
import numpy as np
import pinocchio as pin
import native
from strict_gmpc import StrictGMPC
from model import equilibrium
from runtime_support import DenseTarget,physical_cores,stats
from native_runtime_support import kernel_inputs,publish
from geometric_wbc import pose

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True)
    ap.add_argument('--case',choices=['normal','double','hold','push'],default='normal')
    ap.add_argument('--wbc-mode',choices=['dual','unified'],default='dual')
    ap.add_argument('--wbc-hz',type=int,choices=[200,400],default=400)
    ap.add_argument('--duration',type=float,default=12.)
    ap.add_argument('--worker-delay-ms',type=float,default=0.)
    args=ap.parse_args();root=pathlib.Path(__file__).resolve().parents[1]
    if pathlib.Path(args.name).name!=args.name or not 0<args.duration<=12:raise ValueError('Invalid output name/duration')
    out=root/'results/buffered_wbc';out.mkdir(exist_ok=True)
    cores=physical_cores()
    if len(cores)<2:raise RuntimeError('Two distinct physical cores required')
    os.sched_setaffinity(0,{cores[1]})
    p=native.Plant(str(root));p.fast_reference=True;p.reference_hold_gil=True
    target=DenseTarget(p,root,args.case);s=p.truth();home=s['q'].copy();prev=equilibrium(s)
    b,a,lib=kernel_inputs(root)
    runtime=native.ControlRuntime(str(root),b,a,lib,prev,1/args.wbc_hz,args.wbc_mode=='dual',args.case=='push')
    planner=StrictGMPC(p,root,target,joint_damping=5.)
    initial=time.perf_counter();plan=planner.solve(s,0.,prev,np.zeros(24),first_slew_time=.02)
    publish(runtime,plan,0);initialization_ms=1000*(time.perf_counter()-initial)
    events=[];sequence=1;last_observation=-1;next_start=time.perf_counter()+.03
    runtime.start(args.duration,cores[0])
    try:
        while runtime.running():
            delay=next_start-time.perf_counter()
            if delay>0:time.sleep(min(delay,.02))
            if not runtime.running():break
            start=time.perf_counter();cpu=time.thread_time();obs=runtime.observation()
            if obs['sequence']==last_observation:
                time.sleep(.001);continue
            last_observation=obs['sequence'];next_start=start+.02
            event=dict(sequence=sequence,time=obs['time'],observation_sequence=obs['sequence'])
            try:
                if args.worker_delay_ms:time.sleep(args.worker_delay_ms/1000)
                s=p.evaluate(obs['q'],obs['v']);s['contacts']=obs['contacts']
                plan=planner.solve(s,obs['time'],obs['tau'],obs['acc'],first_slew_time=.02)
                publish(runtime,plan,sequence)
                event.update(success=True,**planner.last_info)
            except Exception as exc:
                # The consumer keeps its last valid policy, then stops on
                # expiry. Failed or uncertified trajectories are never sent.
                event.update(success=False,error=str(exc),traceback=traceback.format_exc())
            event.update(total_ms=1000*(time.perf_counter()-start),cpu_ms=1000*(time.thread_time()-cpu))
            events.append(event);sequence+=1
            if len(events)%100==0:print(args.name,'sim',round(obs['time'],3),'MPC ms',round(event['total_ms'],3),flush=True)
    finally:runtime.finish()
    data=runtime.results();arrays={k:v for k,v in data.items() if isinstance(v,np.ndarray)}
    np.savez_compressed(out/f'{args.name}_trace.npz',**arrays,home=home)
    rows=data['rows'];dt=1/args.wbc_hz
    keys=['time','control_ms','wbc_ms','reference_ms','exchange_ms','start_lateness_ms','response_ms','full_loop_ms','contacts','clips','policy_sequence','policy_age_ms','installed','constraint_violation','task_lock_max','kkt','dynamics_residual','contact_residual','interface_residual']
    np.savetxt(out/f'{args.name}.csv',rows,delimiter=',',header=','.join(keys),comments='')
    result=dict(args=vars(args),failure=data['failure'] or None,completed_seconds=len(rows)*dt,
        wall_seconds=data['wall_seconds'],achieved_wbc_hz=len(rows)/data['wall_seconds'],
        installed_per_wall_second=max(0,data['accepted']-1)/data['wall_seconds'],accepted=data['accepted'],rejected=data['rejected'],overwritten=data['overwritten'],
        initialization_ms=initialization_ms,mpc_events=events,target_hash=target.hash,cpu_affinity=cores[:2],
        model='unchanged frozen affine, first-order geometric errors, N=10, h=40 ms',
        tasks='hard constraints > EE pose/rate > base pose/rate > redundancy/smoothing',
        thread_contract='MPC main Python thread produces nominal policy; pure C++ std::thread consumes at 400/200 Hz with geometric WBC feedback. Capacity-one immutable policy/observation buffers; consumer uses try_lock and never waits for MPC.',
        timing_scope='MPC total includes snapshot/model evaluation, construction, solve, validation, packing and publication. WBC full_loop includes observation, policy exchange, Lie interpolation, contact reconstruction, geometry, dual/unified WBC, checks and 2 kHz MuJoCo/actuator stepping. Diagnostics/error evaluation below are offline. Simulator is control-owned, not independently clocked HIL.',
        source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in pathlib.Path(__file__).parent.glob('*') if f.suffix in ('.py','.cpp','.hpp')})
    if len(rows):
        result.update(minimum_contacts=int(rows[:,8].min()),torque_clips=int(rows[:,9].sum()),
            control_deadline_miss_fraction=float(np.mean(rows[:,6]>1000*dt)),
            full_deadline_miss_fraction=float(np.mean(rows[:,5]+rows[:,7]>1000*dt)),
            planner_total_ms=stats([e['total_ms'] for e in events]),
            successful_plans=sum(e['success'] for e in events),failed_plans=sum(not e['success'] for e in events))
        for i in [1,2,3,4,5,6,7,11,13,14,15,16,17,18]:result[keys[i]]=stats(rows[:,i])
        for frame in ('base','ee'):
            errors=[];angles=[]
            for k,t in enumerate(rows[:,0]):
                T=pose(data['q'][k]).homogeneous if frame=='base' else data['ee_pose_columns'][k].reshape((4,4),order='F')
                D=target.at(t)[frame+'_pose'];errors.append(1000*np.linalg.norm(T[:3,3]-D[:3,3]));angles.append(np.linalg.norm(pin.log3(D[:3,:3].T@T[:3,:3]))*180/np.pi)
            for label,values in [('position_error_mm',errors),('rotation_error_deg',angles)]:result[frame+'_'+label]=dict(rms=float(np.sqrt(np.mean(np.square(values)))),max=float(max(values)))
        trace=data['trace']
        if len(trace)>100:
            from scipy.signal import butter,sosfiltfilt
            mask=trace[:,0]>=min(1.,len(rows)*dt/2)
            for name,sl in [('leg_velocity',slice(111,123)),('arm_velocity',slice(123,129)),('leg_torque',slice(57,69)),('arm_torque',slice(69,75))]:
                x=trace[:,sl];filtered=sosfiltfilt(butter(4,[5,100],btype='bandpass',fs=2000,output='sos'),x,axis=0)
                result[name]=dict(hf_5_100_rms=float(np.sqrt(np.mean(filtered[mask]**2))),step_rms=float(np.sqrt(np.mean(np.diff(x[mask],axis=0)**2))))
    (out/f'{args.name}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('mpc_events','source_sha256')},indent=2),flush=True)
    if result['failure']:raise SystemExit(2)
if __name__=='__main__':main()
