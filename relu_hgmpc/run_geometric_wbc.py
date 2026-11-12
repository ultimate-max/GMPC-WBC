"""Common-reference replay; a WBC ablation, not an online MPC experiment."""
import argparse,csv,hashlib,json,pathlib,time
import numpy as np
import pinocchio as pin
import native
from geometric_wbc import GeometricWBC,pose


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['product','se3'],required=True)
    ap.add_argument('--case',choices=['normal','double','offset'],required=True)
    ap.add_argument('--duration',type=float,default=12.)
    ap.add_argument('--name',required=True)
    args=ap.parse_args()
    if pathlib.Path(args.name).name!=args.name:raise ValueError('Simple run name required')
    root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/geometric_wbc'
    reference=np.load(out/f'reference_{args.case}.npz')
    refmeta=json.loads((out/f'reference_{args.case}.json').read_text())
    dt=float(reference['time'][1]);ticks=round(dt/.001);n=round(args.duration/dt)
    assert n<len(reference['time']) and np.isclose(dt,ticks*.001)
    plant=native.Plant(str(root));home=plant.truth()['q']
    assert np.array_equal(home,reference['home'])
    init=time.perf_counter();ctrl=GeometricWBC(plant,args.mode,dt)
    init_ms=1000*(time.perf_counter()-init)
    plant.instrument=True;plant.instrument_base=True
    # Loading/unpacking the common offline plan is accounted as preparation.
    prep=time.perf_counter()
    arrays={key:reference[key] for key in reference.files if key not in ['time','home']}
    refs=[{key:array[i] for key,array in arrays.items()} for i in range(n)]
    prep_ms=1000*(time.perf_counter()-prep)
    rows=[];details=[];failed=None;advanced=0
    for k,ref in enumerate(refs):
        t=k*dt;start=time.perf_counter();s=plant.truth()
        try:tau,acc=ctrl.control(s,ref)
        except Exception as exc:
            failed=dict(time=t,reason=str(exc));break
        feedback_ms=1000*(time.perf_counter()-start)
        Tb=pose(s['q']);Te=pin.SE3(s['pose'][:3,:3],s['pose'][:3,3])
        row=dict(time=t,feedback_ms=feedback_ms,**ctrl.info)
        for name,T in [('base',Tb),('ee',Te)]:
            Td=ref[name+'_pose']
            row[name+'_position_error_mm']=1000*float(np.linalg.norm(T.translation-Td[:3,3]))
            row[name+'_rotation_error_deg']=180/np.pi*float(np.linalg.norm(pin.log3(Td[:3,:3].T@T.rotation)))
        row.update(contacts=s['contacts'],joint_position_violation=float(max(0,np.max(s['q_min']-s['q'][7:]),np.max(s['q'][7:]-s['q_max']))),
            joint_velocity_violation=float(max(0,np.max(np.abs(s['v'][6:])-s['v_max']))),clips=0)
        details.append(dict(q=s['q'],v=s['v'],tau=tau,acc=acc,force=ctrl.debug['force']))
        if not np.all(np.isfinite(tau)) or s['q'][2]<.35 or row['joint_position_violation']>1e-5 or row['joint_velocity_violation']>1e-4:
            failed=dict(time=t,reason='Physical validity gate',row=row);rows.append(row);break
        row['clips']=plant.advance(tau,acc,ticks);advanced+=1;rows.append(row)
        if k%400==0:print(args.name,{key:round(row[key],6) for key in ['time','base_position_error_mm','base_rotation_error_deg','feedback_ms']},flush=True)
    trace=plant.take_trace();base=plant.take_base_trace();final=plant.truth()
    assert trace.shape==(advanced*ticks,129) and base.shape==(advanced*ticks,20)
    if advanced:
        assert np.array_equal(trace[:,0],base[:,0])
        assert np.max(np.abs(base[::ticks,1:8]-np.array([d['q'][:7] for d in details[:advanced]])))<1e-12
        assert np.max(np.abs(base[::ticks,8:14]-np.array([d['v'][:6] for d in details[:advanced]])))<1e-12
    np.savez_compressed(out/f'{args.name}_trace.npz',trace=trace,base=base,home=home,final_q=final['q'],final_v=final['v'],
        **({key:np.array([d[key] for d in details]) for key in details[0]} if details else {}))
    if rows:
        with (out/f'{args.name}.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=dict(args=vars(args),failed=failed,cycles=len(rows),simulated_cycles=advanced,
        solver='quadprog active set, 1e-10 numerical Hessian diagonal; KKT checked against unregularized normalized objective',
        initialization_ms=init_ms,reference_unpack_ms=prep_ms,reference_sha256=refmeta['reference_sha256'],
        timing_scope='state read + derivatives/matrices + all 3 HQP solves + constraint checks; offline reference generation, initialization, logging and MuJoCo excluded; no compute-delay injection',
        scope='Offline common feasible trajectory; strict hierarchy physics > base > end effector > comfort. Only error representation and its derivatives differ. Both baseline and SE3 use SO3 log rotation. Not online MPC.',
        source_sha256={name:hashlib.sha256((pathlib.Path(__file__).parent/name).read_bytes()).hexdigest() for name in
            ['geometric_wbc.py','run_geometric_wbc.py','common_wbc_reference.py','native.cpp','model.py','osqp_acceleration.py']})
    if rows:
        for key in rows[0]:
            a=np.array([r[key] for r in rows]);summary[key]=dict(mean=float(a.mean()),max=float(a.max()),min=float(a.min()),p95=float(np.quantile(a,.95)),p99=float(np.quantile(a,.99)))
        summary['torque_clips']=int(sum(r['clips'] for r in rows))
        summary['amortized_feedback_ms']=(sum(r['feedback_ms'] for r in rows)+init_ms+prep_ms)/len(rows)
    (out/f'{args.name}.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:summary[k] for k in ['args','failed','simulated_cycles','amortized_feedback_ms'] if k in summary}),flush=True)
    if failed:raise RuntimeError(failed)


if __name__=='__main__':main()
