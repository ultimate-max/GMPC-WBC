"""Matched measured-state replay; alternating solve order avoids timing order bias."""
import json,pathlib,time
import numpy as np
import native
from dual_wbc import CoordinatedWBC,inf
from geometric_wbc import pose

root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/dual_wbc'
p=native.Plant(str(root));p.fast_reference=True
controllers={mode:CoordinatedWBC(p,root,mode,joint_damping=5.) for mode in ('dual','unified')}
records=[];worst=dict(tau=0.,acc=0.,force=0.,interface_acc=0.,dynamics=0.,bound=0.,lock=0.)
for case in ('normal','double','hold','push'):
    data=np.load(out/f'dual_{case}_damped_trace.npz')
    for sample,k in enumerate(np.linspace(1,len(data['q'])-1,48,dtype=int)):
        s=p.evaluate(data['q'][k],data['v'][k]);q,v,a=(data[f'reference_{x}'][k] for x in ('q','v','a'))
        rs=p.evaluate(q,v)
        ref=dict(q=q,v=v,a=a,base_pose=pose(q).homogeneous,base_twist=v[:6],base_acc=a[:6],
            ee_pose=rs['pose'],ee_twist=rs['J']@v,ee_acc=rs['J']@a+rs['jdotv'],
            tau=data['reference_tau'][k],force=data['reference_force'][k])
        values={};times={}
        for mode in (('dual','unified') if sample%2 else ('unified','dual')):
            ctrl=controllers[mode];ctrl.prev_tau=data['tau'][k-1].copy()
            start=time.perf_counter();tau,acc=ctrl.control(s,ref);times[mode]=1000*(time.perf_counter()-start)
            values[mode]=dict(tau=tau,acc=acc,force=ctrl.debug['force'].copy())
            for field,metric in [('bound','constraint_violation'),('lock','task_lock_max')]:
                worst[field]=max(worst[field],ctrl.info[metric])
        for field in ('tau','acc','force'):
            worst[field]=max(worst[field],inf(values['dual'][field]-values['unified'][field]))
        worst['interface_acc']=max(worst['interface_acc'],controllers['dual'].info['interface_acc_residual'])
        worst['dynamics']=max(worst['dynamics'],controllers['dual'].info['dynamics_residual'])
        records.append(dict(case=case,index=int(k),time=.005*k,**{f'{m}_ms':t for m,t in times.items()},
            tau_difference=inf(values['dual']['tau']-values['unified']['tau'])))
passed=max(worst[k] for k in ('tau','acc','force'))<2e-5 and worst['bound']<2e-5 and worst['lock']<2e-7
report=dict(samples=len(records),passed=passed,worst=worst,records=records,
            definition='Same measured q/v, reference, previous torque, bounds, objective, tolerances; alternating solve order.')
for mode in ('dual','unified'):
    x=np.array([r[f'{mode}_ms'] for r in records[2:]])
    report[f'{mode}_ms']=dict(mean=float(x.mean()),p50=float(np.median(x)),p95=float(np.quantile(x,.95)))
(out/'matched_replay.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
assert passed,report['worst']
