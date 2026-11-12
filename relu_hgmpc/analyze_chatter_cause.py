"""Targeted ablation and task/null-space decomposition of recorded switches."""
import json,pathlib
import numpy as np
import scipy.linalg as la
import native
from model import Reference
from prioritized import PrioritizedController

root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/chatter_cause';out.mkdir(exist_ok=True,parents=True)
p=native.Plant(str(root));s=p.truth();ref=Reference(p,s['q'],s['pose']);c=PrioritizedController(p,ref,6)
z=np.load(root/'results/chatter_mitigation/threshold_trace.npz')
csv=np.genfromtxt(root/'results/chatter_mitigation/threshold.csv',delimiter=',',names=True)
switches=np.flatnonzero(np.diff(csv['base_active'])!=0)+1
scale=np.r_[np.full(3,.25),np.full(3,.5),np.ones(6)]
rowscale=np.r_[np.full(3,1/.3),np.ones(3)];energy=0.;null_energy=0.;events=[]
for i in switches:
    st=p.evaluate(z['q'][i],z['v'][i]);_,_,J=c.chart(st)
    A=rowscale[:,None]*J*scale[None,:]
    dy=(z['command'][i]-z['command'][i-1])/scale
    dyN=(np.eye(12)-la.pinv(A)@A)@dy
    total=float(dy@dy);null=float(dyN@dyN);energy+=total;null_energy+=null
    events.append(dict(time=float(csv['time'][i]),null_fraction=null/total if total else 0.,scaled_jump_norm=float(np.sqrt(total))))
limits=np.r_[np.full(3,2.),np.full(3,3.),np.full(6,8.)]
smooth=np.load(root/'results/chatter_mitigation/smooth_qp_trace.npz')
utilization=float(np.max(np.abs(smooth['desired_acc'])/limits))
metrics={}
for name,mode,temporal in [('always','always',None),('smooth_temporal0','smooth_qp',0.),('smooth_temporal2','smooth_qp',2.)]:
    plant=native.Plant(str(root));st=plant.truth();reference=Reference(plant,st['q'],st['pose'])
    ctrl=PrioritizedController(plant,reference,6,activation_mode=mode)
    if ctrl.smooth is not None:ctrl.smooth.temporal=temporal
    plant.instrument=True;errors=[];commands=[];accelerations=[];residuals=[];clips=0
    for k in range(600):
        st=plant.truth();tau,acc=ctrl.control(st,k*.02)
        errors.append(np.linalg.norm(st['pose'][:3,3]-reference.pose_at(k*.02)[:3,3]))
        commands.append(ctrl.debug['command']);accelerations.append(ctrl.debug['desired_acc'])
        residuals.append(ctrl.info['distribution_task_residual'])
        clips+=plant.advance_reference(tau,ctrl.q_reference,ctrl.v_reference,20)
    trace=plant.take_trace();assert trace.shape==(12000,129)
    a=np.array(accelerations);u=trace[:,57:75];v=trace[:,111:129]
    metrics[name]=dict(position_rmse_mm=float(np.sqrt(np.mean(np.square(errors)))*1000),
        missing_contact_ms=int(np.sum((trace[:,1:5]>0).sum(axis=1)<4)),torque_clips=clips,
        task_residual_max=float(np.max(residuals)),arm_command_acc_max=float(np.max(np.abs(a[:,6:]))),
        slew_utilization=float(np.max(np.abs(a)/limits)))
    for label,sl in [('legs',slice(0,12)),('arm',slice(12,18))]:
        metrics[name][label]=dict(velocity_increment_rms=float(np.sqrt(np.mean(np.diff(v[:,sl],axis=0)**2))),
            torque_increment_rms=float(np.sqrt(np.mean(np.diff(u[:,sl],axis=0)**2))))
    np.savez_compressed(out/f'{name}_trace.npz',trace=trace,command=np.array(commands),acceleration=a)
summary=dict(switch_count=len(switches),normalized_nullspace_fraction=null_energy/energy,
    smooth_slew_limit_utilization=utilization,switch_events=events,ablations=metrics,
    scope='Same 12 s trajectory, WBC, plant and physical bounds; central allocator temporal weight alone changes between 0 and 2.')
(out/'cause_analysis.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k!='switch_events'},indent=2))
