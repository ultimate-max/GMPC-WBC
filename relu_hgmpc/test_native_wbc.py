import json,pathlib,time
import numpy as np
import pinocchio as pin
import native
from dual_wbc import CoordinatedWBC
from native_runtime_support import kernel_inputs
from geometric_wbc import pose
root=pathlib.Path(__file__).resolve().parents[1]
p=native.Plant(str(root));home=p.truth()['q'];b,a,lib=kernel_inputs(root)
data=np.load(root/'results/buffered_wbc/legacy400_matched_trace.npz')
result=[]
for mode in ('dual','unified'):
    py=CoordinatedWBC(p,root,mode,dt=.0025,joint_damping=5.)
    cpp=native.WbcKernel(str(root/'models/b2_z1.urdf'),b,a,lib,home,py.prev_tau,.0025,mode=='dual')
    max_tau=max_acc=max_force=max_task=0.;native_ms=[];python_ms=[];kkt=0.
    for k in range(0,len(data['q']),20):
        s=p.evaluate(data['q'][k],data['v'][k]);q=data['reference_q'][k];v=data['reference_v'][k];acc=data['reference_a'][k]
        rs=p.evaluate(q,v)
        previous=data['tau'][max(0,k-1)];py.prev_tau=previous.copy();cpp.set_previous(previous)
        r=dict(q=q,v=v,a=acc,tau=previous,force=np.linalg.lstsq(s['Jc'][:,:6].T,s['h'][:6],rcond=None)[0],
            base_pose=pose(q).homogeneous,base_twist=v[:6],base_acc=acc[:6],ee_pose=rs['pose'],ee_twist=rs['J']@v,ee_acc=rs['J']@acc+rs['jdotv'])
        st=time.perf_counter();tau,aa=py.control(s,r);python_ms.append(1000*(time.perf_counter()-st))
        st=time.perf_counter();out=cpp.control(s,r);native_ms.append(1000*(time.perf_counter()-st))
        max_tau=max(max_tau,float(np.max(np.abs(tau-out['tau']))));max_acc=max(max_acc,float(np.max(np.abs(aa-out['acc']))));max_force=max(max_force,float(np.max(np.abs(py.debug['force']-out['force']))));kkt=max(kkt,out['kkt'])
        # Task outputs/objectives must match; row ordering of constraints and
        # low-priority posture rows is deliberately allowed to differ.
        for i in (0,1):
            C,d=py.last_problem['tasks'][i]
            max_task=max(max_task,float(np.max(np.abs(C-out[f'C{i}']))),float(np.max(np.abs(d-out[f'd{i}']))))
        assert max_tau<2e-5 and max_acc<2e-5 and max_force<2e-4 and max_task<1e-8,(mode,k,max_tau,max_acc,max_force,max_task)
    result.append(dict(mode=mode,snapshots=len(native_ms),torque_difference=max_tau,acceleration_difference=max_acc,force_difference=max_force,task_matrix_difference=max_task,kkt=kkt,native_mean_ms=float(np.mean(native_ms)),python_mean_ms=float(np.mean(python_ms))))
(root/'results/buffered_wbc/native_equivalence.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
