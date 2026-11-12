import pathlib,json
import numpy as np
import native
from prioritized import PrioritizedController,shaped_bounds
from geometric_wbc import rolling_contact
from run_online_wbc_mpc import Target
from run_allocation_common import ToolTarget
root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/allocation_vs_online_mpc';records=[]
for case,contact in [('normal','legacy'),('double','legacy'),('double','rolling')]:
    p=native.Plant(str(root));target=Target(p,case,12.,root/'results/online_wbc_mpc_threads')
    ctrl=PrioritizedController(p,ToolTarget(target),task_dim=6,dt=.005,activation_mode='hysteresis')
    old=json.loads((out/f'{case}_allocation_{contact}_r1.json').read_text())
    for k in range(2400):
        t=k*.005;s=p.truth()
        try:tau,acc=ctrl.control(rolling_contact(s) if contact=='rolling' else s,t)
        except Exception as exc:
            q=ctrl.coordinates(s);home=ctrl.coordinates(dict(q=ctrl.home))
            lo=np.r_[home[:3]+[-.15,-.15,-.08],np.full(3,-.25),s['q_min'][12:]+.02]
            hi=np.r_[home[:3]+[.15,.15,.08],np.full(3,.25),s['q_max'][12:]-.02]
            vmax=np.r_[np.full(3,.25),np.full(3,.5),np.minimum(s['v_max'][12:],4.)]
            amax=np.r_[np.full(3,2.),np.full(3,3.),np.full(6,8.)]
            lower=np.maximum.reduce([(lo-q)/.005,-vmax,-np.sqrt(np.maximum(0,2*amax*(q-lo)))])
            upper=np.minimum.reduce([(hi-q)/.005,vmax,np.sqrt(np.maximum(0,2*amax*(hi-q)))])
            names=['base_x','base_y','base_z','base_roll','base_pitch','base_yaw']+[f'arm_{j+1}' for j in range(6)]
            bad=[dict(coordinate=names[i],value=float(q[i]),lower_position=float(lo[i]),upper_position=float(hi[i]),
                lower_velocity=float(lower[i]),upper_velocity=float(upper[i])) for i in np.flatnonzero(lower>upper+1e-8)]
            assert abs(t-old['failed']['time'])<1e-9
            record=dict(case=case,contact=contact,time=t,error=str(exc),incompatible_bounds=bad,
                previous_contact_acc_residual=ctrl.info['wbc_contact_acc_residual'],contacts=s['contacts'])
            records.append(record);print(json.dumps(record),flush=True);break
        p.advance_reference(tau,ctrl.q_reference,ctrl.v_reference,5)
    else:raise RuntimeError('Expected failure did not reproduce')
(out/'boundary_diagnosis.json').write_text(json.dumps(records,indent=2))
comparison=json.loads((out/'comparison.json').read_text())
for case in ['normal','double','push']:
    print(case,{m:{k:values[k] for k in ['complete','missing_contact_ms','min_contacts','contact_speed_rms_mm_s']} for m,values in comparison['means'][case].items()})
