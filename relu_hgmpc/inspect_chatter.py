import pathlib,json
import numpy as np
root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/chatter_mitigation'
for file in sorted(out.glob('*.json')):
    s=json.loads(file.read_text())
    if not isinstance(s,dict) or 'control_cycles' not in s:continue
    z=np.load(out/(file.stem+'_trace.npz'));a=z['desired_acc'];v=z['command']
    jerk=np.diff(a,axis=0)/.02
    blocks={name:dict(peak_acc=float(np.max(np.abs(a[:,sl]))),rms_acc=float(np.sqrt(np.mean(a[:,sl]**2))),
        peak_jerk=float(np.max(np.abs(jerk[:,sl]))),rms_jerk=float(np.sqrt(np.mean(jerk[:,sl]**2))),
        total_velocity_variation=float(np.sum(np.abs(np.diff(v[:,sl],axis=0)))))
        for name,sl in [('linear',slice(0,3)),('angular',slice(3,6)),('arm',slice(6,12))]}
    fields=['name','failed','control_cycles','position_rmse_mm','rotation_rmse_deg','contacts_min','missing_contact_steps',
        'switch_count','controller_ms_mean','controller_ms_p95','max_task_residual','max_slew_violation','max_contact_post','torque_clips']
    print(json.dumps({**{k:s[k] for k in fields},'slip_mm':max(s['per_foot_slip_path_mm']),'blocks':blocks}))
