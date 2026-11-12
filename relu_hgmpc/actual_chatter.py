import pathlib,json
import numpy as np
root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/chatter_mitigation'
results={}
for name in ['threshold','hysteresis','smooth_qp','smooth_admm','admm_se3']:
    z=np.load(out/f'{name}_trace.npz');v=z['v'][:,6:];trace=z['trace'];torque=trace[:,57:75]
    metrics={}
    for label,sl in [('legs',slice(0,12)),('arm',slice(12,18))]:
        dv=np.diff(v[:,sl],axis=0);dtau=np.diff(torque[:,sl],axis=0)
        metrics[label]=dict(velocity_step_rms=float(np.sqrt(np.mean(dv**2))),velocity_step_max=float(np.max(np.abs(dv))),
            torque_step_rms=float(np.sqrt(np.mean(dtau**2))),torque_step_max=float(np.max(np.abs(dtau))),
            torque_total_variation=float(np.sum(np.abs(dtau))))
    results[name]=metrics
(out/'actual_chatter_50hz.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
