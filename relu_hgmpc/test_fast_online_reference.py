import pathlib,json
import numpy as np
import native
from online_wbc_mpc import Segment,ReferenceMux
from threaded_online_mpc import clone_mux
root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/online_wbc_mpc_threads'
p=native.Plant(str(root));s=p.truth();rng=np.random.default_rng(17);maximum=np.zeros(3);kin=np.zeros(2)
for _ in range(20):
    q0=p.integrate(s['q'],rng.normal(size=24)*.02);q1=p.integrate(q0,rng.normal(size=24)*.03)
    seg=Segment(p,q0,rng.normal(size=24)*.1,rng.normal(size=24)*.2,q1,
        rng.normal(size=24)*.1,rng.normal(size=24)*.2,.04)
    for t in [0.,.007,.02,.035,.04]:
        p.fast_reference=False;old=seg.state(t);p.fast_reference=True;new=seg.state(t)
        errors=[np.max(abs(p.difference(old[0],new[0]))),np.max(abs(old[1]-new[1])),np.max(abs(old[2]-new[2]))]
        maximum=np.maximum(maximum,errors)
        ns=p.evaluate(new[0],new[1]);J,b=p.contact_kinematics(new[0],new[1])
        kin=np.maximum(kin,[np.max(abs(J-ns['Jc'])),np.max(abs(b-ns['cdotv']))])
        partial=seg.state(t,False);assert np.max(abs(partial[1]-new[1]))<1e-10
assert max(maximum)<1e-6 and max(kin)<1e-10,(maximum,kin)
result=dict(segment_qva_max=maximum.tolist(),contact_jacobian_and_drift_max=kin.tolist(),samples=100)
(out/'fast_reference_verification.json').write_text(json.dumps(result,indent=2));print(result)
