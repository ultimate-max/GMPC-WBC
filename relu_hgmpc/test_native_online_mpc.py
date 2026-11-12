import json,pathlib,time
import numpy as np
import native
from online_wbc_mpc import OnlineMPC,ReferenceMux
from threaded_online_mpc import snapshot,clone_mux
from run_online_wbc_mpc import Target
root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/online_wbc_mpc_threads';out.mkdir(exist_ok=True,parents=True)
p=native.Plant(str(root));target=Target(p,'double',12.,root/'results/online_wbc_mpc_50hz')
trace=np.load(root/'results/online_wbc_mpc_50hz/double_fixed50_r1_trace.npz')
engines={b:OnlineMPC(p,root,target,backend='osqp_screened' if b=='osqp' else 'quadprog',assembly='python' if b=='python' else 'native')
    for b in ['python','native','osqp']};records=[]
for k in [0,200,400,800,1200,1600,2000,2200]:
    s=p.evaluate(trace['q'][k],trace['v'][k]);t=k*.005+.02
    anchor=(trace['reference_q'][k+4],trace['reference_v'][k+4],trace['reference_a'][k+4])
    for label,engine in engines.items():
        engine.solve(s,t,trace['tau'][max(0,k-1)],anchor[2],anchor=anchor,first_slew_time=.02)
        b=engines['python'].debug;d=engine.debug;errors={}
        for key in ['H','g','G','lo','hi']:
            finite=np.isfinite(b[key]);assert np.array_equal(finite,np.isfinite(d[key]))
            errors[key]=float(np.max(np.abs(b[key][finite]-d[key][finite])))
        assert max(errors.values())<1e-9,errors
        z=d['z'];objective=.5*z@b['H']@z+b['g']@z;zb=b['z'];delta=abs(objective-(.5*zb@b['H']@zb+b['g']@zb))
        assert delta<1e-10,delta
        records.append(dict(time=t,implementation=label,matrix_errors=errors,objective_error=float(delta),**engine.last_info))
# Packet snapshot must not share mutable caches or coefficient arrays.
s=p.truth();mux=ReferenceMux(p,s);mux.reference(0.);ss,mm=snapshot(s,mux,0.)
ss['q'][0]+=1;assert ss['q'][0]!=s['q'][0]
mm.leg_cache[0.][0]+=1;assert mm.leg_cache[0.][0]!=mux.leg_cache[0.][0]
(out/'native_equivalence.json').write_text(json.dumps(records,indent=2))
for b in engines:
    rr=[x for x in records if x['implementation']==b]
    print(b,'mean planner ms',np.mean([x['planner_ms'] for x in rr]),'mean construction',np.mean([x['construction_ms'] for x in rr]),'max qp residual',max(x['qp_residual'] for x in rr))
print('all matrix/objective and snapshot isolation checks passed')
