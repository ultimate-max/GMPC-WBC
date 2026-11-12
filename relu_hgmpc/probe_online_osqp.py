import pathlib,json,time
import numpy as np
import native
from online_wbc_mpc import OnlineMPC
from online_osqp import OnlineOSQP
from run_online_wbc_mpc import Target
root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/online_wbc_mpc_threads';out.mkdir(exist_ok=True,parents=True)
p=native.Plant(str(root));target=Target(p,'double',12.,root/'results/online_wbc_mpc_50hz')
trace=np.load(root/'results/online_wbc_mpc_50hz/double_fixed50_r1_trace.npz')
baseline=OnlineMPC(p,root,target);engines={b:OnlineOSQP(screen=b=='osqp_screened') for b in ['osqp','osqp_screened']}
records=[]
for k in [0,200,400,800,1200,1600,2000,2200]:
    s=p.evaluate(trace['q'][k],trace['v'][k]);t=k*.005+.02
    anchor=(trace['reference_q'][k+4],trace['reference_v'][k+4],trace['reference_a'][k+4])
    baseline.solve(s,t,trace['tau'][max(0,k-1)],anchor[2],anchor=anchor,first_slew_time=.02)
    d=baseline.debug
    for name,eng in engines.items():
        ts=time.perf_counter();z,y,it,info=eng.solve(d['H'],d['g'],d['G'],d['lo'],d['hi'],t,.04,10)
        rec=dict(time=t,backend=name,wall_ms=1000*(time.perf_counter()-ts),baseline_ms=baseline.last_info['solve_ms'],
            solution_max_error=float(np.max(abs(z-d['z']))),objective_error=float(abs(.5*z@d['H']@z+d['g']@z-(.5*d['z']@d['H']@d['z']+d['g']@d['z']))),
            iterations=it,**info)
        records.append(rec);print(json.dumps(rec),flush=True)
(out/'qp_probe.json').write_text(json.dumps(records,indent=2))
