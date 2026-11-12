"""Verify 20 ms replans inside 40 ms prediction/interpolation segments."""
import json,pathlib
import numpy as np
import native
from online_wbc_mpc import OnlineMPC,ReferenceMux,Schedule
from run_online_wbc_mpc import Target

root=pathlib.Path(__file__).resolve().parents[1]
out=root/'results/online_wbc_mpc_50hz';out.mkdir(parents=True,exist_ok=True)
sch=Schedule('fixed',.02);pending=None;requests=[];releases=[]
for k in range(2400):
    if pending is not None and k==pending:releases.append(k);pending=None
    reason=sch.due(k*.005,pending is not None)
    if reason:requests.append(k);sch.mark(k*.005);pending=k+4
assert np.array_equal(requests,np.arange(0,2400,4))
assert np.array_equal(releases,np.arange(4,2400,4))
p=native.Plant(str(root));s=p.truth();target=Target(p,'normal',2.,out)
mpc=OnlineMPC(p,root,target);mux=ReferenceMux(p,s);continuity=[];velocity_fd=[]
max_reference_speed=0.
for release in [1.02,1.04,1.06,1.08,1.10]:
    anchor=mux.state(release)
    plan=mpc.solve(s,release,target.at(0.)['tau'],anchor[2],anchor=anchor,first_slew_time=.02)
    continuity.append(mux.install(plan,release))
    t=release+.007;h=1e-5;q,v,a=mux.state(t)
    max_reference_speed=max(max_reference_speed,float(np.linalg.norm(v)))
    dp=p.difference(q,mux.state(t+h)[0]);dm=p.difference(q,mux.state(t-h)[0])
    velocity_fd.append(float(np.max(abs((dp-dm)/(2*h)-v))))
assert max(continuity)<1e-8 and max(velocity_fd)<1e-5 and max_reference_speed>.01
result=dict(requests_per_12s=len(requests),request_period_ms=20,release_delay_ms=20,
    prediction_step_ms=40,handover_max=max(continuity),velocity_fd_max=max(velocity_fd),
    max_reference_speed=max_reference_speed)
(out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
