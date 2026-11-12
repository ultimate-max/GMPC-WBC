import pathlib,json,time
import numpy as np
import pinocchio as pin
import native
from online_wbc_mpc import Segment,ReferenceMux,OnlineMPC,Schedule,pose,IDX
from run_online_wbc_mpc import Target
import geometric_wbc
root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/online_wbc_mpc';out.mkdir(parents=True,exist_ok=True)
p=native.Plant(str(root));s=p.truth();rng=np.random.default_rng(15)
q0=s['q'];q1=p.integrate(q0,rng.normal(size=24)*.02)
v0=rng.normal(size=24)*.1;v1=rng.normal(size=24)*.1;a0=rng.normal(size=24)*.1;a1=rng.normal(size=24)*.1
seg=Segment(p,q0,v0,a0,q1,v1,a1,.1)
endpoint=0.
for t,q,v,a in [(0.,q0,v0,a0),(.1,q1,v1,a1)]:
    qq,vv,aa=seg.state(t);endpoint=max(endpoint,np.max(abs(p.difference(q,qq))),np.max(abs(v-vv)),np.max(abs(a-aa)))
assert endpoint<1e-7,endpoint
t=.043;h=1e-5;q,v,a=seg.state(t);qp=seg.state(t+h)[0];qm=seg.state(t-h)[0]
dp=p.difference(q,qp);dm=p.difference(q,qm)
velocity=np.max(abs((dp-dm)/(2*h)-v));acceleration=np.max(abs((dp+dm)/(h*h)-a))
assert velocity<2e-6 and acceleration<2e-5,(velocity,acceleration)
target=Target(p,'normal',2.,out);mpc=OnlineMPC(p,root,target)
plan=mpc.solve(s,0.,target.at(0.)['tau'],np.zeros(24));mux=ReferenceMux(p,s)
continuity=mux.install(plan,.02);assert continuity<1e-8
# Actual state feedback must alter the optimized trajectory for a common target.
q=p.integrate(s['q'],np.r_[.002,np.zeros(23)]);shifted=p.evaluate(q,s['v'])
plan2=mpc.solve(shifted,0.,target.at(0.)['tau'],np.zeros(24))
change=float(np.linalg.norm(plan2.torques-plan.torques));assert change>1e-5
# The release-state constraint removes the tracking-reset interpolation spike.
release=.1;anchor=mux.state(release)
plan3=mpc.solve(s,release,target.at(0.)['tau'],anchor[2],anchor=anchor)
handover=mux.install(plan3,release);assert handover<1e-8
for d in [0.,.01,.025,.04]:
    q,v,a=mux.state(release+d);qp,vp,ap=plan3.state(release+d)
    assert np.max(abs(p.difference(q,qp)[IDX]))<1e-8 and np.max(abs(v[IDX]-vp[IDX]))<1e-8 and np.max(abs(a[IDX]-ap[IDX]))<1e-6
    ns=p.evaluate(q,v)
    assert np.max(abs(ns['Jc']@v))<1e-9 and np.max(abs(ns['Jc']@a+ns['cdotv']))<1e-8
t=release+.021;h=1e-4;q,v,a=mux.state(t);qp=mux.state(t+h)[0];qm=mux.state(t-h)[0]
dp=p.difference(q,qp);dm=p.difference(q,qm)
leg_fd=float(np.max(abs((dp-dm)/(2*h)-v)))
assert leg_fd<1e-5,leg_fd
# Optional force interface must be passive at zero and have the correct sign.
p0=native.Plant(str(root));p1=native.Plant(str(root));tau=target.at(0.)['tau']
p1.set_base_wrench(np.zeros(3),np.zeros(3));p0.advance(tau,np.zeros(24),5);p1.advance(tau,np.zeros(24),5)
assert np.array_equal(p0.truth()['q'],p1.truth()['q']) and np.array_equal(p0.truth()['v'],p1.truth()['v'])
p1.set_base_wrench(np.array([60.,0,0]),np.zeros(3));p0.advance(tau,np.zeros(24),5);p1.advance(tau,np.zeros(24),5)
dv=float(p1.truth()['v'][0]-p0.truth()['v'][0]);assert dv>0
sch=Schedule('event');assert sch.due(0.)=='initial';sch.mark(0.)
fb=dict(task_residual_score=0.,torque_margin=.5,friction_margin=.5,normal_margin=.5)
for i in range(1,40):sch.observe(fb);assert sch.due(i*.005) is None
assert sch.due(.2)=='max_age'
sch.mark(.2);fb['task_residual_score']=10.
for i in range(10):sch.observe(fb)
assert sch.due(.249) is None and sch.due(.25)=='wbc_feedback' and sch.due(.25,True) is None
# An explicitly exercised backup solve must recover correct inequality duals
# and pass the same unregularized KKT gate, not merely return a feasible point.
ctrl=geometric_wbc.GeometricWBC(p,'se3');original=geometric_wbc.quadprog.solve_qp
try:
    def fail_active_set(*args,**kwargs):raise ValueError('injected active-set breakdown')
    geometric_wbc.quadprog.solve_qp=fail_active_set
    xx,rr,_=ctrl.solve_level(0,np.eye(2),np.array([-2.,.5]),np.eye(2),-np.ones(2),np.ones(2),np.zeros(2))
    assert np.max(abs(xx-np.array([1.,-.5])))<1e-7 and ctrl.fallbacks==1
finally:geometric_wbc.quadprog.solve_qp=original
result=dict(segment_endpoint_error=float(endpoint),velocity_fd_error=float(velocity),acceleration_fd_error=float(acceleration),
    splice_error=continuity,committed_handover_error=handover,state_feedback_torque_change=change,
    external_force_velocity_change=dv,zero_force_passive=True,rolling_reference_velocity_error=leg_fd,
    initial_planner=mpc.last_info,causal_trigger_tests=True,backup_solver_verified=True)
(out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
