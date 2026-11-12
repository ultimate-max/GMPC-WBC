import json,pathlib,numpy as np,scipy.linalg as la,scipy.sparse as sp,osqp
import native,pinocchio as pin
from model import contact,equilibrium,Reference,log,ad
from controller import Controller
from relu_solver import ReLUQP

root=pathlib.Path(__file__).resolve().parents[1]
p=native.Plant(str(root));s=p.truth();q=s['q'];v=s['v'];T,a0,F,f0=contact(s)
tau=equilibrium(s);S=np.vstack((np.zeros((6,18)),np.eye(18)))
assert np.max(np.abs(s['M']@T-S-s['Jc'].T@F))<1e-9
assert np.max(np.abs(s['Jc']@T))<1e-9
assert np.max(np.abs(T@tau+a0))<1e-8
rng=np.random.default_rng(91);dq=rng.normal(size=24);eps=1e-7
sn=p.evaluate(p.integrate(q,eps*dq),v)
fd=log(np.linalg.inv(s['pose'])@sn['pose'])/eps
jerr=float(np.max(np.abs(fd-s['J']@dq)));assert jerr<2e-6,jerr
phi=1e-4*rng.normal(size=6);vd=rng.normal(size=6);dv=1e-4*rng.normal(size=6)
E=pin.exp6(phi)
exact=pin.Jlog6(E)@(vd+dv-E.inverse().action@vd)
approx=-ad(vd)@phi+dv
gerr=float(np.max(np.abs(exact-approx)));assert gerr<2e-7,gerr
ref=Reference(p,q,s['pose']);ctrl=Controller(p,ref,'hgmpc',N=5,dt=.06,iterations=300)
# Force a nonzero tracking error and secondary correction.
st=p.evaluate(p.integrate(q,np.r_[np.zeros(18),np.ones(6)*.01]),np.zeros(24))
torque,acc=ctrl.control(st,1.)
assert ctrl.info['hierarchy']<1e-8,ctrl.info
problem=ctrl.last_problem;w0=ctrl.primary_solution;w=ctrl.last_solution
cost0=.5*w0@problem['H']@w0+problem['g']@w0
cost1=.5*w@problem['H']@w+problem['g']@w
assert abs(cost1-cost0)<1e-7,(cost0,cost1)
assert np.linalg.norm(w-w0)>1e-5,'Hierarchy test must use a nonzero secondary correction'
# Actual semidefinite GMPC QP, independently solved to high accuracy.
solver=ReLUQP(problem['H'],problem['C'],rho=.1)
wr=solver.solve(problem['g'],problem['l'],problem['u'],iterations=4000,xwarm=w0)
oracle=osqp.OSQP();oracle.setup(P=sp.csc_matrix(problem['H']),q=problem['g'],A=sp.csc_matrix(problem['C']),l=problem['l'],u=problem['u'],eps_abs=1e-7,eps_rel=1e-7,max_iter=30000,verbose=False,polishing=True)
res=oracle.solve();assert res.info.status_val in (1,2),res.info.status
obj=lambda z:.5*z@problem['H']@z+problem['g']@z
gap=abs(obj(wr)-obj(res.x));assert gap<1e-5,(gap,solver.info)
result=dict(body_jacobian_fd_error=jerr,geometric_linearization_error=gerr,
            hierarchy_prediction_error=ctrl.info['hierarchy'],primary_cost_change=cost1-cost0,
            secondary_correction_norm=float(np.linalg.norm(w-w0)),actual_gmpc_objective_gap=gap,
            reluqp_residuals=solver.info)
out=root/'results'/'relu_hgmpc';out.mkdir(parents=True,exist_ok=True)
(out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
