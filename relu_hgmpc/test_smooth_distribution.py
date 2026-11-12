import json,pathlib
import numpy as np
import scipy.sparse as sp
import osqp
import pinocchio as pin
import native
from model import Reference
from prioritized import PrioritizedController
from smooth_distribution import SmoothDistributor

root=pathlib.Path(__file__).resolve().parents[1]
p=native.Plant(str(root));s=p.truth();ref=Reference(p,s['q'],s['pose'])
c=PrioritizedController(p,ref,6);_,_,J=c.chart(s)
cases=[]
rng=np.random.default_rng(20261110)
for index in range(8):
    # Feasible physical B2+Z1 task; narrow box puts genuine bounds in play.
    previous=rng.normal(size=12)*.02;amax=np.r_[np.full(3,2.),np.full(3,3.),np.full(6,8.)]
    low=np.maximum(-.1,previous-.02*amax);high=np.minimum(.1,previous+.02*amax)
    witness=low+(high-low)*rng.uniform(.05,.95,12);task=J@witness
    secondary=rng.normal(size=12)*.2
    central=SmoothDistributor('central');v,info=central.solve(J,task,low,high,secondary,previous,.02,amax)
    d=central.last_problem
    oracle=osqp.OSQP();oracle.setup(P=sp.csc_matrix(d['H']),q=-d['a'],
        A=sp.csc_matrix(np.vstack((d['A'],np.eye(12)))),
        l=np.r_[d['b'],d['low']],u=np.r_[d['b'],d['high']],
        eps_abs=1e-9,eps_rel=1e-9,max_iter=100000,polishing=True,verbose=False)
    answer=oracle.solve();assert answer.info.status_val==1,answer.info.status
    oracle_error=float(np.max(np.abs(answer.x-d['y'])))
    admm=SmoothDistributor('admm',iterations=6000)
    va,ia=admm.solve(J,task,low,high,secondary,previous,.02,amax)
    error=float(np.max(np.abs(va-v)))
    assert oracle_error<1e-6 and error<3e-6,(index,oracle_error,error,ia)
    assert info['slew_violation']<1e-8 and ia['slew_violation']<1e-8
    cases.append(dict(case=index,oracle_error=oracle_error,admm_error=error,**ia))
# Bounds conflicting with slew must fail explicitly, never silently clip afterward.
try:
    central.solve(J,np.zeros(6),np.ones(12),np.ones(12)*2,np.zeros(12),np.zeros(12),.02,np.ones(12))
except RuntimeError:pass
else:raise AssertionError('Infeasible slew intersection was accepted')
# Independent finite-difference check of the optional moving SE(3) target law.
E=pin.exp6(np.array([.1,-.05,.03,.2,-.15,.1]));vd=np.array([.1,.02,-.04,.3,-.1,.2])
eta=pin.log6(E).vector;V=E.action@vd+10*eta;eps=1e-7
next_error=pin.exp6(-eps*V)*E*pin.exp6(eps*vd)
error_law=float(np.max(np.abs((pin.log6(next_error).vector-eta)/eps+10*eta)))
assert error_law<1e-5,error_law
out=root/'results/chatter_mitigation';out.mkdir(parents=True,exist_ok=True)
(out/'solver_verification.json').write_text(json.dumps(cases,indent=2))
print(json.dumps(dict(cases=len(cases),max_admm_error=max(x['admm_error'] for x in cases),
    max_oracle_error=max(x['oracle_error'] for x in cases),se3_error_law_difference=error_law)))
