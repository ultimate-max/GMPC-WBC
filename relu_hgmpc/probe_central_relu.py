"""Compare the existing ReLU-QP kernel on an unchanged central GMPC QP."""
import pathlib, json, time
import numpy as np
import scipy.linalg as la
import native
from model import Reference
from split_gmpc import SplitGMPC, QPSolver
from relu_solver import ReLUQP
from central_relu import CentralReLU,residuals

root=pathlib.Path(__file__).resolve().parents[1]
p=native.Plant(str(root));s=p.truth();ref=Reference(p,s['q'],s['pose'])
c=SplitGMPC(p,ref,root,mode='central');parts=c.build(s,1.)
a,b=parts
H=la.block_diag(a['H'],b['H']);g=np.r_[a['g'],b['g']]
A=np.vstack((la.block_diag(a['A'],b['A']),np.column_stack((a['E'],-b['E']))))
l=np.r_[a['l'],b['l'],b['d']-a['d']];u=np.r_[a['u'],b['u'],b['d']-a['d']]
warm=np.concatenate(c.warm(parts))
osqp=QPSolver(H,g,A,l,u,warm);x,info=osqp.solve()
print(json.dumps(dict(n=len(g),m=len(l),equalities=int(np.sum(l==u)),osqp_iterations=info.iter)),flush=True)
for rho in [.001,.01,.1,1.]:
    start=time.perf_counter();solver=CentralReLU(H,g,A,l,u,warm,rho=rho)
    setup_ms=1000*(time.perf_counter()-start);start=time.perf_counter();z,si=solver.solve()
    print(json.dumps(dict(rho=rho,setup_ms=setup_ms,solve_ms=1000*(time.perf_counter()-start),
      objective_gap=float(.5*z@H@z+g@z-(.5*x@H@x+g@x)),
      normalized_control_max=float(np.max(np.abs(z-x))),**vars(si))),flush=True)
