import pathlib,time,importlib.util,numpy as np,scipy.sparse as sp,osqp
from scipy.optimize import minimize,LinearConstraint
root=pathlib.Path(__file__).resolve().parents[1];p=np.load(root/'results/prioritized/failure_6d.npz')
A=p['A'];l=p['l'];u=p['u'];H=np.diag(np.r_[np.full(6,2e4),np.full(12,2e-4)])
D=1/np.sqrt(np.diag(H));G=A*D
E=1/np.maximum(np.linalg.norm(G,axis=1),1e-10);G=E[:,None]*G;lo=E*l;hi=E*u
for rho in (.1,10,1000):
    qp=osqp.OSQP();qp.setup(P=sp.eye(18,format='csc'),q=np.zeros(18),A=sp.csc_matrix(G),l=lo,u=hi,verbose=False,eps_abs=1e-7,eps_rel=1e-7,max_iter=30000,polishing=True,rho=rho)
    st=time.perf_counter();r=qp.solve();print('rho',rho,'status',r.info.status,'iter',r.info.iter,'resid',r.info.prim_res,r.info.dual_res,'ms',1000*(time.perf_counter()-st))
st=time.perf_counter();r=minimize(lambda x:.5*x@x,np.zeros(18),jac=lambda x:x,method='SLSQP',constraints=[LinearConstraint(G[:6],lo[:6],hi[:6]),LinearConstraint(G[6:],lo[6:],hi[6:])],options={'maxiter':1000,'ftol':1e-8})
print('slsqp',r.success,r.message,'ms',1000*(time.perf_counter()-st),'primal',max(np.max(lo-G@r.x),np.max(G@r.x-hi)))
print({x:str(importlib.util.find_spec(x)) for x in ('quadprog','clarabel','proxsuite')})
