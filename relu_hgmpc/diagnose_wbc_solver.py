import numpy as np,time
from osqp_acceleration import PersistentOSQP
from central_relu import residuals
z=np.load('/tmp/geometric_wbc_failed_qp.npz');H=z['H'];g=z['g'];G=z['G'];lo=z['lo'];hi=z['hi']
scale=max(1,np.max(abs(H)));zero=np.linalg.norm(G,axis=1)<1e-10
G[zero]=0.;lo[zero]=-np.inf;hi[zero]=np.inf
import quadprog
il=np.flatnonzero(np.isfinite(lo));iu=np.flatnonzero(np.isfinite(hi))
C=np.vstack((G[il],-G[iu])).T;b=np.r_[lo[il],-hi[iu]]
rows=1/np.maximum(np.linalg.norm(C,axis=0),1e-12)
try:
    start=time.perf_counter()
    sol=quadprog.solve_qp(H/scale+1e-10*np.eye(len(g)),-g/scale,C*rows,b*rows)
    y=np.zeros(len(lo));np.add.at(y,il,-sol[4][:len(il)]*rows[:len(il)]);np.add.at(y,iu,sol[4][len(il):]*rows[len(il):])
    print('quadprog',1000*(time.perf_counter()-start),residuals(H/scale,g/scale,G,lo,hi,sol[0],y),flush=True)
except Exception as exc:print('quadprog failed',exc)
for tol in []:
    for rho in [.1,10.]:
        e=PersistentOSQP(tol);e.prepare(H/scale,g/scale,G,lo,hi,z['warm'])
        e.solver.update_settings(max_iter=100000,rho=rho,polishing=True,adaptive_rho_interval=50)
        start=time.perf_counter();r=e.solver.solve();y=e.rows*r.y
        print(tol,rho,r.info.status,r.info.iter,1000*(time.perf_counter()-start),residuals(H/scale,g/scale,G,lo,hi,r.x,y),flush=True)
