import numpy as np,scipy.linalg as la,scipy.sparse as sp,osqp,time
from central_relu import residuals
z=np.load('/tmp/shared_hierarchy_failed.npz');C,d,G,lo,hi=[z[k] for k in ['C','d','G','lo','hi']]
H=2*C.T@C;g=2*C.T@d;sc=max(1.,np.max(abs(H)));H/=sc;g/=sc
rows=1/np.maximum(la.norm(G,axis=1),1e-10)
for eps in [1e-8,1e-9,1e-10]:
    start=time.perf_counter();qp=osqp.OSQP();qp.setup(P=sp.csc_matrix(np.triu(H)),q=g,A=sp.csc_matrix(rows[:,None]*G),l=lo*rows,u=hi*rows,
        eps_abs=eps,eps_rel=eps,max_iter=30000,polishing=True,verbose=False,check_termination=20)
    s=qp.solve();r=residuals(H,g,G,lo,hi,s.x,s.y*rows)
    print(eps,s.info.status,s.info.iter,1000*(time.perf_counter()-start),r,flush=True)
