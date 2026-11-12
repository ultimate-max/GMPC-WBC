import numpy as np,scipy.linalg as la
from scipy.optimize import nnls
from central_relu import residuals
a=np.load('/tmp/anchor_qp_recovery_failure.npz');H,g,G,lo,hi,x=[a[k] for k in ('H','g','G','lo','hi','x')]
il=np.flatnonzero(np.isfinite(lo));iu=np.flatnonzero(np.isfinite(hi))
C=np.vstack((G[il],-G[iu])).T;b=np.r_[lo[il],-hi[iu]]
rs=1/np.maximum(la.norm(C,axis=0),1e-12)
for eps in (1e-10,1e-9,1e-8,1e-7):
    active=np.flatnonzero(C.T@x-b<eps);mult=np.zeros(len(b))
    try:
        mult[active]=nnls(C[:,active]*rs[active],H@x+g,maxiter=10000)[0]*rs[active]
        y=np.zeros(len(lo));np.add.at(y,il,-mult[:len(il)]);np.add.at(y,iu,mult[len(il):])
        print(eps,len(active),residuals(H,g,G,lo,hi,x,y),flush=True)
    except Exception as exc:print(str(exc),flush=True)
