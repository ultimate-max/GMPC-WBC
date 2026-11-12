import numpy as np,scipy.linalg as la,quadprog,json
from central_relu import residuals
a=np.load('/tmp/anchor_lower_qp.npz');H,g,G,lo,hi=[a[k] for k in ('H','g','G','lo','hi')]
print('dimension',len(g),'inherited feasibility',a['inherited'],'eigen',la.eigvalsh(H)[[0,-1]],'g',max(abs(g)),flush=True)
il=np.flatnonzero(np.isfinite(lo));iu=np.flatnonzero(np.isfinite(hi))
C=np.vstack((G[il],-G[iu])).T;b=np.r_[lo[il],-hi[iu]];rs=1/np.maximum(la.norm(C,axis=0),1e-12)
for eps in (1e-12,1e-11,1e-10,1e-9):
    try:
        sol=quadprog.solve_qp(H,-g,C*rs,(b-eps)*rs);x=sol[0];mult=sol[4]*rs;y=np.zeros(len(lo))
        np.add.at(y,il,-mult[:len(il)]);np.add.at(y,iu,mult[len(il):])
        print(eps,residuals(H,g,G,lo,hi,x,y),flush=True)
    except Exception as exc:print(eps,str(exc),flush=True)
