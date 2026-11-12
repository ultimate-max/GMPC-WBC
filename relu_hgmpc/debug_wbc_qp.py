import numpy as np
from scipy.optimize import linprog
z=np.load('/tmp/geometric_wbc_failed_qp.npz')
G,lo,hi=z['G'],z['lo'],z['hi']
print('level',z['level'],'shape',G.shape,'zero-feasible',max(0,np.max(lo),np.max(-hi)))
norm=np.linalg.norm(G,axis=1);bad=np.flatnonzero((lo>0)|(hi<0))
for i in bad:print('bound',i,'rownorm',norm[i],'lo,hi',lo[i],hi[i])
rows=norm>1e-10
for select in [np.ones(len(lo),bool),rows]:
    l=np.flatnonzero(select&np.isfinite(lo));u=np.flatnonzero(select&np.isfinite(hi))
    lp=linprog(np.zeros(G.shape[1]),A_ub=np.vstack((-G[l],G[u])),b_ub=np.r_[-lo[l],hi[u]],bounds=[(None,None)]*G.shape[1],method='highs')
    print('LP',lp.message)
