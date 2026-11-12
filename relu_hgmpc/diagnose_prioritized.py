import pathlib,numpy as np
from scipy.optimize import linprog
root=pathlib.Path(__file__).resolve().parents[1]
p=np.load(root/'results/prioritized/failure_6d.npz');A=p['A'];l=p['l'];u=p['u']
ineq=np.r_[np.flatnonzero(np.isfinite(u[6:]))+6,np.flatnonzero(np.isfinite(l[6:]))+6]
Au=np.vstack((A[6:][np.isfinite(u[6:])],-A[6:][np.isfinite(l[6:])]))
bu=np.r_[u[6:][np.isfinite(u[6:])],-l[6:][np.isfinite(l[6:])]]
r=linprog(np.zeros(18),A_ub=Au,b_ub=bu,A_eq=A[:6],b_eq=l[:6],bounds=[(None,None)]*18,method='highs')
print('time',p['time'],'sigma',p['sigma'],'acc_cmd',p['desired_acc'],'feasible',r.success,r.message)
