"""ReLU-QP central backend with exact equality reduction and whitening.

The physical QP is unchanged: x=x0+T*y, Aeq*x0=beq, Aeq*T=0.
T whitens the reduced positive-definite Hessian. This is solver preprocessing,
not a relaxed constraint, learned policy, or fallback to a different QP solver.
Every call rebuilds numerical matrices; all this work is timed by the caller.
"""
import time
from types import SimpleNamespace
import numpy as np
import scipy.linalg as la
from relu_solver import ReLUQP, torch


def residuals(H,g,A,l,u,x,dual):
    rows=1/np.maximum(1.,np.linalg.norm(A,axis=1))
    ax=A@x; projected=np.clip(ax,l,u)
    hx=H@x; aty=A.T@dual
    dual_scale=max(1.,np.max(np.abs(hx)),np.max(np.abs(g)),np.max(np.abs(aty)))
    row_ax=rows*ax; row_projected=rows*projected
    primal_scale=max(1.,np.max(np.abs(row_ax)),np.max(np.abs(row_projected)))
    # Normal-cone residual independently checks multiplier signs/complementarity.
    normal=np.max(np.abs(row_ax-np.clip(row_ax+dual/rows,rows*l,rows*u)))
    return dict(qp_primal_raw=float(np.max(np.abs(ax-projected))),
                qp_primal_relative=float(np.max(np.abs(rows*(ax-projected)))/primal_scale),
                qp_dual_raw=float(np.max(np.abs(hx+g+aty))),
                qp_dual_relative=float(np.max(np.abs(hx+g+aty))/dual_scale),
                qp_normal_relative=float(normal/primal_scale),
                qp_objective=float(.5*x@hx+g@x))


class CentralReLU:
    def __init__(self,H,g,A,l,u,warm=None,tol=2e-7,max_iter=3000,rho=.1):
        start=time.perf_counter()
        self.H=(H+H.T)/2;self.g=g;self.A=A;self.l=l;self.u=u
        self.tol=tol;self.max_iter=max_iter
        eq=np.isfinite(l)&np.isfinite(u)&(l==u);self.eq=eq
        E=A[eq];b=l[eq];self.erows=1/np.maximum(1.,np.linalg.norm(E,axis=1))
        Es=self.erows[:,None]*E;bs=self.erows*b
        Q,R,piv=la.qr(Es.T,mode='full',pivoting=True)
        diag=np.abs(np.diag(R));rank=int(np.sum(diag>1e-11*max(1.,diag.max())))
        self.rank=rank;self.Q=Q;self.R=R;self.piv=piv
        coeff=la.solve_triangular(R[:rank,:rank].T,bs[piv[:rank]],lower=True)
        xe=Q[:,:rank]@coeff;Z=Q[:,rank:]
        if np.max(np.abs(E@xe-b))>1e-7:raise RuntimeError('Inconsistent equality elimination')
        Hr=Z.T@self.H@Z;Hr=(Hr+Hr.T)/2
        L=la.cholesky(Hr,lower=True)
        T=la.solve_triangular(L,Z.T,lower=True).T
        gr=T.T@(self.H@xe+g)
        self.x0=xe-T@gr;self.T=T
        if np.max(np.abs(E@self.x0-b))>1e-6:raise RuntimeError('Equality reduction lost precision')
        # The transformed cost is 1/2*y.T*y + constant. No physical weight changes.
        C=A[~eq]@T;self.cl=l[~eq]-A[~eq]@self.x0;self.cu=u[~eq]-A[~eq]@self.x0
        self.kernel=ReLUQP(np.eye(T.shape[1]),C,rho=rho)
        self.ywarm=None if warm is None else L.T@(Z.T@(warm-self.x0))
        self.preprocessing_ms=1000*(time.perf_counter()-start)-self.kernel.setup_ms
        self.kernel_setup_ms=self.kernel.setup_ms
        self.reduced_n=T.shape[1]

    def solve(self):
        total=0;info=None
        while total<self.max_iter:
            count=min(25,self.max_iter-total)
            y=self.kernel.solve(np.zeros(self.reduced_n),self.cl,self.cu,count,
                                xwarm=self.ywarm if total==0 else None)
            total+=count
            raw=self.kernel.state.cpu().numpy()
            di=self.kernel.E*raw[self.kernel.n+self.kernel.m:]
            x=self.x0+self.T@y
            dual=np.zeros(len(self.l));dual[~self.eq]=di
            rhs=-(self.H@x+self.g+self.A[~self.eq].T@di)
            qe=self.Q[:,:self.rank].T@rhs
            de=np.zeros(np.sum(self.eq))
            de[self.piv[:self.rank]]=la.solve_triangular(self.R[:self.rank,:self.rank],qe)
            dual[self.eq]=self.erows*de
            metrics=residuals(self.H,self.g,self.A,self.l,self.u,x,dual)
            # Common original-QP metrics, not a solver-specific scaled residual.
            converged=max(metrics['qp_primal_relative'],metrics['qp_dual_relative'],
                          metrics['qp_normal_relative'])<=self.tol
            if converged:break
        self.x=x;self.dual=dual;self.metrics=metrics
        return x,SimpleNamespace(iter=total,status='solved' if converged else 'iteration limit',
                   status_val=1 if converged else 7,converged=converged,
                   preprocessing_ms=self.preprocessing_ms,kernel_setup_ms=self.kernel_setup_ms,
                   reduced_n=self.reduced_n,**metrics)
