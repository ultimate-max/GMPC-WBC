"""Auditable ReLU-QP eq. (12) implementation; no trained network or CPU fallback.

We implement the published recurrence rather than patch the upstream API, which
does not support changing H/A and has a separate internal warm-start buffer.
All reported solve times synchronize CUDA and include residual evaluation/copy.
"""
import os, sys, time
import numpy as np
import scipy.linalg as la
sys.path.append(os.environ.get('RELU_TORCH_SITE','/home/ganhe/miniconda3/envs/frictionest/lib/python3.11/site-packages'))
import torch
torch.set_num_threads(1)
if not torch.cuda.is_available():
    raise RuntimeError('CUDA required: this comparison must not silently use CPU')
DTYPE=torch.float64

class ReLUQP:
    def __init__(self,H,A,rho=.1):
        torch.cuda.synchronize(); start=time.perf_counter()
        self.H=(H+H.T)/2; self.A=A; self.n=len(H); self.m=len(A)
        # Explicit variable/row scaling; residuals are always also computed in
        # original units. Sigma is algorithmic proximal regularization, not cost.
        self.D=1/np.sqrt(np.maximum(np.diag(H),1e-5))
        hs=self.D[:,None]*self.H*self.D[None,:]
        a=A*self.D[None,:]
        self.E=1/np.maximum(np.linalg.norm(a,axis=1),1e-6)
        a=self.E[:,None]*a
        self.ascaled=a;self.rho=rho; sigma=1e-7
        K=la.cho_solve(la.cho_factor(hs+sigma*np.eye(self.n)+rho*a.T@a),np.eye(self.n))
        k=K@a.T; t=K@(sigma*np.eye(self.n)-rho*a.T@a)
        W=np.block([[t,2*rho*k,-k],
                    [a@t+a,2*rho*a@k-np.eye(self.m),-a@k+np.eye(self.m)/rho],
                    [rho*a,-rho*np.eye(self.m),np.eye(self.m)]])
        B=np.vstack((-K,-a@K,np.zeros((self.m,self.n))))
        self.W=torch.as_tensor(W,device='cuda',dtype=DTYPE)
        self.B=torch.as_tensor(B,device='cuda',dtype=DTYPE)
        self.state=None
        torch.cuda.synchronize();self.setup_ms=1000*(time.perf_counter()-start)
    def solve(self,g,l,u,iterations=100,xwarm=None):
        torch.cuda.synchronize(); start=time.perf_counter()
        lt=torch.as_tensor(self.E*l,device='cuda',dtype=DTYPE)
        ut=torch.as_tensor(self.E*u,device='cuda',dtype=DTYPE)
        b=self.B@torch.as_tensor(self.D*g,device='cuda',dtype=DTYPE)
        if xwarm is not None:
            xs=xwarm/self.D
            zs=np.clip(self.ascaled@xs,self.E*l,self.E*u)
            self.state=torch.as_tensor(np.r_[xs,zs,np.zeros(self.m)],device='cuda',dtype=DTYPE)
        elif self.state is None:
            self.state=torch.zeros(self.n+2*self.m,device='cuda',dtype=DTYPE)
        # Fixed budget, dual-first recurrence, single GEMV + saturation per layer.
        for _ in range(iterations):
            self.state=self.W@self.state+b
            self.state[self.n:self.n+self.m].clamp_(lt,ut)
        raw=self.state.cpu().numpy()
        x=self.D*raw[:self.n]; lam=self.E*raw[self.n+self.m:]
        ax=self.A@x
        primal=float(max(0,np.max(l-ax),np.max(ax-u)))
        stationarity=float(np.max(np.abs(self.H@x+g+self.A.T@lam)))
        gs=self.D*g; xs=raw[:self.n]; zs=raw[self.n:self.n+self.m]
        rp=float(np.max(np.abs(self.ascaled@xs-zs)))
        rd=float(np.max(np.abs(self.D*(self.H@x+g+self.A.T@lam))))
        self.info=dict(qp_ms=1000*(time.perf_counter()-start),primal=primal,
                       dual=stationarity,scaled_primal=rp,scaled_dual=rd,iterations=iterations)
        return x
