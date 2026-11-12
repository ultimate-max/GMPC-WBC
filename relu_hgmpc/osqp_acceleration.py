"""Exact contact/interface elimination and persistent OSQP for PaperGMPC.

No physical weight, constraint, model, horizon, or acceptance tolerance changes.
The full original QP is retained for independent primal/dual/normal-cone checks.
"""
import time
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import osqp
from paper_gmpc import PaperGMPC
from central_relu import residuals

def assemble(parts):
    a,b=parts
    H=la.block_diag(a['H'],b['H']);g=np.r_[a['g'],b['g']]
    A=np.vstack((la.block_diag(a['A'],b['A']),np.c_[a['E'],-b['E']]))
    l=np.r_[a['l'],b['l'],b['d']-a['d']]
    u=np.r_[a['u'],b['u'],b['d']-a['d']]
    return H,g,A,l,u

class ContactReduction:
    """x=T z+c, where z contains all 18 scaled physical torques per node.

    The 24 dependent variables per node are 12 contact forces and two 6D
    interface wrenches. The 24 equations are contact acceleration, base-twist
    consensus, and opposing interface wrench. Their dependent-column matrix
    is block lower triangular. This uses the ACTUAL QP equations, including
    its frozen-C velocity coupling, not a different full-body approximation.
    """
    def __init__(self,parts,H,g,A,l,u):
        self.H,self.g,self.A,self.l,self.u=H,g,A,l,u
        a,b=parts;N=len(a['controls']);na=len(a['g']);nb=len(b['g'])
        if a['nu']!=30 or b['nu']!=12 or a['A'].shape[0]!=86*N or b['A'].shape[0]!=30*N:
            raise ValueError('Contact reduction requires the verified four-foot B2+Z1 layout')
        offset=len(a['l'])+len(b['l'])
        self.free=np.concatenate([np.r_[30*k+np.arange(12),na+12*k+np.arange(6)] for k in range(N)])
        self.dep=np.concatenate([np.r_[30*k+np.arange(12,30),na+12*k+np.arange(6,12)] for k in range(N)])
        self.eq=np.concatenate([np.r_[86*k+np.arange(12),offset+12*k+np.arange(12)] for k in range(N)])
        expected=np.flatnonzero(np.isfinite(l)&np.isfinite(u)&(l==u))
        if not np.array_equal(np.sort(self.eq),expected):raise ValueError('Unexpected equalities')
        self.ineq=np.setdiff1d(np.arange(len(l)),self.eq,assume_unique=True)
        E=A[self.eq];be=l[self.eq]
        self.erows=1/np.maximum(1.,la.norm(E,axis=1))
        Es=self.erows[:,None]*E;bs=self.erows*be
        self.D=Es[:,self.dep];G=Es[:,self.free]
        nr=len(self.free);nd=len(self.dep);self.N=N;self.factors=[]
        dependent=np.zeros((nd,nr));d=np.zeros(nd)
        for k in range(N):
            rr=slice(24*k,24*(k+1));previous=slice(0,24*k)
            if np.any(self.D[rr,24*(k+1):]!=0):raise ValueError('Noncausal equality layout')
            factor=la.lu_factor(self.D[rr,rr]);self.factors.append(factor)
            rhs=np.c_[-G[rr]-self.D[rr,previous]@dependent[previous],
                      bs[rr]-self.D[rr,previous]@d[previous]]
            sol=la.lu_solve(factor,rhs);dependent[rr]=sol[:,:-1];d[rr]=sol[:,-1]
        self.T=np.zeros((na+nb,nr));self.c=np.zeros(na+nb)
        self.T[self.free]=np.eye(nr);self.T[self.dep]=dependent;self.c[self.dep]=d
        self.equality_map_max=float(np.max(np.abs(E@self.T)))
        self.equality_offset_max=float(np.max(np.abs(E@self.c-be)))
        if max(self.equality_map_max,self.equality_offset_max)>1e-7:
            raise RuntimeError('Equality parameterization lost accuracy')
        HT=H@self.T;Hr=self.T.T@HT
        self.Hr=(Hr+Hr.T)/2;self.gr=self.T.T@(H@self.c+g)
        Ai=A[self.ineq];offset_i=Ai@self.c
        self.Ar=Ai@self.T;self.lr=l[self.ineq]-offset_i;self.ur=u[self.ineq]-offset_i

    def recover(self,z,yi):
        x=self.T@z+self.c
        rhs=-(self.H@x+self.g+self.A[self.ineq].T@yi)[self.dep]
        ye=np.zeros(len(self.dep))
        for k in reversed(range(self.N)):
            rr=slice(24*k,24*(k+1));after=slice(24*(k+1),len(self.dep))
            ye[rr]=la.lu_solve(self.factors[k],rhs[rr]-self.D[after,rr].T@ye[after],trans=1)
        dual=np.zeros(len(self.l));dual[self.ineq]=yi;dual[self.eq]=self.erows*ye
        return x,dual

class PersistentOSQP:
    """Union of observed exact sparsity, with safe rebuild on pattern growth.

    Zero->nonzero transitions are never discarded or thresholded. Matrix
    values are updated every cycle; symbolic structure alone is cached.
    Primal warm start is the same shifted physical sequence as the baseline.
    Dual warm start is zero in both, to isolate structural reuse.
    """
    def __init__(self,tol=2e-7,reuse=True,algebra='builtin'):
        self.tol=tol;self.reuse=reuse;self.solver=None
        self.pmask=None;self.amask=None;self.total_setups=0;self.total_updates=0
        if algebra not in ('builtin','cuda'):raise ValueError(algebra)
        if algebra not in osqp.algebras_available():raise RuntimeError(f'OSQP algebra unavailable: {algebra}')
        self.algebra=algebra;self.solve_wall_ms=0.
        self.cuda_runtime=None
        if algebra=='cuda':
            import ctypes
            import osqp_cuda
            if osqp_cuda.OSQP_USE_FLOAT:
                raise RuntimeError('This comparison requires the FP64 CUDA build; source env_cuda_osqp.sh')
            self.cuda_runtime=ctypes.CDLL('libcudart.so.12')
            self.cuda_runtime.cudaDeviceSynchronize.restype=ctypes.c_int

    def synchronize(self):
        if self.cuda_runtime is not None:
            code=self.cuda_runtime.cudaDeviceSynchronize()
            if code:raise RuntimeError(f'CUDA synchronization failed: {code}')

    @staticmethod
    def structure(mask):
        matrix=sp.csc_matrix(mask.astype(float))
        rr=matrix.indices.copy();cc=np.repeat(np.arange(mask.shape[1]),np.diff(matrix.indptr))
        return matrix,rr,cc

    def prepare(self,H,g,A,l,u,warm):
        start=time.perf_counter();self.synchronize();self.solve_wall_ms=0.;H=(H+H.T)/2
        self.rows=1/np.maximum(1.,la.norm(A,axis=1));As=A*self.rows[:,None]
        pm=np.triu(H!=0);am=A!=0
        incompatible=self.pmask is None or self.pmask.shape!=pm.shape or self.amask.shape!=am.shape
        grow=incompatible or np.any(pm&~self.pmask) or np.any(am&~self.amask)
        rebuild=(not self.reuse) or self.solver is None or grow
        self.rebuilt=int(rebuild);self.pattern_growth=int(self.solver is not None and grow)
        if rebuild:
            self.pmask=pm if incompatible or not self.reuse else self.pmask|pm
            self.amask=am if incompatible or not self.reuse else self.amask|am
            P,self.pr,self.pc=self.structure(self.pmask)
            C,self.ar,self.ac=self.structure(self.amask)
            P.data[:]=H[self.pr,self.pc];C.data[:]=As[self.ar,self.ac]
            self.solver=osqp.OSQP(algebra=self.algebra)
            self.solver.setup(P=P,q=g,A=C,l=self.rows*l,u=self.rows*u,
                eps_abs=self.tol,eps_rel=self.tol,max_iter=12000,
                polishing=False,warm_starting=True,verbose=False,check_termination=10,time_limit=30.)
            self.total_setups+=1
        else:
            # Both numerical matrices change; numerical refactorization is
            # still required. No stale numerical factor is treated as exact.
            self.solver.update(Px=H[self.pr,self.pc],Ax=As[self.ar,self.ac],
                               q=g,l=self.rows*l,u=self.rows*u)
            self.solver.update_settings(eps_abs=self.tol,eps_rel=self.tol)
            self.total_updates+=1
        self.solver.warm_start(x=warm,y=np.zeros(len(l)))
        self.synchronize()
        self.prepare_ms=1000*(time.perf_counter()-start)
        self.nnz_p=int(np.sum(self.pmask));self.nnz_a=int(np.sum(self.amask))

    def solve(self,level=0):
        start=time.perf_counter();self.synchronize()
        if level:self.solver.update_settings(eps_abs=self.tol*10.**(-level),eps_rel=self.tol*10.**(-level))
        result=self.solver.solve()
        self.synchronize();self.solve_wall_ms+=1000*(time.perf_counter()-start)
        self.last_result=result
        if result.info.status_val not in (1,2):
            raise RuntimeError(f'OSQP status {result.info.status}; refinement {level}')
        return result.x,self.rows*result.y,result.info

class AcceleratedGMPC(PaperGMPC):
    def __init__(self,*args,optimization='combined',algebra='builtin',**kwargs):
        if optimization not in ['baseline','reuse','eliminate','combined']:raise ValueError(optimization)
        super().__init__(*args,**kwargs)
        if self.mode!='central' or self.backend!='osqp':raise ValueError('Central OSQP only')
        self.optimization=optimization
        if optimization=='baseline' and algebra!='builtin':raise ValueError('Use combined/reuse/eliminate for CUDA')
        self.engine=PersistentOSQP(self.qp_tol,reuse=optimization in ['reuse','combined'],algebra=algebra)

    def solve(self,parts):
        if self.optimization=='baseline':
            result,info=super().solve(parts)
            info.update(reduction_ms=0.,assembly_ms=0.,solver_prepare_ms=0.,dual_recovery_ms=0.,
                common_check_ms=0.,osqp_rebuilt=1,pattern_growth=0,optimized_variables=sum(len(p['g']) for p in parts),
                optimized_constraints=sum(len(p['l']) for p in parts)+len(parts[0]['d']),
                elimination_map_max=0.,elimination_offset_max=0.,qp_refinements=0)
            return result,info
        start=time.perf_counter();guesses=self.warm(parts);H,g,A,l,u=assemble(parts)
        assembly_ms=1000*(time.perf_counter()-start);warm=np.r_[*guesses]
        reduction=None;ts=time.perf_counter()
        if self.optimization in ['eliminate','combined']:
            reduction=ContactReduction(parts,H,g,A,l,u)
            hr,gr,ar,lr,ur=reduction.Hr,reduction.gr,reduction.Ar,reduction.lr,reduction.ur
            zwarm=warm[reduction.free]
        else:hr,gr,ar,lr,ur,zwarm=H,g,A,l,u,warm
        reduction_ms=1000*(time.perf_counter()-ts)
        self.engine.prepare(hr,gr,ar,lr,ur,zwarm)
        setup_ms=1000*(time.perf_counter()-start);ts=time.perf_counter()
        count=0;recovery_ms=0.;checks_ms=0.
        for level in range(7):
            z,di,info=self.engine.solve(level);count+=info.iter
            t0=time.perf_counter()
            if reduction is None:x,dual=z,di
            else:x,dual=reduction.recover(z,di)
            recovery_ms+=1000*(time.perf_counter()-t0);t0=time.perf_counter()
            metrics=residuals(H,g,A,l,u,x,dual)
            checks_ms+=1000*(time.perf_counter()-t0)
            if max(metrics[k] for k in ['qp_primal_relative','qp_dual_relative','qp_normal_relative'])<=self.qp_tol:break
        else:raise RuntimeError(f'Original QP tolerance not reached: {metrics}')
        solve_ms=1000*(time.perf_counter()-ts)
        n0=len(parts[0]['g']);result=[x[:n0],x[n0:]]
        ys=[p['E']@w+p['d'] for p,w in zip(parts,result)]
        primal=float(np.max(np.abs(ys[0]-ys[1])))
        violation=max(float(max(0.,np.max(p['l']-p['A']@w),np.max(p['A']@w-p['u']))) for p,w in zip(parts,result))
        self.last=[r.copy() for r in result]
        self.previous_rotation=self.current_rotation.copy() if hasattr(self,'current_rotation') else None
        self.last_full_dual=dual;self.last_reduction=reduction
        metrics.update(setup_ms=setup_ms,solve_ms=solve_ms,iterations=1,interface_scaled=primal,
            dual_residual=0.,local_primal=violation,local_iterations=count,admm_converged=int(primal<1e-5),
            qp_iterations=count,qp_converged=1,preprocessing_ms=reduction_ms,kernel_setup_ms=self.engine.prepare_ms,
            reduction_ms=reduction_ms,assembly_ms=assembly_ms,solver_prepare_ms=self.engine.prepare_ms,
            dual_recovery_ms=recovery_ms,common_check_ms=checks_ms,osqp_rebuilt=self.engine.rebuilt,
            pattern_growth=self.engine.pattern_growth,optimized_variables=len(gr),optimized_constraints=len(lr),
            elimination_map_max=0. if reduction is None else reduction.equality_map_max,
            elimination_offset_max=0. if reduction is None else reduction.equality_offset_max,qp_refinements=level)
        return result,metrics
