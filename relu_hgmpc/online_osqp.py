"""Exact whitened QP, persistent OSQP, and certified constraint generation.

No constraints are accepted as inactive without checking every original row.
Changing Hessians are refactorized; only symbolic structures are reused.
"""
import time
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import osqp
from central_relu import residuals

class OnlineOSQP:
    def __init__(self,screen=True,tol=2e-7):
        self.screen=screen;self.tol=tol;self.solver=None;self.capacity=0
        self.last=None;self.last_time=None;self.previous_active=np.array([],dtype=int)
        self.last_dual=None
        self.total_setups=0;self.total_updates=0

    def solve(self,H,g,G,lo,hi,t,dt,N):
        start=time.perf_counter();L=la.cholesky(H,lower=True)
        center=la.cho_solve((L,True),-g)
        # z = center + L^{-T} y, objective = .5 y^T y + constant.
        offset=G@center;lower=lo-offset;upper=hi-offset
        if self.screen:
            # Predict the preceding active face, solve its equality QP exactly,
            # then certify ALL original constraints and KKT. Rejecting a face
            # always falls through to OSQP; no approximate plan is published.
            face=self.previous_active
            if len(face):
                Cface=la.solve_triangular(L,G[face].T,lower=True).T
                rscale=1/np.maximum(la.norm(Cface,axis=1),1e-12);Cface*=rscale[:,None]
                rhs=np.where(self.last_dual[face]<0,lower[face],upper[face])*rscale
                U,sval,Vt=la.svd(Cface,full_matrices=False)
                if len(sval) and sval[-1]>1e-11*sval[0]:
                    projected=U.T@rhs;yy=Vt.T@(projected/sval)
                    candidate=center+la.solve_triangular(L.T,yy,lower=False)
                    candidate_dual=np.zeros(len(lo));candidate_dual[face]=-(U@(projected/sval**2))*rscale
                else:candidate=None
            else:candidate=center;candidate_dual=np.zeros(len(lo))
            if candidate is not None:
                rr=residuals(H,g,G,lo,hi,candidate,candidate_dual)
                error=max(rr[k] for k in ['qp_primal_relative','qp_dual_relative','qp_normal_relative'])
                violation=float(max(0,np.max(lo-G@candidate),np.max(G@candidate-hi)))
                if error<=min(self.tol,1e-10) and violation<=1e-9:
                    self.last=candidate.copy();self.last_time=t;self.last_dual=candidate_dual.copy()
                    self.previous_active=np.flatnonzero(abs(candidate_dual)>1e-9)
                    return candidate,candidate_dual,0,dict(osqp_screened=1,osqp_constraints=len(face),osqp_capacity=self.capacity,
                        osqp_rounds=0,osqp_setup_ms=0.,osqp_iteration_ms=0.,osqp_total_ms=1000*(time.perf_counter()-start),
                        osqp_setups=self.total_setups,osqp_updates=self.total_updates,osqp_pattern_growth=0,
                        original_qp_residual=error,original_bound_violation=violation,active_face_fastpath=1)
        if self.last is None:warm=center.copy()
        else:
            shift=max(0.,(t-self.last_time)/dt);idx=np.minimum(np.arange(N)+shift,N-1)
            i=np.floor(idx).astype(int);j=np.minimum(i+1,N-1);w=(idx-i)[:,None]
            old=self.last.reshape(N,18);warm=((1-w)*old[i]+w*old[j]).ravel()
        ywarm=L.T@(warm-center)
        # Seed with all unconstrained violations and the preceding active set.
        violation=np.maximum(lower,np.maximum(-upper,0.))
        initial=np.flatnonzero(violation>1e-10)
        active=np.unique(np.r_[self.previous_active,initial]) if self.screen else np.arange(len(lo))
        setup_ms=0.;solve_ms=0.;iterations=0;rounds=0;growth=0
        while True:
            rounds+=1
            if rounds>len(lo)+1:raise RuntimeError('Constraint generation did not terminate')
            needed=len(active);cap=max(32,1<<max(0,needed-1).bit_length())
            cap=min(cap,len(lo))
            ts=time.perf_counter()
            # Transform only selected rows. Full feasibility is still checked
            # in physical coordinates, so the omitted dense solve is exact.
            C=la.solve_triangular(L,G[active].T,lower=True).T
            scale=1/np.maximum(la.norm(C,axis=1),1e-12);C*=scale[:,None]
            if self.solver is None or cap>self.capacity:
                self.capacity=cap;growth+=1
                # Dense fixed slots preserve exact zeros and all future values.
                A=sp.csc_matrix(np.ones((cap,len(g))))
                self.solver=osqp.OSQP(algebra='builtin')
                self.solver.setup(P=sp.eye(len(g),format='csc'),q=np.zeros(len(g)),A=A,
                    l=np.full(cap,-np.inf),u=np.full(cap,np.inf),eps_abs=1e-9,eps_rel=1e-9,
                    max_iter=4000,check_termination=10,polishing=True,warm_starting=True,verbose=False,
                    adaptive_rho=True)
                self.total_setups+=1
            values=np.zeros((self.capacity,len(g)),order='F');values[:needed]=C
            l=np.full(self.capacity,-np.inf);u=np.full(self.capacity,np.inf)
            l[:needed]=lower[active]*scale;u[:needed]=upper[active]*scale
            self.solver.update(Ax=values.ravel(order='F'),l=l,u=u)
            self.solver.update_settings(polishing=bool(needed))
            self.solver.warm_start(x=ywarm,y=np.zeros(self.capacity));self.total_updates+=1
            setup_ms+=1000*(time.perf_counter()-ts);ts=time.perf_counter()
            sol=self.solver.solve(raise_error=False);solve_ms+=1000*(time.perf_counter()-ts)
            iterations+=sol.info.iter
            if sol.info.status_val not in (1,2):
                raise RuntimeError(f'OSQP {sol.info.status}; rows={needed}, rounds={rounds}')
            ywarm=sol.x;z=center+la.solve_triangular(L.T,sol.x,lower=False)
            dual=np.zeros(len(lo));dual[active]=sol.y[:needed]*scale
            # Excluded rows have zero dual; check their feasibility in ORIGINAL
            # physical coordinates. This certifies the screened QP solution.
            gz=G@z;v=np.maximum(lo-gz,np.maximum(gz-hi,0.))
            excluded=np.ones(len(lo),bool);excluded[active]=False
            missing=np.flatnonzero(excluded & (v>1e-9))
            if len(missing):active=np.union1d(active,missing);continue
            rr=residuals(H,g,G,lo,hi,z,dual)
            error=max(rr[k] for k in ['qp_primal_relative','qp_dual_relative','qp_normal_relative'])
            if error>self.tol:raise RuntimeError(f'Original OSQP KKT gate {rr}')
            break
        self.last=z.copy();self.last_time=t
        self.last_dual=dual.copy()
        self.previous_active=active[np.abs(dual[active])>1e-9]
        info=dict(osqp_screened=int(self.screen),osqp_constraints=needed,osqp_capacity=self.capacity,
            osqp_rounds=rounds,osqp_setup_ms=setup_ms,osqp_iteration_ms=solve_ms,
            osqp_total_ms=1000*(time.perf_counter()-start),osqp_setups=self.total_setups,
            osqp_updates=self.total_updates,osqp_pattern_growth=growth,
            original_qp_residual=error,original_bound_violation=float(max(0,np.max(v))),active_face_fastpath=0)
        return z,dual,iterations,info
