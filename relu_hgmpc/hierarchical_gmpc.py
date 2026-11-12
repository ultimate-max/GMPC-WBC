"""Lexicographic geometric MPC: hard physics > tracking > posture/smoothing.

Level 1 minimizes ONLY the existing weighted SE(3) pose/error-rate residual.
Level 2 locks every component of that horizon residual, then minimizes the
remaining original costs. No large weight approximates a priority relation.
Both levels use the same contact reduction and physical constraints. Numerical
acceptance is checked in the original coordinates, including every task lock.
"""
import time
import numpy as np
import scipy.linalg as la
from controller import Costs
from paper_gmpc import PaperGMPC
from osqp_acceleration import ContactReduction, PersistentOSQP, assemble
from central_relu import residuals

RESIDUAL_KEYS=('qp_primal_relative','qp_dual_relative','qp_normal_relative')

class PartitionedCosts(Costs):
    def __init__(self,n):
        super().__init__(n)
        self.primary=Costs(n);self.secondary=Costs(n);self.task_scales=[]

    def add(self,A,b,weight):
        super().add(A,b,weight)
        self.secondary.add(A,b,weight)

    def add_tracking(self,A,b,weight):
        super().add(A,b,weight)
        self.primary.add(A,b,weight)
        self.task_scales.append(np.sqrt(np.broadcast_to(weight,(len(b),))).copy())

    def metadata(self):
        if self.primary.a:
            h1,g1,C,d=self.primary.arrays();scales=np.concatenate(self.task_scales)
        else:
            h1=np.zeros((self.n,self.n));g1=np.zeros(self.n)
            C=np.zeros((0,self.n));d=np.zeros(0);scales=np.zeros(0)
        h2,g2,_,_=self.secondary.arrays()
        # Match the original tiny numerical ridge in the SECONDARY cost only.
        # Adding it to level 1 would change the intended primary optimum.
        h2=h2+1e-9*np.eye(self.n)
        return dict(primary_H=h1,primary_g=g1,tracking_C=C,tracking_d=d,
                    tracking_scales=scales,secondary_H=h2,secondary_g=g2)


def hierarchy_costs(parts):
    H1=la.block_diag(*[p['primary_H'] for p in parts])
    g1=np.concatenate([p['primary_g'] for p in parts])
    H2=la.block_diag(*[p['secondary_H'] for p in parts])
    g2=np.concatenate([p['secondary_g'] for p in parts])
    C=la.block_diag(*[p['tracking_C'] for p in parts])
    d=np.concatenate([p['tracking_d'] for p in parts])
    scales=np.concatenate([p['tracking_scales'] for p in parts])
    return H1,g1,H2,g2,C,d,scales


def independent_task_locks(C,scales,N,dt):
    """Keep 6N rate rows; analytically reconstruct all 12N tracking rows.

    For differences between predictions with the same initial state,
    delta(phi_1)=0 and delta(phi_(i+1))=delta(phi_i)+dt*delta(phidot_i).
    Thus locking every phidot_i also locks every phi_i. No numerical rank
    threshold or truncated singular values are used. Verify the identity on
    the actual weighted matrices on every feedback cycle.
    """
    if C.shape[0]!=12*N or scales.shape!=(12*N,) or np.any(scales<=0):
        raise ValueError('Unexpected task output ordering or weights')
    indices=np.concatenate([12*k+np.arange(6,12) for k in range(N)])
    D=C[indices]
    L=np.zeros((12*N,6*N))
    for k in range(N):
        L[12*k+6:12*k+12,6*k:6*k+6]=np.eye(6)
        for j in range(k):
            L[12*k:12*k+6,6*j:6*j+6]=np.diag(dt*scales[12*k:12*k+6]/scales[12*j+6:12*j+12])
    error=float(np.max(np.abs(C-L@D)))
    if error>1e-11*max(1.,float(np.max(np.abs(C)))):
        raise RuntimeError(f'Task recurrence no longer supports exact lock compression: {error}')
    return D,L,error


def recover_level(red,H,g,z,yi,C=None,yt=None):
    """Recover contact multipliers with the actual level's stationarity."""
    x=red.T@z+red.c
    grad=H@x+g+red.A[red.ineq].T@yi
    if C is not None:grad=grad+C.T@yt
    rhs=-grad[red.dep];ye=np.zeros(len(red.dep))
    for k in reversed(range(red.N)):
        rr=slice(24*k,24*(k+1));after=slice(24*(k+1),len(red.dep))
        ye[rr]=la.lu_solve(red.factors[k],rhs[rr]-red.D[after,rr].T@ye[after],trans=1)
    dual=np.zeros(len(red.l));dual[red.ineq]=yi;dual[red.eq]=red.erows*ye
    if C is not None:dual=np.r_[dual,yt]
    return x,dual


class HierarchicalGMPC(PaperGMPC):
    def __init__(self,*args,reuse=True,secondary_scale=1.,lock_mode='nullspace',algebra='builtin',**kwargs):
        super().__init__(*args,**kwargs)
        if self.mode!='central' or self.backend!='osqp':
            raise ValueError('The validated hierarchy uses centralized OSQP')
        if secondary_scale<=0:raise ValueError('Positive secondary scale required')
        if lock_mode not in ('nullspace','rate','all'):raise ValueError(lock_mode)
        self.secondary_scale=secondary_scale
        self.lock_mode=lock_mode
        self.primary_engine=PersistentOSQP(self.qp_tol,reuse=reuse,algebra=algebra)
        self.secondary_engine=PersistentOSQP(self.qp_tol,reuse=reuse,algebra=algebra)

    def make_costs(self,n):return PartitionedCosts(n)
    def add_tracking_cost(self,cost,A,b,weight):cost.add_tracking(A,b,weight)
    def cost_metadata(self,cost):return cost.metadata()

    def solve(self,parts):
        start=time.perf_counter();guesses=self.warm(parts)
        H,g,A,l,u=assemble(parts)
        H1,g1,H2,g2,C,d,scales=hierarchy_costs(parts)
        H2=H2*self.secondary_scale;g2=g2*self.secondary_scale
        assembly_ms=1000*(time.perf_counter()-start)
        t0=time.perf_counter();red=ContactReduction(parts,H1,g1,A,l,u)
        reduction_ms=1000*(time.perf_counter()-t0)
        warm=np.r_[*guesses][red.free]
        self.primary_engine.prepare(red.Hr,red.gr,red.Ar,red.lr,red.ur,warm)
        primary_setup_ms=self.primary_engine.prepare_ms
        t0=time.perf_counter();it1=0;recover_ms=0.;check_ms=0.
        for lev1 in range(7):
            z1,yi1,sinfo=self.primary_engine.solve(lev1);it1+=sinfo.iter
            tr=time.perf_counter();x1,dual1=recover_level(red,H1,g1,z1,yi1)
            recover_ms+=1000*(time.perf_counter()-tr);tr=time.perf_counter()
            m1=residuals(H1,g1,A,l,u,x1,dual1)
            check_ms+=1000*(time.perf_counter()-tr)
            if max(m1[k] for k in RESIDUAL_KEYS)<=self.qp_tol:break
        else:raise RuntimeError(f'Primary KKT acceptance failed: {m1}')
        primary_solve_ms=1000*(time.perf_counter()-t0)

        t0=time.perf_counter()
        D,L,compression_error=independent_task_locks(C,scales,self.N,self.dt)
        lock_C=C if self.lock_mode=='all' else D
        Cr=lock_C@red.T;target=Cr@z1;full_target=lock_C@x1
        hr2=red.T.T@H2@red.T;hr2=(hr2+hr2.T)/2
        gr2=red.T.T@(H2@red.c+g2)
        ar2=np.vstack((red.Ar,Cr));lr2=np.r_[red.lr,target];ur2=np.r_[red.ur,target]
        A2=np.vstack((A,lock_C));l2=np.r_[l,full_target];u2=np.r_[u,full_target]
        ns=self.lock_mode=='nullspace';ni=len(red.lr)
        null_error=0.;rank_ratio=1.
        if ns:
            # Full-rank task-rate map: exact equality parameterization,
            # never discard a small singular direction or soften priority.
            Q,R=la.qr(Cr.T,mode='full');nt=len(Cr);R=R[:nt]
            diagonal=np.abs(np.diag(R));rank_ratio=float(diagonal.min()/max(diagonal.max(),1e-30))
            if rank_ratio<1e-10:raise RuntimeError(f'Task lock is rank deficient: {rank_ratio}')
            Q1=Q[:,:nt];Z=Q[:,nt:]
            null_error=float(np.max(np.abs(Cr@Z)))
            if null_error>1e-10*max(1.,float(np.max(np.abs(Cr)))):
                raise RuntimeError(f'Task null-space identity lost accuracy: {null_error}')
            hs=Z.T@hr2@Z;gs=Z.T@(hr2@z1+gr2)
            hs=(hs+hs.T)/2;as_=red.Ar@Z;offset=red.Ar@z1
            ls=red.lr-offset;us=red.ur-offset;ws=np.zeros(Z.shape[1])
        else:hs,gs,as_,ls,us,ws=hr2,gr2,ar2,lr2,ur2,z1
        lock_build_ms=1000*(time.perf_counter()-t0)
        self.secondary_engine.prepare(hs,gs,as_,ls,us,ws)
        secondary_setup_ms=self.secondary_engine.prepare_ms
        t0=time.perf_counter();it2=0
        primary_residual=C@x1+d
        for lev2 in range(7):
            solution,ds,sinfo=self.secondary_engine.solve(lev2);it2+=sinfo.iter
            tr=time.perf_counter()
            if ns:
                z2=z1+Z@solution
                # Restore the task-lock multiplier before contact multipliers.
                grad=hr2@z2+gr2+red.Ar.T@ds
                yt=la.solve_triangular(R,-Q1.T@grad)
                yi2=np.r_[ds,yt]
            else:z2,yi2=solution,ds
            x2,dual2=recover_level(red,H2,g2,z2,yi2[:ni],lock_C,yi2[ni:])
            recover_ms+=1000*(time.perf_counter()-tr);tr=time.perf_counter()
            m2=residuals(H2,g2,A2,l2,u2,x2,dual2)
            lock=C@(x2-x1)
            lock_relative=float(np.max(np.abs(lock))/max(1.,np.max(np.abs(primary_residual))))
            check_ms+=1000*(time.perf_counter()-tr)
            if max(*(m2[k] for k in RESIDUAL_KEYS),lock_relative)<=self.qp_tol:break
        else:raise RuntimeError(f'Secondary KKT/task lock acceptance failed: {m2}, lock={lock_relative}')
        secondary_solve_ms=1000*(time.perf_counter()-t0)

        n0=len(parts[0]['g']);result=[x2[:n0],x2[n0:]]
        ys=[p['E']@w+p['d'] for p,w in zip(parts,result)]
        interface=float(np.max(np.abs(ys[0]-ys[1])))
        violation=float(max(0.,np.max(l-A@x2),np.max(A@x2-u)))
        r2=C@x2+d;j1=float(primary_residual@primary_residual);j2=float(r2@r2)
        physical_lock=(lock/scales).reshape(self.N,2,6)
        metrics={k:max(m1[k],m2[k]) for k in RESIDUAL_KEYS}
        metrics.update({f'primary_{k}':v for k,v in m1.items()})
        metrics.update({f'secondary_{k}':v for k,v in m2.items()})
        metrics.update(qp_primal_raw=max(m1['qp_primal_raw'],m2['qp_primal_raw']),
            qp_dual_raw=max(m1['qp_dual_raw'],m2['qp_dual_raw']),qp_objective=m2['qp_objective'],
            setup_ms=assembly_ms+reduction_ms+primary_setup_ms+lock_build_ms+secondary_setup_ms,
            solve_ms=primary_solve_ms+secondary_solve_ms,iterations=2,interface_scaled=interface,
            dual_residual=0.,local_primal=violation,local_iterations=it1+it2,admm_converged=int(interface<1e-5),
            qp_iterations=it1+it2,qp_converged=1,primary_iterations=it1,secondary_iterations=it2,
            primary_setup_ms=primary_setup_ms,secondary_setup_ms=secondary_setup_ms,
            primary_solve_ms=primary_solve_ms,secondary_solve_ms=secondary_solve_ms,
            primary_solver_api_ms=self.primary_engine.solve_wall_ms,secondary_solver_api_ms=self.secondary_engine.solve_wall_ms,
            preprocessing_ms=reduction_ms+lock_build_ms,kernel_setup_ms=primary_setup_ms+secondary_setup_ms,
            reduction_ms=reduction_ms,assembly_ms=assembly_ms,lock_build_ms=lock_build_ms,
            solver_prepare_ms=primary_setup_ms+secondary_setup_ms,dual_recovery_ms=recover_ms,
            common_check_ms=check_ms,osqp_rebuilt=self.primary_engine.rebuilt+self.secondary_engine.rebuilt,
            pattern_growth=self.primary_engine.pattern_growth+self.secondary_engine.pattern_growth,
            optimized_variables=len(gs),optimized_constraints=len(ls),primary_variables=len(red.gr),primary_constraints=len(red.lr),
            elimination_map_max=red.equality_map_max,elimination_offset_max=red.equality_offset_max,
            qp_refinements=lev1+lev2,task_lock_relative=lock_relative,
            task_lock_rows=len(lock_C),task_lock_compression_error=compression_error,
            task_nullspace_error=null_error,task_lock_rank_ratio=rank_ratio,
            task_lock_weighted_max=float(np.max(np.abs(lock))),
            task_lock_pose_linear_max=float(np.max(np.abs(physical_lock[:,0,:3]))),
            task_lock_pose_angular_max=float(np.max(np.abs(physical_lock[:,0,3:]))),
            task_lock_rate_max=float(np.max(np.abs(physical_lock[:,1]))),
            primary_objective=j1,primary_objective_after_secondary=j2,
            primary_objective_change=j2-j1,
            secondary_objective_before=float(.5*x1@H2@x1+g2@x1),
            secondary_objective_after=float(.5*x2@H2@x2+g2@x2),
            secondary_move_norm=float(la.norm(x2-x1)))
        self.last=[x.copy() for x in result]
        self.previous_rotation=self.current_rotation.copy() if hasattr(self,'current_rotation') else None
        self.last_reduction=red;self.last_primary=x1;self.last_secondary=x2
        self.last_primary_dual=dual1;self.last_full_dual=dual2
        self.hierarchy_data=dict(H1=H1,g1=g1,H2=H2,g2=g2,C=C,d=d,lock_C=lock_C,lock_lift=L,A=A,l=l,u=u,A2=A2,l2=l2,u2=u2,
                                 primary_metrics=m1,secondary_metrics=m2)
        return result,metrics
