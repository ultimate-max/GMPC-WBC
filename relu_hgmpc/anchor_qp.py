"""Certified same-QP recovery for screened OSQP numerical failure."""
import numpy as np
import scipy.linalg as la
import quadprog
from online_osqp import OnlineOSQP
from central_relu import residuals
from shared_hierarchy import HierarchyQP

class CertifiedOSQP(OnlineOSQP):
    def solve(self,H,g,G,lo,hi,*args):
        try:return super().solve(H,g,G,lo,hi,*args)
        except (RuntimeError,ValueError) as exc:
            # Every lower level inherits a feasible zero increment. Preserve
            # the original QP and certify its unregularized KKT on recovery.
            il=np.flatnonzero(np.isfinite(lo));iu=np.flatnonzero(np.isfinite(hi))
            C=np.vstack((G[il],-G[iu])).T;b=np.r_[lo[il],-hi[iu]]
            scale=1/np.maximum(la.norm(C,axis=0),1e-12)
            try:
                sol=quadprog.solve_qp(H,-g,C*scale,b*scale)
                x=sol[0];mult=sol[4]*scale;iterations=int(sol[3][0])
            except ValueError:
                # A 1e-10 physical-unit roundoff allowance prevents opposing
                # inherited faces from being declared inconsistent. It is
                # 2000x smaller than the unchanged 2e-7 acceptance tolerance.
                # Certification below always uses ORIGINAL lo/hi and H/g.
                inherited=max(0.,float(np.max(lo)),float(np.max(-hi)))
                np.savez('/tmp/anchor_lower_qp.npz',H=H,g=g,G=G,lo=lo,hi=hi,inherited=inherited)
                if inherited>self.tol:raise
                sol=quadprog.solve_qp(H,-g,C*scale,(b-1e-10)*scale)
                x=sol[0];mult=sol[4]*scale;iterations=int(sol[3][0])
                self.roundoff_recoveries=getattr(self,'roundoff_recoveries',0)+1
            y=np.zeros(len(lo))
            np.add.at(y,il,-mult[:len(il)]);np.add.at(y,iu,mult[len(il):])
            rr=residuals(H,g,G,lo,hi,x,y)
            error=max(rr[k] for k in ('qp_primal_relative','qp_dual_relative','qp_normal_relative'))
            if error>self.tol and max(0.,float(np.max(lo)),float(np.max(-hi)))<=self.tol:
                sol=quadprog.solve_qp(H,-g,C*scale,(b-1e-10)*scale)
                x=sol[0];mult=sol[4]*scale;iterations+=int(sol[3][0]);y=np.zeros(len(lo))
                np.add.at(y,il,-mult[:len(il)]);np.add.at(y,iu,mult[len(il):])
                rr=residuals(H,g,G,lo,hi,x,y)
                error=max(rr[k] for k in ('qp_primal_relative','qp_dual_relative','qp_normal_relative'))
                self.roundoff_recoveries=getattr(self,'roundoff_recoveries',0)+1
            if error>self.tol:
                np.savez('/tmp/anchor_qp_recovery_failure.npz',H=H,g=g,G=G,lo=lo,hi=hi,x=x,y=y)
                raise RuntimeError(f'Same-QP recovery certificate failed: {rr}; initial={exc}')
            self.recoveries=getattr(self,'recoveries',0)+1
            self.previous_active=np.array([],int)
            return x,y,iterations,dict(active_face_fastpath=0,recovery=str(exc))

class AnchorQP(HierarchyQP):
    def __init__(self):super().__init__(engine_factory=CertifiedOSQP)
    def solve(self,*args):
        x,info=super().solve(*args)
        info['same_qp_recoveries']=getattr(self.engine,'recoveries',0)
        info['roundoff_recoveries']=getattr(self.engine,'roundoff_recoveries',0)
        return x,info
