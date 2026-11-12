"""Continuous constrained velocity allocation; not MPC or a dynamics split.

The two ADMM blocks are base chart velocity and arm joint velocity. They share
the world-frame task velocity equality. WBC still handles coupled dynamics.
"""
import numpy as np
import quadprog


def box_qp(H, a, low, high):
    n=len(a)
    return quadprog.solve_qp(H, a, np.column_stack((np.eye(n),-np.eye(n))),
                            np.r_[low,-high])[0]


class SmoothDistributor:
    def __init__(self, backend='central', iterations=80, rho=20.):
        if backend not in ('central','admm') or iterations<1 or rho<=0:
            raise ValueError('Invalid smooth allocation solver configuration')
        self.backend=backend;self.iterations=iterations;self.rho=rho
        self.dual=None
        # Normalize m/s and rad/s before assigning the base preference.
        self.scale=np.r_[np.full(3,.25),np.full(3,.5),np.ones(6)]
        self.weight=np.r_[np.full(6,25.),np.ones(6)]
        self.temporal=2.

    def solve(self,J,task,low,high,secondary,previous,dt,amax):
        # These are command slew limits, distinct from braking-distance bounds.
        lo=np.maximum(low,previous-dt*amax)
        hi=np.minimum(high,previous+dt*amax)
        if np.any(lo>hi+1e-10):
            raise RuntimeError('Physical bounds and velocity slew bounds conflict')
        scale=self.scale
        # 0.3 m is the explicit characteristic length for mixed 6D task rows.
        row=np.r_[np.full(min(3,len(task)),1/.3),np.ones(max(0,len(task)-3))]
        A=row[:,None]*J*scale[None,:];b=row*task
        yl=lo/scale;yu=hi/scale;yp=previous/scale
        H=np.diag(self.weight+self.temporal)
        a=self.temporal*yp+secondary/scale
        if self.backend=='central':
            C=np.column_stack((A.T,np.eye(12),-np.eye(12)))
            y=quadprog.solve_qp(H,a,C,np.r_[b,yl,-yu],meq=len(b))[0]
            count=1;dual_residual=0.
        else:
            # Standard sequential two-block ADMM; no claim of parallel speedup.
            Ab,Am=A[:,:6],A[:,6:];rho=self.rho
            d=np.zeros(len(b)) if self.dual is None else self.dual.copy()
            y=np.clip(yp,yl,yu)
            Hb=H[:6,:6]+rho*Ab.T@Ab;Hm=H[6:,6:]+rho*Am.T@Am
            dual_residual=np.inf
            for count in range(1,self.iterations+1):
                old_arm=y[6:].copy()
                y[:6]=box_qp(Hb,a[:6]+rho*Ab.T@(b-Am@y[6:]-d),yl[:6],yu[:6])
                y[6:]=box_qp(Hm,a[6:]+rho*Am.T@(b-Ab@y[:6]-d),yl[6:],yu[6:])
                residual=A@y-b;d+=residual
                dual_residual=float(np.max(np.abs(rho*Ab.T@Am@(y[6:]-old_arm))))
                if np.max(np.abs(residual))<1e-8 and dual_residual<1e-7:break
            self.dual=d
        v=scale*y
        residual=float(np.max(np.abs(J@v-task)))
        scaled_residual=float(np.max(np.abs(A@y-b)))
        bound_error=float(max(0,np.max(lo-v),np.max(v-hi)))
        slew=float(max(0,np.max(np.abs(v-previous)/dt-amax)))
        if bound_error>1e-8 or scaled_residual>1e-4:
            raise RuntimeError(f'Allocation not accepted: normalized task residual={scaled_residual}, bound error={bound_error}')
        self.last_problem=dict(H=H.copy(),a=a.copy(),A=A.copy(),b=b.copy(),
                               low=yl.copy(),high=yu.copy(),y=y.copy())
        return v,dict(base_active=1,saturations=0,phi=1.,psi=1.,scaling_lp=0,
            velocity_violation=bound_error,distribution_task_residual=residual,
            tertiary_task_change=0.,frozen_dof_change=0.,slew_violation=slew,
            allocation_iterations=count,allocation_dual_residual=dual_residual,
            allocation_primal_residual=scaled_residual,
            allocation_converged=int(scaled_residual<1e-8 and dual_residual<1e-7))
