"""Fixed-contact hierarchical WBC with matched product/SE(3) error feedback.

Pinocchio order: [linear; angular]. E=Td^-1 T. Joint configuration is never
subtracted across the floating-base quaternion. The log-map drift derivative
uses a centered directional difference (two small Jlog evaluations), not a
claim of analytic second derivatives. No activation/motion allocator is used.
"""
import time
import numpy as np
import scipy.linalg as la
import pinocchio as pin
import quadprog
from model import contact, equilibrium
from central_relu import residuals


def pose(q):
    return pin.SE3(pin.Quaternion(q[6], *q[3:6]).matrix(), q[:3])


def rolling_contact(s):
    """Sphere bottom no-slip velocity; only normal position is holonomic.

    Native Jc maps to v_center + omega x (0,0,-radius). Its drift differentiates
    that velocity with the world vertical offset fixed. Horizontal center
    position must NOT be anchored while the sphere rolls without slip.
    """
    s = dict(s)
    s['ce'] = s['ce'].copy().reshape(4,3)
    s['ce'][:,:2] = 0.
    s['ce'] = s['ce'].ravel()
    return s


def task_error(T, V, drift, J, Td, Vd, Ad, mode, eps=1e-5):
    """Return e, edot, Je, be with eddot=Je*a+be, for a moving reference.

    V and drift are body twist and its derivative at generalized a=0.
    Ad is the time derivative of the reference body twist.
    """
    E = Td.inverse()*T
    if mode == 'se3':
        transported = E.inverse().action@Vd
        eta = V-transported
        L = pin.Jlog6(E)
        Ldot = (pin.Jlog6(E*pin.exp6(eps*eta))-pin.Jlog6(E*pin.exp6(-eps*eta)))/(2*eps)
        edot = L@eta
        b = L@(drift+pin.Motion(eta).action@transported-E.inverse().action@Ad)+Ldot@eta
        return pin.log6(E).vector, edot, L@J, b
    if mode != 'product':
        raise ValueError(mode)
    Re = E.rotation
    wr = Re.T@Vd[3:]
    eta = V[3:]-wr
    L = pin.Jlog3(Re)
    Ldot = (pin.Jlog3(Re@pin.exp3(eps*eta))-pin.Jlog3(Re@pin.exp3(-eps*eta)))/(2*eps)
    e = np.r_[T.translation-Td.translation, pin.log3(Re)]
    edot = np.r_[T.rotation@V[:3]-Td.rotation@Vd[:3], L@eta]
    Je = np.vstack((T.rotation@J[:3], L@J[3:]))
    b = np.r_[T.rotation@(drift[:3]+np.cross(V[3:], V[:3]))-
              Td.rotation@(Ad[:3]+np.cross(Vd[3:], Vd[:3])),
              L@(drift[3:]+np.cross(eta, wr)-Re.T@Ad[3:])+Ldot@eta]
    return e, edot, Je, b


def cone_matrix():
    mu = .6/np.sqrt(2)
    return la.block_diag(*([np.array([[1,0,-mu],[-1,0,-mu],[0,1,-mu],[0,-1,-mu],[0,0,1]])]*4))


class GeometricWBC:
    """Hard physics > base pose > EE pose > posture/force/torque smoothing.

    Rigid contact KKT elimination makes a=T*tau+a0 and f=F*tau+f0.
    All levels enforce the same torque, force, slew and one-step state bounds.
    Nullspace parameterization preserves each preceding task output exactly.
    The formulation and all gains are identical for both error choices.
    """
    def __init__(self, plant, mode='product', dt=.005, qp_tol=2e-7):
        self.plant, self.mode, self.dt = plant, mode, dt
        self.scale = np.r_[np.full(12,100.),np.full(6,10.)]
        self.prev_tau = equilibrium(plant.truth())
        self.qp_tol = qp_tol
        self.cone = cone_matrix()
        self.Jbase = np.eye(24)[:6]
        self.active_by_level={}
        self.fallbacks=0

    def solve_level(self, i, C, d, G, lo, hi, warm):
        H = 2*C.T@C
        g = 2*C.T@d
        objective_scale=max(1.,float(np.max(np.abs(H))))
        # A task lock can fix a bounded actuator. Numerical null-space rows
        # then look like 1e-12*z >= 1e-9 and falsely demand enormous z. Check
        # the inherited bound in physical units before removing zero rows.
        zero=la.norm(G,axis=1)<1e-10
        if np.any(zero) and max(0.,np.max(lo[zero]),np.max(-hi[zero]))>2e-7:
            raise RuntimeError('Inherited zero-row bound is infeasible')
        Gsolve=G.copy();lsolve=lo.copy();usolve=hi.copy()
        if i>0:
            # z=0 is the inherited feasible higher-priority solution. Snap
            # only its sub-tolerance bound roundoff to zero; original bounds
            # remain in residuals() and in the final physical gate.
            lsolve[(lsolve>0)&(lsolve<2e-7)]=0.
            usolve[(usolve<0)&(usolve>-2e-7)]=0.
        Gsolve[zero]=0.;lsolve[zero]=-np.inf;usolve[zero]=np.inf
        il=np.flatnonzero(np.isfinite(lsolve));iu=np.flatnonzero(np.isfinite(usolve))
        constraints=np.vstack((Gsolve[il],-Gsolve[iu])).T
        bounds=np.r_[lsolve[il],-usolve[iu]]
        row_scale=1/np.maximum(la.norm(constraints,axis=0),1e-12)
        # A 1e-10 diagonal makes semidefinite task Hessians usable by the small
        # active-set solver. Acceptance checks the UNREGULARIZED objective.
        try:
            solution=quadprog.solve_qp(H/objective_scale+1e-10*np.eye(len(g)),
                -g/objective_scale,constraints*row_scale,bounds*row_scale)
        except ValueError as exc:
            np.savez('/tmp/geometric_wbc_failed_qp.npz',H=H,g=g,G=G,lo=lo,hi=hi,warm=warm,level=i)
            # Nearly opposing active inequalities can defeat Goldfarb-Idnani
            # despite a feasible inherited point. Use the SAME small QP with
            # SLSQP, recover nonnegative active multipliers, and accept only
            # the original unregularized KKT + physical checks below.
            from scipy.optimize import minimize,nnls
            hn=H/objective_scale;gn=g/objective_scale
            cq=constraints*row_scale;bd=bounds*row_scale
            candidate=minimize(lambda z:.5*z@hn@z+gn@z,warm,
                jac=lambda z:hn@z+gn,method='SLSQP',
                constraints={'type':'ineq','fun':lambda z:cq.T@z-bd,'jac':lambda z:cq.T},
                options={'ftol':1e-13,'maxiter':300,'disp':False})
            x=candidate.x;active=np.flatnonzero(cq.T@x-bd<1e-8)
            multipliers=np.zeros(len(bd))
            if len(active):multipliers[active]=nnls(cq[:,active],hn@x+gn,maxiter=3000)[0]
            solution=(x,None,None,[candidate.nit],multipliers)
            self.fallbacks+=1
        x=solution[0];dual=solution[4]*row_scale;y=np.zeros(len(lo))
        np.add.at(y,il,-dual[:len(il)]);np.add.at(y,iu,dual[len(il):])
        res=residuals(H/objective_scale,g/objective_scale,G,lo,hi,x,y)
        if max(res[k] for k in ['qp_primal_relative','qp_dual_relative','qp_normal_relative'])>self.qp_tol:
            raise RuntimeError(f'Level {i} residual gate: {res}')
        # Positive multipliers identify the optimal face: every minimizer of
        # this convex task must retain these active boundaries. Carrying them
        # explicitly avoids nearly dependent inequalities in later stages.
        self.active_by_level[i]=np.flatnonzero(np.abs(y)>1e-8)
        return x,res,int(solution[3][0])

    def control(self, s, reference):
        start = time.perf_counter()
        self.fallbacks=0
        s = rolling_contact(s)
        T,a0,F,f0 = contact(s)
        A = T*self.scale[None,:]
        B = F*self.scale[None,:]
        dt = self.dt
        blocks = [np.diag(self.scale), self.cone@B, np.diag(self.scale), dt*A[6:], .5*dt*dt*A[6:], A[6:]]
        lo = np.r_[-.85*s['tau_max'], np.tile([-np.inf]*4+[10.],4)-self.cone@f0,
            self.prev_tau-dt*np.r_[np.full(12,1000.),np.full(6,100.)],
            -np.minimum(s['v_max'],4.)-s['v'][6:]-dt*a0[6:],
            s['q_min']+.02-s['q'][7:]-dt*s['v'][6:]-.5*dt*dt*a0[6:],
            -np.r_[np.full(12,50.),np.full(6,30.)]-a0[6:]]
        hi = np.r_[.85*s['tau_max'], np.tile([0.]*4+[800.],4)-self.cone@f0,
            self.prev_tau+dt*np.r_[np.full(12,1000.),np.full(6,100.)],
            np.minimum(s['v_max'],4.)-s['v'][6:]-dt*a0[6:],
            s['q_max']-.02-s['q'][7:]-dt*s['v'][6:]-.5*dt*dt*a0[6:],
            np.r_[np.full(12,50.),np.full(6,30.)]-a0[6:]]
        # Optional moving arm-reference envelope, used only by the new online
        # planner/WBC experiment. The original standalone controller is intact.
        if hasattr(self,'arm_tracking_accel_bounds'):
            arm_lo,arm_hi=self.arm_tracking_accel_bounds
            blocks.append(A[18:]);lo=np.r_[lo,arm_lo-a0[18:]];hi=np.r_[hi,arm_hi-a0[18:]]
        G = np.vstack(blocks)
        terms = []
        for frame, current_T, J, drift, kp, kd in [
            ('base',pose(s['q']),self.Jbase,np.zeros(6),200.,28.),
            ('ee',pin.SE3(s['pose'][:3,:3],s['pose'][:3,3]),s['J'],s['jdotv'],100.,20.)]:
            Td = pin.SE3(reference[f'{frame}_pose'][:3,:3],reference[f'{frame}_pose'][:3,3])
            e,edot,Je,be = task_error(current_T,J@s['v'],drift,J,Td,
                reference[f'{frame}_twist'],reference[f'{frame}_acc'],self.mode)
            target = -be-kp*e-kd*edot
            # Same local metric for both methods: 5 cm and 0.2 rad scales.
            w = np.r_[np.full(3,20.),np.full(3,5.)]
            terms.append((w[:,None]*(Je@A), w*(Je@a0-target), e, edot))
        model_ms = 1000*(time.perf_counter()-start)
        x = np.zeros(18);Z = np.eye(18);locked = [];metrics = [];iters = [];ranks = []
        def empty_metrics():return dict(qp_primal_relative=0.,qp_dual_relative=0.,qp_normal_relative=0.)
        for i,(C,d,_,_) in enumerate(terms):
            Cr = C@Z
            if Z.shape[1]:
                reduced, res, count = self.solve_level(i,Cr,C@x+d,G@Z,lo-G@x,hi-G@x,
                    self.prev_tau/self.scale if i==0 else np.zeros(Z.shape[1]))
                x = x+Z@reduced
            else:
                res,count=empty_metrics(),0;self.active_by_level[i]=np.array([],dtype=int)
            locked.append((C.copy(),C@x))
            metrics.append(res);iters.append(count)
            face=np.vstack((Cr,(G@Z)[self.active_by_level[i]]))
            norms=la.norm(face,axis=1);face=face[norms>1e-10]/norms[norms>1e-10,None]
            Zi = la.null_space(face,rcond=1e-10)
            ranks.append(Z.shape[1]-Zi.shape[1]);Z=Z@Zi
        qerr = self.plant.difference(s['q'],reference['q'])
        target_a = reference['a']+100*qerr+20*(reference['v']-s['v'])
        C3 = np.vstack((.1*A[6:], .1*B, np.eye(18), .2*np.eye(18)))
        d3 = np.r_[.1*(a0[6:]-target_a[6:]), .1*(f0-reference['force']),
            -self.prev_tau/self.scale, -.2*reference['tau']/self.scale]
        if Z.shape[1]:
            z,res,count = self.solve_level(2,C3@Z,C3@x+d3,G@Z,lo-G@x,hi-G@x,np.zeros(Z.shape[1]))
            x=x+Z@z
        else:res,count=empty_metrics(),0
        metrics.append(res);iters.append(count)
        lock = max(float(np.max(np.abs(C@x-value))) for C,value in locked)
        tau = self.scale*x;acc=T@tau+a0;force=F@tau+f0
        dynamics = s['M']@acc+s['h']-np.r_[np.zeros(6),tau]-s['Jc'].T@force
        contact_res = s['Jc']@acc+s['cdotv']+100*s['ce']+20*s['Jc']@s['v']
        violation = float(max(0,np.max(lo-G@x),np.max(G@x-hi)))
        if lock>2e-7 or violation>2e-5 or np.max(np.abs(dynamics))>1e-7 or np.max(np.abs(contact_res))>1e-7:
            raise RuntimeError(f'WBC physical/lock gate: {lock}, {violation}, {dynamics}, {contact_res}')
        self.prev_tau=tau.copy()
        self.info=dict(controller_ms=1000*(time.perf_counter()-start),model_ms=model_ms,
            qp_primal_relative=max(m['qp_primal_relative'] for m in metrics),
            qp_dual_relative=max(m['qp_dual_relative'] for m in metrics),
            qp_normal_relative=max(m['qp_normal_relative'] for m in metrics),
            constraint_violation=violation,contact_residual=float(np.max(np.abs(contact_res))),
            dynamics_residual=float(np.max(np.abs(dynamics))),task_lock_max=lock,
            qp_iterations=sum(iters),base_iterations=iters[0],ee_iterations=iters[1],smooth_iterations=iters[2],
            task_error_norm=float(np.linalg.norm(terms[0][2])),
            remaining_dof_after_base=18-ranks[0],remaining_dof_after_ee=Z.shape[1],
            setups=sum(i>0 for i in iters),pattern_growth=0,qp_fallbacks=self.fallbacks)
        self.debug=dict(tau=tau.copy(),acc=acc.copy(),force=force.copy(),
            primary_achieved=locked[0][1],secondary_achieved=locked[1][1])
        return tau,acc
