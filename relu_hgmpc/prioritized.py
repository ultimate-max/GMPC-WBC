"""Xie et al. (2025) motion distribution + hierarchical WBC, standing mode.

The equation-to-code audit in PRIORITIZED.md explicitly documents corrections
required by the printed Jacobian / selection / null-space expressions.
"""
import time
import pathlib
import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import pinocchio as pin
import quadprog

def mp(A): return la.pinv(A,rtol=1e-10)

def rpy_rate_map(rpy):
    r,p,_=rpy
    return np.array([[1,0,-np.sin(p)],[0,np.cos(r),np.cos(p)*np.sin(r)],[0,-np.sin(r),np.cos(p)*np.cos(r)]])

def rpy_rate_map_dot(rpy,rates):
    r,p,_=rpy;rd,pd,_=rates
    Er=np.array([[0,0,0],[0,-np.sin(r),np.cos(p)*np.cos(r)],[0,-np.cos(r),-np.cos(p)*np.sin(r)]])
    Ep=np.array([[0,0,-np.cos(p)],[0,0,-np.sin(p)*np.sin(r)],[0,0,-np.sin(p)*np.cos(r)]])
    return Er*rd+Ep*pd

def shaped_bounds(q,lo,hi,vmax,amax,dt):
    """Paper (12),(13): position, velocity and braking-distance bounds."""
    lower=np.maximum.reduce([(lo-q)/dt,-vmax,-np.sqrt(np.maximum(0,2*amax*(q-lo)))])
    upper=np.minimum.reduce([(hi-q)/dt,vmax,np.sqrt(np.maximum(0,2*amax*(hi-q)))])
    if np.any(lower>upper+1e-8):raise RuntimeError('No feasible velocity inside position/braking bounds')
    return lower,upper

def distribute(J,ff,feedback,lo,hi,secondary,force_base=False):
    """Arm first SNS; disabled/saturated velocities are subtracted explicitly.

    Phi scales the feedforward part only, as in paper (18). Psi maximizes the
    feasible tertiary step. Activation is latched within one active-set search.
    """
    n=J.shape[1];r=J.shape[0];free=np.r_[np.full(6,force_base),np.ones(n-6,dtype=bool)]
    fixed=np.zeros(n);base=bool(force_base);saturations=0;used_lp=False;phi=1.
    task=ff+feedback
    for _ in range(n+2):
        JS=J*free[None,:]
        if np.linalg.matrix_rank(JS,tol=1e-9)<r and not base:
            base=True;free[:6]=True;JS=J*free[None,:]
        v=fixed+mp(JS)@(task-J@fixed)
        violation=np.maximum(lo-v,v-hi)
        if np.max(violation)<1e-9 and np.max(np.abs(J@v-task))<1e-7:break
        candidates=np.where(free,violation/np.maximum(hi-lo,1e-5),-np.inf)
        j=int(np.argmax(candidates))
        if candidates[j]<=0:break
        fixed[j]=lo[j] if v[j]<lo[j] else hi[j];free[j]=False;saturations+=1
    if np.max(np.maximum(lo-v,v-hi))>1e-8 or np.max(np.abs(J@v-task))>1e-7:
        # Maximal feasible task scaling. Bounds describe the physical DOFs,
        # not the possibly suboptimal order of the preceding saturation search.
        base=True;used_lp=True
        res=opt.linprog(np.r_[np.zeros(n),-1.],A_eq=np.column_stack((J,-ff)),b_eq=feedback,
                        bounds=list(zip(lo,hi))+[(0.,1.)],method='highs')
        if not res.success:raise RuntimeError('Paper (18) infeasible even with feedforward scaling: '+res.message)
        v=res.x[:n];phi=float(res.x[-1]);free=(v>lo+1e-7)&(v<hi-1e-7)
    # The intersection of null(J) and fixed-DOF constraints implements the
    # (17): for D=I-Se, (D P)^dagger D=(D P)^dagger, so the printed
    # expression equals projection onto null(J) intersect null(D). P must
    # be defined on the full 12-DOF task Jacobian, not a 6-DOF arm alone.
    D=np.eye(n)[~free];Z=la.null_space(np.vstack((J,D)),rcond=1e-10)
    dz=Z@(Z.T@secondary);psi=1.
    for j in range(n):
        if dz[j]>1e-12:psi=min(psi,(hi[j]-v[j])/dz[j])
        elif dz[j]<-1e-12:psi=min(psi,(lo[j]-v[j])/dz[j])
    psi=float(np.clip(psi,0,1));out=v+psi*dz
    info=dict(base_active=int(base),saturations=saturations,phi=phi,psi=psi,scaling_lp=int(used_lp),
              velocity_violation=float(max(0,np.max(lo-out),np.max(out-hi))),
              distribution_task_residual=float(np.max(np.abs(J@out-(phi*ff+feedback)))),
              tertiary_task_change=float(np.max(np.abs(J@(psi*dz)))),
              frozen_dof_change=float(np.max(np.abs(D@(psi*dz)))) if len(D) else 0.)
    return out,info

def hierarchy(M,tasks):
    """Equations (20),(21), with inverses restricted to previous task nullspace.

    tasks contain (J, configuration error, desired velocity, desired acceleration
    including drift compensation). Inverse-mass weighting is used for acceleration.
    """
    n=len(M);Z=np.eye(n);dq=np.zeros(n);vel=np.zeros(n);acc=np.zeros(n)
    preserved=[];maxchange=0.
    for J,e,v,a in tasks:
        Jr=J@Z
        kin=Z@mp(Jr)
        dq+=kin@(e-J@dq);vel+=kin@(v-J@vel)
        if Z.shape[1]:
            MinvJ=la.solve(Z.T@M@Z,Jr.T,assume_a='pos')
            dyn=Z@MinvJ@mp(Jr@MinvJ)
            change=dyn@(a-J@acc);acc+=change
            for old in preserved:maxchange=max(maxchange,float(np.max(np.abs(old@change))))
        preserved.append(J)
        Z=Z@la.null_space(Jr,rcond=1e-10)
    return dq,vel,acc,maxchange

def solve_wbc_qp(A,low,upp):
    """Same SPD WBC QP, whitened and solved by Goldfarb-Idnani; no relaxation."""
    scale=1/np.sqrt(np.r_[np.full(6,2e4),np.full(12,2e-4)])
    As=A*scale[None,:];rowscale=1/np.maximum(np.linalg.norm(As,axis=1),1e-10)
    G=rowscale[:,None]*As;lo=rowscale*low;hi=rowscale*upp
    haslo=np.isfinite(lo[6:]);hashi=np.isfinite(hi[6:])
    C=np.vstack((G[:6],G[6:][haslo],-G[6:][hashi])).T.copy()
    b=np.r_[lo[:6],lo[6:][haslo],-hi[6:][hashi]]
    result=quadprog.solve_qp(np.eye(18),np.zeros(18),C,b,meq=6)
    x=result[0];dual=float(np.max(np.abs(x-C@result[4])));solution=scale*x
    primal=float(max(0,np.max(low-A@solution),np.max(A@solution-upp)))
    return solution,primal,dual,int(result[3][0])

class PrioritizedController:
    def __init__(self,plant,ref,task_dim=3,dt=.02,sigma_min=.1,activation_mode='threshold',allocation_iterations=80,task_error='cartesian'):
        self.plant=plant;self.ref=ref;self.r=task_dim;self.dt=dt;self.sigma_min=sigma_min
        s=plant.truth();self.home=s['q'].copy();self.desired=self.coordinates(s)
        self.previous_velocity=np.zeros(12)
        self.kx=10.;self.kN=5.;self.kD=.5;self.index=0
        self.activation_mode=activation_mode;self.activation_latched=False
        self.task_error=task_error
        if task_error not in ('cartesian','se3'):raise ValueError('Unknown task error')
        if task_error=='se3' and task_dim!=6:raise ValueError('SE(3) error requires a 6D task')
        self.smooth=None
        if activation_mode in ('smooth_qp','smooth_admm'):
            from smooth_distribution import SmoothDistributor
            self.smooth=SmoothDistributor('central' if activation_mode=='smooth_qp' else 'admm',allocation_iterations)

    def coordinates(self,s):
        q=s['q'];R=pin.Quaternion(q[6],*q[3:6]).matrix()
        return np.r_[q[:3],pin.rpy.matrixToRpy(R),q[19:]]

    def chart(self,s):
        qbw=self.coordinates(s);roll,pitch,yaw=qbw[3:6]
        R=pin.rpy.rpyToMatrix(qbw[3:6]);E=rpy_rate_map(qbw[3:6])
        L=np.zeros((24,12));L[:3,:3]=R.T;L[3:6,3:6]=E;L[18:,6:]=np.eye(6)
        Rt=s['pose'][:3,:3];Jworld=la.block_diag(Rt,Rt)@s['J']
        return qbw,L,Jworld[:self.r]@L

    def manipulability(self,s):
        J=s['J'][:self.r,18:]
        return float(np.prod(la.svdvals(J))),float(la.svdvals(J)[-1])

    def allocate(self,s,t):
        start=time.perf_counter();qbw,L,J=self.chart(s)
        velocity=mp(L)@s['v'];pose=self.ref.pose_at(t);vd=self.ref.twist(t)
        Rref=pose[:3,:3]
        ff=np.r_[Rref@vd[:3],Rref@vd[3:]][:self.r]
        error=np.r_[pose[:3,3]-s['pose'][:3,3],pin.log3(Rref@s['pose'][:3,:3].T)][:self.r]
        if self.task_error=='se3':
            actual=pin.SE3(s['pose'][:3,:3],s['pose'][:3,3])
            desired=pin.SE3(pose[:3,:3],pose[:3,3]);E=actual.inverse()*desired
            rotation=la.block_diag(actual.rotation,actual.rotation)
            # Body-frame command V=Ad_E Vd+k log(E) mapped to the same
            # local-world-aligned point-velocity coordinates as chart().
            ff=rotation@(E.action@vd)
            error=rotation@pin.log6(E).vector
        H,sigma=self.manipulability(s);grad=np.zeros(6);eps=1e-5
        for j in range(6):
            d=np.zeros(24);d[18+j]=eps
            plus=self.plant.evaluate(self.plant.integrate(s['q'],d),s['v'])
            minus=self.plant.evaluate(self.plant.integrate(s['q'],-d),s['v'])
            grad[j]=(self.manipulability(plus)[0]-self.manipulability(minus)[0])/(2*eps)
        sec=self.kN*np.r_[np.zeros(6),grad]-self.kD*velocity
        qhome=self.coordinates(dict(q=self.home))
        lo=np.r_[qhome[:3]+[-.15,-.15,-.08],np.full(3,-.25),s['q_min'][12:]+.02]
        hi=np.r_[qhome[:3]+[.15,.15,.08],np.full(3,.25),s['q_max'][12:]-.02]
        vmax=np.r_[np.full(3,.25),np.full(3,.5),np.minimum(s['v_max'][12:],4.)]
        amax=np.r_[np.full(3,2.),np.full(3,3.),np.full(6,8.)]
        vlo,vhi=shaped_bounds(qbw,lo,hi,vmax,amax,self.dt)
        force_base=sigma<self.sigma_min
        if self.activation_mode=='always':force_base=True
        elif self.activation_mode=='arm':force_base=False
        elif self.activation_mode=='hysteresis':
            if sigma<self.sigma_min:self.activation_latched=True
            elif sigma>1.2*self.sigma_min:self.activation_latched=False
            force_base=self.activation_latched
        elif self.activation_mode not in ('threshold','smooth_qp','smooth_admm'):raise ValueError('unknown activation mode')
        if self.smooth is None:
            command,di=distribute(J,ff,self.kx*error,vlo,vhi,sec,force_base=force_base)
        else:
            command,di=self.smooth.solve(J,ff+self.kx*error,vlo,vhi,sec,self.previous_velocity,self.dt,amax)
        # Integrate prescribed base and arm commands as described before (20).
        desired_acc=(command-self.previous_velocity)/self.dt
        self.desired+=self.dt*command;self.previous_velocity=command.copy()
        distribution_ms=1000*(time.perf_counter()-start)
        self.allocation_info=dict(distribution_ms=distribution_ms,sigma_min=sigma,manipulability=H,**di)
        return qbw,command,desired_acc,ff,error

    def control(self,s,t):
        start=time.perf_counter()
        qbw,command,desired_acc,ff,error=self.allocate(s,t)
        ai=self.allocation_info;distribution_ms=ai['distribution_ms'];sigma=ai['sigma_min'];H=ai['manipulability']
        di={k:v for k,v in ai.items() if k not in ('distribution_ms','sigma_min','manipulability')}
        ts=time.perf_counter();R=pin.rpy.rpyToMatrix(qbw[3:6]);Rd=pin.rpy.rpyToMatrix(self.desired[3:6])
        Jrot=np.zeros((3,24));Jrot[:,3:6]=np.eye(3)
        Jpos=np.zeros((3,24));Jpos[:,:3]=R
        Jarm=np.zeros((6,24));Jarm[:,18:]=np.eye(6)
        Ed=rpy_rate_map(self.desired[3:6]);Edot=rpy_rate_map_dot(self.desired[3:6],command[3:6])
        omega=R.T@Rd@Ed@command[3:6]
        alpha=R.T@Rd@(Ed@desired_acc[3:6]+Edot@command[3:6])-np.cross(s['v'][3:6],omega)
        er=pin.log3(R.T@Rd);ep=self.desired[:3]-qbw[:3];em=self.desired[6:]-s['q'][19:]
        tasks=[(s['Jc'],-s['ce'],np.zeros(12),-s['cdotv']-100*s['ce']-20*s['Jc']@s['v']),
               (Jrot,er,omega,alpha+200*er+28*(omega-s['v'][3:6])),
               (Jpos,ep,command[:3],desired_acc[:3]+200*ep+28*(command[:3]-R@s['v'][:3])-R@np.cross(s['v'][3:6],s['v'][:3])),
               (Jarm,em,command[6:],desired_acc[6:]+100*em+20*(command[6:]-s['v'][18:]))]
        dq,vcmd,acmd,preserve=hierarchy(s['M'],tasks)
        self.q_reference=self.plant.integrate(s['q'],dq);self.v_reference=vcmd
        projection_ms=1000*(time.perf_counter()-ts);ts=time.perf_counter()
        # Printed (22): delta_a corrects the unactuated six accelerations;
        # delta_f is the absolute ground force used in the balance equation.
        # We add the same actuator/normal-force bounds as the MPC comparison.
        B=s['M'][:,:6];Jt=s['Jc'].T;b=s['M']@acmd+s['h']
        Aeq=np.column_stack((B[:6],-Jt[:6]));beq=-b[:6]
        mu=.6/np.sqrt(2)
        cone=la.block_diag(*([np.array([[1,0,-mu],[-1,0,-mu],[0,1,-mu],[0,-1,-mu],[0,0,1]])]*4))
        Atau=np.column_stack((B[6:],-Jt[6:]))
        A=np.vstack((Aeq,np.column_stack((np.zeros((20,6)),cone)),Atau))
        low=np.r_[beq,np.tile([-np.inf]*4+[10.],4),-.85*s['tau_max']-b[6:]]
        upp=np.r_[beq,np.tile([0.]*4+[800.],4),.85*s['tau_max']-b[6:]]
        try:solution,violation,dual,niter=solve_wbc_qp(A,low,upp)
        except ValueError as error_qp:
            out=pathlib.Path(__file__).resolve().parents[1]/'results'/'prioritized'
            np.savez(out/f'failure_{self.r}d.npz',A=A,l=low,u=upp,q=s['q'],v=s['v'],acmd=acmd,
                     command=command,desired_acc=desired_acc,sigma=sigma,time=t)
            raise RuntimeError(f'WBC QP failed: {error_qp}; sigma={sigma}; command_acc_max={np.max(np.abs(desired_acc))}') from error_qp
        acc=acmd.copy();acc[:6]+=solution[:6];force=solution[6:]
        torque=(s['M']@acc+s['h']-Jt@force)[6:]
        violation=float(max(0,np.max(low-A@solution),np.max(A@solution-upp)))
        qp_ms=1000*(time.perf_counter()-ts)
        self.info=dict(controller_ms=1000*(time.perf_counter()-start),distribution_ms=distribution_ms,
            projection_ms=projection_ms,qp_ms=qp_ms,setup_ms=0.,primal=violation,
            wbc_contact_acc_residual=float(np.max(np.abs(s['Jc']@acc+s['cdotv']+100*s['ce']+20*s['Jc']@s['v']))),
            wbc_float_dynamics_residual=float(np.max(np.abs((s['M']@acc+s['h']-Jt@force)[:6]))),
            hierarchy=preserve,primary_primal=violation,primary_dual=dual,
            secondary_dual=0.,sigma_min=sigma,manipulability=H,**di,qp_iterations=niter,n=18,m=len(A),qps=1)
        self.debug=dict(command=command.copy(),desired_acc=desired_acc.copy(),acmd=acmd.copy(),acc=acc.copy(),
            delta_a=solution[:6].copy(),force=force.copy(),task_velocity=ff+self.kx*error,
            contact_pre=s['Jc']@acmd+s['cdotv']+100*s['ce']+20*s['Jc']@s['v'],
            contact_post=s['Jc']@acc+s['cdotv']+100*s['ce']+20*s['Jc']@s['v'],
            A=A.copy(),low=low.copy(),upp=upp.copy())
        self.index+=1;return torque,acc
