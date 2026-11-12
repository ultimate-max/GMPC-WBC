"""Two local inverse-dynamics WBCs coordinated at a rigid mounting interface.

Single-thread structured hierarchical QP, not two independently solved torque
vectors pasted together. Local WBCs own their dynamics, constraints and tasks.
The coordinator eliminates six interface-wrench variables with acceleration
consensus, then solves the globally ordered task levels in 18 torque variables.

lambda: force ON BODY, at z1_link00 in LOCAL [force; moment] axes. The arm sees
-lambda. Physical inertias are partitioned, never counted in both models.
"""
from dataclasses import dataclass
import time
import numpy as np
import scipy.linalg as la
import pinocchio as pin
from model import contact
from body_arm_interface import BodyArmInterface,BODY,ARM
from anchored_geometry import mount_transform,error_dynamics
from geometric_wbc import GeometricWBC,pose,rolling_contact,task_error

def inf(x):return float(np.max(np.abs(x))) if np.size(x) else 0.

@dataclass
class LocalModel:
    M: np.ndarray
    h: np.ndarray
    Ji: np.ndarray
    Jc: np.ndarray
    T: np.ndarray
    W: np.ndarray
    a0: np.ndarray
    F: np.ndarray=None
    FW: np.ndarray=None
    f0: np.ndarray=None

class LocalWBC:
    def __init__(self,model,data,idx,torques,Ji):
        self.model=model;self.data=data;self.idx=idx;self.torques=torques;self.Ji=Ji
        self.nt=len(torques);self.last=None
    def inertias(self,s):
        M=np.array(pin.crba(self.model,self.data,s['q']))
        h=np.array(pin.nonLinearEffects(self.model,self.data,s['q'],s['v']))
        return M[np.ix_(self.idx,self.idx)],h[self.idx]
    def joint_bounds(self,s,A,b,E,prev,dt):
        # A,b map common normalized torque to this subsystem acceleration.
        ix=self.torques;q=s['q'][7:][ix];v=s['v'][6:][ix]
        rate=np.full(self.nt,1000. if self.nt==12 else 100.)
        accel=np.full(self.nt,50. if self.nt==12 else 30.)
        C=np.vstack((E,E,dt*A[6:],.5*dt*dt*A[6:],A[6:]))
        lo=np.r_[-.85*s['tau_max'][ix],prev[ix]-dt*rate,
            -np.minimum(s['v_max'][ix],4.)-v-dt*b[6:],
            s['q_min'][ix]+.02-q-dt*v-.5*dt*dt*b[6:],-accel-b[6:]]
        hi=np.r_[.85*s['tau_max'][ix],prev[ix]+dt*rate,
            np.minimum(s['v_max'][ix],4.)-v-dt*b[6:],
            s['q_max'][ix]-.02-q-dt*v-.5*dt*dt*b[6:],accel-b[6:]]
        return C,lo,hi
    def smooth(self,A,b,U,target_a,prev_scaled,ref_scaled,weight):
        return [(.1*A[6:],.1*(b[6:]-target_a[self.idx[6:]])),
                (np.sqrt(weight)*U,-np.sqrt(weight)*prev_scaled[self.torques]),
                (.2*U,-.2*ref_scaled[self.torques])]

class BodyWBC(LocalWBC):
    def dynamics(self,s):
        M,h=self.inertias(s);J=s['Jc'][:,BODY]
        K=np.block([[M,-J.T],[J,np.zeros((12,12))]])
        S=np.vstack((np.zeros((6,12)),np.eye(12)))
        rhs=np.column_stack((np.vstack((S,np.zeros((12,12)))),
                             np.vstack((self.Ji.T,np.zeros((12,6)))),
                             np.r_[-h,-s['cdotv']-100*s['ce']-20*s['Jc']@s['v']]))
        st=time.perf_counter();z=la.solve(K,rhs);self.factor_ms=1000*(time.perf_counter()-st)
        self.last=LocalModel(M,h,self.Ji,J,z[:18,:12],z[:18,12:18],z[:18,18],
                             z[18:,:12],z[18:,12:18],z[18:,18])
        return self.last
    def task(self,s,r,A,b):
        J=np.eye(24)[:6]
        e,de,Je,be=task_error(pose(s['q']),J@s['v'],np.zeros(6),J,
            pin.SE3(r['base_pose']),r['base_twist'],r['base_acc'],'se3')
        desired=-be-200*e-28*de;w=np.r_[np.full(3,20.),np.full(3,5.)]
        return w[:,None]*(Je[:,BODY]@A),w*(Je[:,BODY]@b-desired),e

class ArmWBC(LocalWBC):
    def dynamics(self,s):
        M,h=self.inertias(s);S=np.vstack((np.zeros((6,6)),np.eye(6)))
        st=time.perf_counter();z=la.solve(M,np.column_stack((S,-self.Ji.T,-h)),assume_a='pos')
        self.factor_ms=1000*(time.perf_counter()-st)
        self.last=LocalModel(M,h,self.Ji,np.zeros((0,12)),z[:,:6],z[:,6:12],z[:,12])
        return self.last
    def task(self,s,r,A,b,C):
        e,de,Je,be=error_dynamics(s,C,pin.SE3(r['ee_pose']),r['ee_twist'],r['ee_acc'])
        desired=-be-100*e-20*de;w=np.r_[np.full(3,20.),np.full(3,5.)]
        return w[:,None]*(Je[:,ARM]@A),w*(Je[:,ARM]@b-desired),e

class CoordinatedWBC(GeometricWBC):
    """mode=dual uses local dynamics; unified is an independent full-model reference.

    Both modes share all task definitions, hard bounds, smoothing weights,
    solver, tolerances, interpolation and actuator execution.
    """
    def __init__(self,plant,root,mode='dual',dt=.005,torque_smoothing=1.,joint_damping=0.):
        if mode not in ('dual','unified'):raise ValueError(mode)
        super().__init__(plant,'se3',dt=dt)
        self.execution_mode=mode;self.torque_smoothing=float(torque_smoothing)
        self.joint_damping=float(joint_damping)
        if not np.isfinite(joint_damping) or joint_damping<0:raise ValueError('Invalid joint damping')
        if not np.isfinite(torque_smoothing) or torque_smoothing<0:raise ValueError('Invalid smoothing weight')
        self.partition=BodyArmInterface(root);self.C=mount_transform(self.partition.full,plant.truth()['q'])
        X=self.C.inverse().action
        self.body=BodyWBC(self.partition.body,self.partition.data['body'],BODY,np.arange(12),np.c_[X,np.zeros((6,12))])
        self.arm=ArmWBC(self.partition.arm,self.partition.data['arm'],ARM,np.arange(12,18),np.c_[X,np.zeros((6,6))])
        self.Ub=np.eye(18)[:12];self.Ua=np.eye(18)[12:]
        self.last_problem=None
    def maps(self,s):
        st=time.perf_counter();times={}
        if self.execution_mode=='unified':
            T,a0,F,f0=contact(s)
            times=dict(body_model_ms=0.,arm_model_ms=0.,interface_condense_ms=0.,dynamics_factor_ms=1000*(time.perf_counter()-st))
            return dict(Ab=T[BODY]*self.scale,bb=a0[BODY],Aa=T[ARM]*self.scale,ba=a0[ARM],
                        B=F*self.scale,f0=f0,T=T,a0=a0,F=F,**times)
        st=time.perf_counter();b=self.body.dynamics(s);times['body_model_ms']=1000*(time.perf_counter()-st)
        st=time.perf_counter();a=self.arm.dynamics(s);times['arm_model_ms']=1000*(time.perf_counter()-st)
        st=time.perf_counter()
        D=b.Ji@b.W-a.Ji@a.W
        rhs=np.column_stack((a.Ji@a.T@self.Ua-b.Ji@b.T@self.Ub,a.Ji@a.a0-b.Ji@b.a0))
        wm=la.solve(D,rhs);W=wm[:,:18];w0=wm[:,18]
        Tb=b.T@self.Ub+b.W@W;bb=b.a0+b.W@w0
        Ta=a.T@self.Ua+a.W@W;ba=a.a0+a.W@w0
        F=b.F@self.Ub+b.FW@W;f0=b.f0+b.FW@w0
        T=np.vstack((Tb,Ta[6:]));a0=np.r_[bb,ba[6:]]
        times['interface_condense_ms']=1000*(time.perf_counter()-st)
        times['dynamics_factor_ms']=self.body.factor_ms+self.arm.factor_ms+times['interface_condense_ms']
        return dict(Ab=Tb*self.scale,bb=bb,Aa=Ta*self.scale,ba=ba,B=F*self.scale,f0=f0,
                    W=W,w0=w0,T=T,a0=a0,F=F,**times)
    def assemble(self,s,r):
        start=time.perf_counter();maps=self.maps(s)
        Ab,bb,Aa,ba,B,f0=[maps[k] for k in ('Ab','bb','Aa','ba','B','f0')]
        body_bounds=self.body.joint_bounds(s,Ab,bb,self.Ub*self.scale,self.prev_tau,self.dt)
        arm_bounds=self.arm.joint_bounds(s,Aa,ba,self.Ua*self.scale,self.prev_tau,self.dt)
        # Both local WBCs send their bounds to the interface coordinator.
        G=np.vstack((body_bounds[0],arm_bounds[0],self.cone@B))
        lo=np.r_[body_bounds[1],arm_bounds[1],np.tile([-np.inf]*4+[10.],4)-self.cone@f0]
        hi=np.r_[body_bounds[2],arm_bounds[2],np.tile([0.]*4+[800.],4)-self.cone@f0]
        if self.joint_damping:
            # Linear velocity damper at the NEXT constant-acceleration state:
            # -eta(q_next-q_min) <= v_next <= eta(q_max-q_next).
            # It starts braking before a one-step position bound becomes
            # incompatible with the actuator acceleration limit.
            eta=self.joint_damping;h=self.dt
            Aj=np.vstack((Ab[6:],Aa[6:]));bj=np.r_[bb[6:],ba[6:]]
            q=s['q'][7:];v=s['v'][6:];coefficient=h+.5*eta*h*h
            G=np.vstack((G,coefficient*Aj))
            lo=np.r_[lo,-eta*(q-s['q_min']-.02)-(1+eta*h)*v-coefficient*bj]
            hi=np.r_[hi, eta*(s['q_max']-.02-q)-(1+eta*h)*v-coefficient*bj]
        ee=self.arm.task(s,r,Aa,ba,self.C);base=self.body.task(s,r,Ab,bb)
        target_a=r['a']+100*self.plant.difference(s['q'],r['q'])+20*(r['v']-s['v'])
        sb=self.body.smooth(Ab,bb,self.Ub,target_a,self.prev_tau/self.scale,r['tau']/self.scale,self.torque_smoothing)
        sa=self.arm.smooth(Aa,ba,self.Ua,target_a,self.prev_tau/self.scale,r['tau']/self.scale,self.torque_smoothing)
        smooth=[sb[0],sa[0],(.1*B,.1*(f0-r['force'])),sb[1],sa[1],sb[2],sa[2]]
        tasks=[ee[:2],base[:2],(np.vstack([x[0] for x in smooth]),np.concatenate([x[1] for x in smooth]))]
        return dict(maps=maps,G=G,lo=lo,hi=hi,tasks=tasks,
                    construction_ms=1000*(time.perf_counter()-start))
    def solve_level(self,i,C,d,G,lo,hi,warm):
        # The final residual problem is often unconstrained. Solve its
        # original least-squares objective directly to avoid an artificial
        # diagonal regularizer dominating a nearly-zero reduced objective.
        if i==2:
            from central_relu import residuals
            x=la.lstsq(C,-d,cond=1e-12)[0];H=2*C.T@C;g=2*C.T@d
            scale=max(1.,float(np.max(np.abs(H))))
            rr=residuals(H/scale,g/scale,G,lo,hi,x,np.zeros(len(lo)))
            if max(rr[k] for k in ('qp_primal_relative','qp_dual_relative','qp_normal_relative'))<1e-10:
                self.active_by_level[i]=np.array([],int)
                return x,rr,0
        return super().solve_level(i,C,d,G,lo,hi,warm)
    def solve_tasks(self,problem):
        G,lo,hi=[problem[k] for k in ('G','lo','hi')]
        x=np.zeros(18);Z=np.eye(18);locks=[];metrics=[];counts=[];dimensions=[]
        for i,(C,d) in enumerate(problem['tasks']):
            if Z.shape[1]:
                y,res,n=self.solve_level(i,C@Z,C@x+d,G@Z,lo-G@x,hi-G@x,
                    self.prev_tau/self.scale if i==0 else np.zeros(Z.shape[1]))
                x+=Z@y
            else:
                res=dict(qp_primal_relative=0.,qp_dual_relative=0.,qp_normal_relative=0.);n=0
                self.active_by_level[i]=np.array([],int)
            metrics.append(res);counts.append(n);locks.append((C,C@x))
            if i<2:
                face=np.vstack((C@Z,(G@Z)[self.active_by_level[i]]))
                norm=la.norm(face,axis=1);face=face[norm>1e-10]/norm[norm>1e-10,None]
                Z=Z@la.null_space(face,rcond=1e-10)
            dimensions.append(Z.shape[1])
        lock=max(inf(C@x-value) for C,value in locks)
        bound=max(0.,float(np.max(lo-G@x)),float(np.max(G@x-hi)))
        return x,dict(task_lock_max=lock,constraint_violation=bound,qp_iterations=sum(counts),
            qp_primal_relative=max(r['qp_primal_relative'] for r in metrics),
            qp_dual_relative=max(r['qp_dual_relative'] for r in metrics),
            qp_normal_relative=max(r['qp_normal_relative'] for r in metrics),
            first_remaining=dimensions[0],second_remaining=dimensions[1])
    def validate(self,s,tau,acc,force,maps):
        dyn=s['M']@acc+s['h']-np.r_[np.zeros(6),tau]-s['Jc'].T@force
        con=s['Jc']@acc+s['cdotv']+100*s['ce']+20*s['Jc']@s['v']
        result=dict(dynamics_residual=inf(dyn),contact_residual=inf(con),
                    interface_checked=int(self.execution_mode=='dual'),interface_acc_residual=0.,
                    body_dynamics_residual=0.,arm_dynamics_residual=0.,joint_damping_violation=0.)
        if self.joint_damping:
            qnext=s['q'][7:]+self.dt*s['v'][6:]+.5*self.dt**2*acc[6:]
            vnext=s['v'][6:]+self.dt*acc[6:]
            violation=max(0.,float(np.max(vnext-self.joint_damping*(s['q_max']-.02-qnext))),
                float(np.max(-vnext-self.joint_damping*(qnext-s['q_min']-.02))))
            result['joint_damping_violation']=violation
            if violation>2e-5:raise RuntimeError(f'Physical joint damping check: {violation}')
        if self.execution_mode=='dual':
            b=self.body.last;a=self.arm.last;w=maps['W']@tau+maps['w0']
            ab=maps['Ab']@(tau/self.scale)+maps['bb'];aa=maps['Aa']@(tau/self.scale)+maps['ba']
            db=b.M@ab+b.h-np.r_[np.zeros(6),tau[:12]]-b.Jc.T@force-b.Ji.T@w
            da=a.M@aa+a.h-np.r_[np.zeros(6),tau[12:]]+a.Ji.T@w
            result.update(interface_acc_residual=inf(b.Ji@ab-a.Ji@aa),
                body_dynamics_residual=inf(db),arm_dynamics_residual=inf(da))
            self.interface=dict(wrench=w.copy(),arm_wrench=-w.copy(),body_acc=b.Ji@ab,arm_acc=a.Ji@aa)
        if max(result[k] for k in ('dynamics_residual','contact_residual','interface_acc_residual',
                                  'body_dynamics_residual','arm_dynamics_residual'))>1e-7:
            raise RuntimeError(f'Dynamics/interface certificate failed: {result}')
        return result
    def control(self,s,reference):
        start=time.perf_counter();self.fallbacks=0;s=rolling_contact(s)
        problem=self.assemble(s,reference);self.last_problem=problem
        stamp=time.perf_counter();x,info=self.solve_tasks(problem);solver_ms=1000*(time.perf_counter()-stamp)
        maps=problem['maps'];tau=self.scale*x;acc=maps['T']@tau+maps['a0'];force=maps['F']@tau+maps['f0']
        stamp=time.perf_counter();cert=self.validate(s,tau,acc,force,maps)
        if info['task_lock_max']>2e-7 or info['constraint_violation']>2e-5:
            raise RuntimeError(f'WBC task/bound certificate failed: {info}')
        validation_ms=1000*(time.perf_counter()-stamp);self.prev_tau=tau.copy()
        self.info=dict(controller_ms=1000*(time.perf_counter()-start),model_ms=problem['construction_ms'],
            wbc_construction_ms=problem['construction_ms'],wbc_solve_ms=solver_ms,wbc_validation_ms=validation_ms,
            **{k:maps[k] for k in ('body_model_ms','arm_model_ms','interface_condense_ms','dynamics_factor_ms')},
            **info,**cert)
        self.debug=dict(tau=tau.copy(),acc=acc.copy(),force=force.copy(),solution=x.copy())
        return tau,acc
