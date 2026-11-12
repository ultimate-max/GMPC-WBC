"""Online full-body frozen-linearization MPC + continuous geometric WBC plans.

Same planner for periodic/event scheduling. N=10, dt=40 ms, 180 torque variables.
The exact instantaneous contact KKT is eliminated; contact stabilization and
frozen Coriolis velocity coupling propagate through the condensed prediction.
Task costs use first-order SE(3) log errors at each future target. This is a
single linearized QP per replan, not a converged nonlinear MPC/SQP solution.
"""
import time
import bisect
import numpy as np
import scipy.linalg as la
import pinocchio as pin
import quadprog
from geometric_wbc import pose,rolling_contact,cone_matrix,GeometricWBC
from central_relu import residuals

IDX=np.r_[np.arange(6),np.arange(18,24)]

def log_rates(x,V,a):
    E=pin.exp6(x);L=pin.Jlog6(E);eps=1e-5
    Ldot=(pin.Jlog6(E*pin.exp6(eps*V))-pin.Jlog6(E*pin.exp6(-eps*V)))/(2*eps)
    return L@V,L@a+Ldot@V


class Segment:
    """C2 quintic on one SE(3) chart times the actuated joint coordinates."""
    def __init__(self,plant,q0,v0,a0,q1,v1,a1,duration):
        self.plant=plant;self.q0=q0.copy();self.T0=pose(q0);self.dt=duration
        x=plant.difference(q0,q1);v1=v1.copy();a1=a1.copy()
        v1[:6],a1[:6]=log_rates(x[:6],v1[:6],a1[:6])
        # x=0 at the initial chart origin: xdot=V and xddot=a.
        self.c=np.zeros((6,24));self.c[1]=duration*v0;self.c[2]=.5*duration**2*a0
        rhs=np.vstack((x-self.c[1]-self.c[2],duration*v1-self.c[1]-2*self.c[2],
            duration**2*a1-2*self.c[2]))
        self.c[3:]=la.solve(np.array([[1.,1,1],[3,4,5],[6,12,20]]),rhs)
    def state(self,t,acceleration=True):
        if getattr(self.plant,'fast_reference',False):
            return self.plant.segment_state(self.q0,self.c,self.dt,t,acceleration)
        u=np.clip(t/self.dt,0,1);c=self.c
        x=sum(c[k]*u**k for k in range(6))
        dx=sum(k*c[k]*u**(k-1) for k in range(1,6))/self.dt
        ddx=sum(k*(k-1)*c[k]*u**(k-2) for k in range(2,6))/self.dt**2
        q=self.plant.integrate(self.q0,x);V=dx.copy();a=ddx.copy()
        E=pin.exp6(x[:6]);J=la.inv(pin.Jlog6(E));V[:6]=J@dx[:6]
        eps=1e-5
        Jdot=(la.inv(pin.Jlog6(pin.exp6(x[:6]+eps*dx[:6])))-
              la.inv(pin.Jlog6(pin.exp6(x[:6]-eps*dx[:6]))))/(2*eps)
        a[:6]=J@ddx[:6]+Jdot@dx[:6]
        return q,V,a


class Plan:
    def __init__(self,plant,t,dt,states,torques,forces):
        self.t=t;self.dt=dt;self.end=t+dt*len(torques)
        self.segments=[Segment(plant,*states[k],*states[k+1],dt) for k in range(len(torques))]
        self.torques=torques;self.forces=forces
    def state(self,t,acceleration=True):
        age=np.clip(t-self.t,0,self.end-self.t);k=min(int(age/self.dt),len(self.segments)-1)
        return self.segments[k].state(age-k*self.dt,acceleration)
    def feedforward(self,t):
        k=np.clip(int((t-self.t)/self.dt),0,len(self.torques)-1)
        return self.torques[k],self.forces[k]


class ReferenceMux:
    """Plans may be spliced only at release, with matched q/v/a endpoints."""
    def __init__(self,plant,s):
        self.plant=plant;self.home=(s['q'].copy(),np.zeros(24),np.zeros(24))
        self.plan=None;self.bridge=None;self.release=0.;self.blend=.04
        self.leg_times=[0.];self.leg_cache={0.:s['q'][7:19].copy()}
    def raw_state(self,t,acceleration=True):
        if self.plan is None:return tuple(x.copy() for x in self.home)
        if t<self.release+self.blend:return self.bridge.state(t-self.release,acceleration)
        return self.plan.state(t,acceleration)
    def leg_velocity(self,t,legs):
        q,v,_=self.raw_state(t,False);q[7:19]=legs
        if getattr(self.plant,'fast_reference',False):
            J,_=self.plant.contact_kinematics(q,np.zeros(24))
            return la.solve(J[:,6:18],-J[:,:6]@v[:6])
        s=self.plant.evaluate(q,np.zeros(24))
        return la.solve(s['Jc'][:,6:18],-s['Jc'][:,:6]@v[:6])
    def state(self,t):
        # Reconstruct stance legs by integrating the SAME nonlinear rolling
        # kinematics as WBC. Reusing linearized predicted leg positions across
        # many packets accumulates contact-chart error even when base/arm
        # tasks remain continuous. Cache only the currently committed curve.
        t=round(t,10);idx=bisect.bisect_right(self.leg_times,t+1e-11)-1
        cursor=self.leg_times[idx];legs=self.leg_cache[cursor].copy()
        while cursor<t-1e-10:
            h=min(.005,t-cursor)
            k1=self.leg_velocity(cursor,legs)
            k2=self.leg_velocity(cursor+h/2,legs+h*k1/2)
            k3=self.leg_velocity(cursor+h/2,legs+h*k2/2)
            k4=self.leg_velocity(cursor+h,legs+h*k3)
            legs=legs+h*(k1+2*k2+2*k3+k4)/6;cursor=round(cursor+h,10)
            if cursor not in self.leg_cache:bisect.insort(self.leg_times,cursor)
            self.leg_cache[cursor]=legs.copy()
        q,v,a=self.raw_state(t);q[7:19]=legs
        if getattr(self.plant,'fast_reference',False):
            J,_=self.plant.contact_kinematics(q,np.zeros(24))
            v[6:18]=la.solve(J[:,6:18],-J[:,:6]@v[:6])
            J,drift=self.plant.contact_kinematics(q,v)
            a[6:18]=la.solve(J[:,6:18],-drift-J[:,:6]@a[:6])
            return q,v,a
        s=self.plant.evaluate(q,np.zeros(24))
        v[6:18]=la.solve(s['Jc'][:,6:18],-s['Jc'][:,:6]@v[:6])
        s=self.plant.evaluate(q,v)
        a[6:18]=la.solve(s['Jc'][:,6:18],-s['cdotv']-s['Jc'][:,:6]@a[:6])
        return q,v,a
    def install(self,plan,t):
        old=self.state(t);goal=plan.state(t+self.blend)
        self.bridge=Segment(self.plant,*old,*goal,self.blend)
        self.plan=plan;self.release=t
        self.leg_times=[x for x in self.leg_times if x<=t+1e-10]
        self.leg_cache={x:self.leg_cache[x] for x in self.leg_times}
        now=self.state(t)
        continuity=max(np.max(np.abs(self.plant.difference(old[0],now[0]))),
            np.max(np.abs(old[1]-now[1])),np.max(np.abs(old[2]-now[2])))
        return float(continuity)
    def reference(self,t):
        q,v,a=self.state(t);s=self.plant.evaluate(q,v)
        if self.plan is None:
            f=la.lstsq(s['Jc'][:,:6].T,s['h'][:6])[0];tau=s['h'][6:]-s['Jc'][:,6:].T@f
        else:tau,f=self.plan.feedforward(t)
        return dict(q=q,v=v,a=a,base_pose=pose(q).homogeneous,base_twist=v[:6],base_acc=a[:6],
            ee_pose=s['pose'],ee_twist=s['J']@v,ee_acc=s['J']@a+s['jdotv'],tau=tau,force=f)


class FeedbackWBC(GeometricWBC):
    """Execution feedback plus the common moving joint-reference envelope."""
    def control(self,s,reference):
        # Joint error can grow in weakly observable task-space directions even
        # with small end-effector error. A common, explicit execution envelope
        # is applied to ALL scheduling policies, above soft pose tasks.
        e=s['q'][19:]-reference['q'][19:];de=s['v'][18:]-reference['v'][18:]
        center=reference['a'][18:]-16*de-64*e
        self.arm_tracking_accel_bounds=(center-64*.08,center+64*.08)
        tau,a=super().control(s,reference)
        from geometric_wbc import task_error
        residual=[];tracking=[]
        for frame,T,J,drift,kp,kd in [('base',pose(s['q']),self.Jbase,np.zeros(6),200.,28.),
            ('ee',pin.SE3(s['pose'][:3,:3],s['pose'][:3,3]),s['J'],s['jdotv'],100.,20.)]:
            Td=reference[frame+'_pose'];Td=pin.SE3(Td[:3,:3],Td[:3,3])
            e,de,Je,b=task_error(T,J@s['v'],drift,J,Td,reference[frame+'_twist'],reference[frame+'_acc'],'se3')
            r=Je@a+b+kp*e+kd*de
            residual.extend([np.linalg.norm(r[:3])/.5,np.linalg.norm(r[3:])/2.])
            tracking.extend([np.linalg.norm(e[:3])/(.002 if frame=='base' else .005),
                np.linalg.norm(e[3:])/np.deg2rad(.5 if frame=='base' else 2.)])
        f=self.debug['force'].reshape(4,3);mu=.6/np.sqrt(2)
        torque=float(np.min((.85*s['tau_max']-np.abs(tau))/(.85*s['tau_max'])))
        friction=float(np.min((mu*f[:,2,None]-np.abs(f[:,:2]))/np.maximum(mu*f[:,2,None],1.)))
        normal=float(np.min((f[:,2]-10)/np.maximum(f[:,2],1.)))
        self.feedback=dict(task_residual_score=float(max(residual)),tracking_residual_score=float(max(tracking)),torque_margin=torque,
            friction_margin=friction,normal_margin=normal,
            joint_tracking_residual_score=float(np.max(np.abs(reference['q'][7:]-s['q'][7:]))/.03),
            trust_margin=float(1-np.max(np.abs(reference['q'][7:]-s['q'][7:]))/.15),
            arm_envelope_margin=float(1-np.max(np.abs(e))/.08),
            acceleration_margin=float(np.min(1-np.abs(a[6:])/np.r_[np.full(12,50.),np.full(6,30.)])))
        return tau,a


class Schedule:
    """Causal previous-cycle WBC feedback; no lookahead at simulation outcomes."""
    def __init__(self,mode,period=.1):
        self.mode=mode;self.period=period;self.last=-np.inf;self.filtered=0.;self.consecutive=0
    def observe(self,feedback):
        self.filtered=.8*self.filtered+.2*max(feedback['task_residual_score'],feedback.get('tracking_residual_score',0.),feedback.get('joint_tracking_residual_score',0.))
        pressure=min(feedback['torque_margin'],feedback['friction_margin'],feedback['normal_margin'])<.08 or feedback.get('trust_margin',1.)<.2
        self.consecutive=self.consecutive+1 if self.filtered>1. or pressure else 0
    def due(self,t,pending=False):
        if pending:return None
        if not np.isfinite(self.last):return 'initial'
        age=t-self.last
        if self.mode=='fixed':return 'periodic' if age>=self.period-1e-9 else None
        if age>=.2-1e-9:return 'max_age'
        if age>=.05-1e-9 and self.consecutive>=3:return 'wbc_feedback'
        return None
    def mark(self,t):self.last=t;self.consecutive=0


class OnlineMPC:
    def __init__(self,plant,root,target,N=10,dt=.04,backend='quadprog',assembly='python'):
        self.plant=plant;self.target=target;self.N=N;self.dt=dt;self.n=N*18
        self.scale=np.r_[np.full(12,100.),np.full(6,10.)]
        self.model=pin.buildModelFromUrdf(str(root/'models/b2_z1.urdf'),pin.JointModelFreeFlyer())
        self.data=self.model.createData();self.cone=cone_matrix();self.last_info={}
        self.backend=backend
        if assembly not in ('python','native'):raise ValueError(assembly)
        self.assembly=assembly
        if backend not in ('quadprog','osqp','osqp_screened'):raise ValueError(backend)
        if backend!='quadprog':
            from online_osqp import OnlineOSQP
            self.engine=OnlineOSQP(screen=backend=='osqp_screened')
    def solve(self,s,t,prev_tau,initial_a,anchor=None,first_slew_time=.04):
        start=time.perf_counter();s=rolling_contact(s);M,J=s['M'],s['Jc'];v0=s['v'];q0=s['q']
        C=np.array(pin.computeCoriolisMatrix(self.model,self.data,q0,v0))
        K=np.block([[M,-J.T],[J,np.zeros((12,12))]])
        S=np.vstack((np.zeros((6,18)),np.eye(18)))
        Znormal=np.diag(np.tile([0.,0.,1.],4))
        rhs=np.column_stack((np.vstack((S,np.zeros((12,18)))),
            np.r_[-s['h'],-s['cdotv']-100*s['ce']-20*J@v0],
            np.vstack((np.zeros((24,24)),-100*Znormal@J)),np.vstack((-C,-20*J))))
        maps=la.solve(K,rhs)
        T=maps[:24,:18]*self.scale;F=maps[24:,:18]*self.scale
        a0=maps[:24,18];f0=maps[24:,18]
        Aq=maps[:24,19:43];Av=maps[:24,43:67]
        Fq=maps[24:,19:43];Fv=maps[24:,43:67]
        if anchor is None:anchor=(q0.copy(),v0.copy(),initial_a.copy())
        # This is the reference-generating MPC layer. Its initial commanded
        # q/v/a is the already committed reference at release, not a reset to
        # lagging measured joints. Measured q/v still update dynamics, contact,
        # actuator bounds and previous applied torque on EVERY online solve.
        initial_a=anchor[2]
        P=np.zeros((24,self.n));p=self.plant.difference(q0,anchor[0]);V=np.zeros_like(P);v=anchor[1].copy()
        objective=[];bias=[];blocks=[];lower=[];upper=[];nodes=[];controls=[]
        def cost(A,b,w):
            w=np.sqrt(np.broadcast_to(w,len(b)));objective.append(w[:,None]*A);bias.append(w*b)
        def bound(A,b,l,u):
            blocks.append(A);lower.append(np.broadcast_to(l,len(b))-b);upper.append(np.broadcast_to(u,len(b))-b)
        old=None;oldA=None;oldb=None;Tb=pose(q0);Te=pin.SE3(s['pose'][:3,:3],s['pose'][:3,3]);Jb=np.eye(24)[:6]
        if self.assembly=='native':
            import native
            taskG=[];taskE=[];taskV=[];targetQ=[];targetA=[];targetTau=[]
            for k in range(self.N):
                target=self.target.at(t+(k+1)*self.dt);errors=[];velocities=[]
                for name,Tcur,Jtask in [('base',Tb,Jb),('ee',Te,s['J'])]:
                    Td=target[name+'_pose'];E=pin.SE3(Td[:3,:3],Td[:3,3]).inverse()*Tcur
                    taskG.append(pin.Jlog6(E)@Jtask);errors.append(pin.log6(E).vector)
                    velocities.append(E.inverse().action@target[name+'_twist'])
                taskE.append(np.concatenate(errors));taskV.append(np.concatenate(velocities))
                targetQ.append(self.plant.difference(q0,target['q'])[6:]);targetA.append(target['a'][6:]);targetTau.append(target['tau'])
            assembled=native.assemble_online_mpc(dict(N=self.N,dt=self.dt,first=first_slew_time,
                T=T,F=F,Aq=Aq,Av=Av,Fq=Fq,Fv=Fv,cone=self.cone,a0=a0,f0=f0,p=p,v=v,v0=v0,
                scale=self.scale,qjoint=q0[7:],qmin=s['q_min'],qmax=s['q_max'],vmax=s['v_max'],taumax=s['tau_max'],
                prev=prev_tau,initialA=initial_a,taskG=np.vstack(taskG),taskJ=np.vstack((Jb,s['J'])),
                taskE=np.array(taskE),taskV=np.array(taskV),targetQ=np.array(targetQ),targetA=np.array(targetA),targetTau=np.array(targetTau)))
            objective.append(assembled['D']);bias.append(assembled['d']);blocks.append(assembled['G'])
            lower.append(assembled['lo']);upper.append(assembled['hi'])
        for k in range(self.N if self.assembly=='python' else 0):
            U=np.zeros((18,self.n));U[:,18*k:18*(k+1)]=np.eye(18)
            A=T@U+Aq@P+Av@V;b=a0+Aq@p+Av@(v-v0)
            B=F@U+Fq@P+Fv@V;f=f0+Fq@p+Fv@(v-v0)
            tm=self.scale[:,None]*U
            bound(tm,np.zeros(18),-.85*s['tau_max'],.85*s['tau_max'])
            bound(self.cone@B,self.cone@f,np.tile([-np.inf]*4+[10.],4),np.tile([0.]*4+[800.],4))
            # The first planned input starts at release, not at the current
            # 5 ms WBC tick. WBC still enforces its own 5 ms applied-torque
            # slew limit. Here use the actual declared time until release.
            step=first_slew_time if k==0 else self.dt
            bound(tm if old is None else tm-old,-prev_tau if old is None else np.zeros(18),
                -step*np.r_[np.full(12,1000.),np.full(6,100.)],step*np.r_[np.full(12,1000.),np.full(6,100.)])
            bound(A[6:],b[6:],-np.r_[np.full(12,50.),np.full(6,30.)],np.r_[np.full(12,50.),np.full(6,30.)])
            P=P+self.dt*V+.5*self.dt**2*A;p=p+self.dt*v+.5*self.dt**2*b
            V=V+self.dt*A;v=v+self.dt*b
            bound(P[6:],q0[7:]+p[6:],s['q_min']+.02,s['q_max']-.02)
            # Local pose/Jacobian approximation must not exploit distant joint
            # configurations. The same trust region is used for every policy.
            bound(P[6:],p[6:],-.15,.15)
            bound(V[6:],v[6:],-np.minimum(s['v_max'],4.),np.minimum(s['v_max'],4.))
            bound(P[:6],p[:6],np.r_[np.full(3,-.10),np.full(3,-.30)],np.r_[np.full(3,.10),np.full(3,.30)])
            target=self.target.at(t+(k+1)*self.dt)
            mult=3. if k==self.N-1 else 1.
            for name,Tcur,Jtask,wp,wr in [('base',Tb,Jb,4000.,1600.),('ee',Te,s['J'],4000.,1000.)]:
                Td=target[name+'_pose'];E=pin.SE3(Td[:3,:3],Td[:3,3]).inverse()*Tcur
                L=pin.Jlog6(E);e=pin.log6(E).vector;G=L@Jtask
                cost(G@P,e+G@p,mult*np.r_[np.full(3,wp),np.full(3,wr)])
                # Reference velocity is transported to the current body axes.
                vd=E.inverse().action@target[name+'_twist']
                cost(Jtask@V,Jtask@v-vd,np.r_[np.full(3,20.),np.full(3,8.)])
            desired=self.plant.difference(q0,target['q'])
            cost(P[6:],p[6:]-desired[6:],np.r_[np.full(12,5.),np.full(6,40.)])
            cost(A[6:],b[6:]-target['a'][6:],.02)
            cost(A[6:] if oldA is None else A[6:]-oldA[6:],
                b[6:]-initial_a[6:] if oldb is None else b[6:]-oldb[6:],.02)
            cost(U,-target['tau']/self.scale,.002)
            cost(U if old is None else (tm-old)/self.scale[:,None],
                -prev_tau/self.scale if old is None else np.zeros(18),.03)
            nodes.append((P.copy(),p.copy(),V.copy(),v.copy(),A.copy(),b.copy(),B.copy(),f.copy()))
            controls.append(U);old=tm;oldA=A;oldb=b
        D=np.vstack(objective);d=np.concatenate(bias);H=2*D.T@D+1e-9*np.eye(self.n);g=2*D.T@d
        G=np.vstack(blocks);lo=np.concatenate(lower);hi=np.concatenate(upper)
        scale=max(1.,np.max(np.abs(H)));H/=scale;g/=scale
        if self.backend=='quadprog':
            il=np.flatnonzero(np.isfinite(lo));iu=np.flatnonzero(np.isfinite(hi))
            Cq=np.vstack((G[il],-G[iu])).T;bd=np.r_[lo[il],-hi[iu]]
            rows=1/np.maximum(la.norm(Cq,axis=0),1e-12)
        construction_ms=1000*(time.perf_counter()-start);st=time.perf_counter()
        solver_info={}
        try:
            if self.backend=='quadprog':
                sol=quadprog.solve_qp(H,-g,Cq*rows,bd*rows)
                z=sol[0];du=sol[4]*rows;y=np.zeros(len(lo));iterations=int(sol[3][0])
                np.add.at(y,il,-du[:len(il)]);np.add.at(y,iu,du[len(il):])
            else:
                z,y,iterations,solver_info=self.engine.solve(H,g,G,lo,hi,t,self.dt,self.N)
        except ValueError:
            np.savez('/tmp/online_mpc_failed_qp.npz',H=H,g=g,G=G,lo=lo,hi=hi,q=q0,v=v0,
                anchor_q=anchor[0],anchor_v=anchor[1],anchor_a=anchor[2],time=t)
            raise
        solve_ms=1000*(time.perf_counter()-st);res=residuals(H,g,G,lo,hi,z,y)
        if max(res[k] for k in ['qp_primal_relative','qp_dual_relative','qp_normal_relative'])>2e-7:
            raise RuntimeError(f'MPC KKT check: {res}')
        states=[tuple(x.copy() for x in anchor)];torques=[];forces=[]
        if self.assembly=='native':
            pp=self.plant.difference(q0,anchor[0]);vv=anchor[1].copy()
            for k in range(self.N):
                u=z[18*k:18*(k+1)]
                aa=T@u+a0+Aq@pp+Av@(vv-v0);ff=F@u+f0+Fq@pp+Fv@(vv-v0)
                pp=pp+self.dt*vv+.5*self.dt**2*aa;vv=vv+self.dt*aa
                states.append((self.plant.integrate(q0,pp),vv.copy(),aa));torques.append(self.scale*u);forces.append(ff)
        for k,(P,p,V,v,A,b,B,f) in enumerate(nodes):
            states.append((self.plant.integrate(q0,P@z+p),V@z+v,A@z+b))
            torques.append(self.scale*(controls[k]@z));forces.append(B@z+f)
        plan=Plan(self.plant,t,self.dt,states,np.array(torques),np.array(forces))
        nonlinear=[];contact_defect=[]
        for k,(q,v,a) in enumerate(states[1:]):
            ns=rolling_contact(self.plant.evaluate(q,v))
            nonlinear.append(ns['M']@a+ns['h']-np.r_[np.zeros(6),torques[k]]-ns['Jc'].T@forces[k])
            contact_defect.append(ns['Jc']@a+ns['cdotv']+100*ns['ce']+20*ns['Jc']@v)
        self.last_info=dict(planner_ms=1000*(time.perf_counter()-start),construction_ms=construction_ms,
            solve_ms=solve_ms,iterations=iterations,backend=self.backend,assembly=self.assembly,**solver_info,
            qp_residual=max(res[k] for k in ['qp_primal_relative','qp_dual_relative','qp_normal_relative']),
            physical_bound_violation=float(max(0,np.max(lo-G@z),np.max(G@z-hi))),variables=self.n,
            nonlinear_dynamics_defect_max=float(np.max(np.abs(nonlinear))),
            nonlinear_contact_defect_max=float(np.max(np.abs(contact_defect))))
        self.debug=dict(H=H,g=g,G=G,lo=lo,hi=hi,z=z,states=states)
        return plan
