"""Matched hierarchy experiment. Allocation is an explicitly adapted baseline.

Allocation: instantaneous prioritized differential IK with physical feasibility
in a 40 ms command interval, expressed through contact-eliminated torques. This
is NOT Xie et al.'s unmodified SNS implementation or a claim of exact replication.
MPC: ten future 40 ms nodes and strict task locks. Both use the same frozen
dynamics assembly/limits and the identical downstream geometric WBC.
"""
import time
import numpy as np
import scipy.linalg as la
import pinocchio as pin
from online_wbc_mpc import OnlineMPC,Plan
from geometric_wbc import pose,rolling_contact
from shared_hierarchy import HierarchyQP

class TaskSource:
    def __init__(self,target,mode):self.target=target;self.mode=mode;self.query_time=0.
    def at(self,t):return self.target.at(self.query_time if self.mode=='allocation' else t)

class SharedUpper(OnlineMPC):
    def __init__(self,plant,root,target,mode):
        if mode not in ('allocation','mpc'):raise ValueError(mode)
        super().__init__(plant,root,TaskSource(target,mode),N=1 if mode=='allocation' else 10,
                         dt=.04,backend='quadprog',assembly='native')
        self.mode=mode;self.hierarchy=HierarchyQP()
    def solve(self,s,t,prev_tau,initial_a,anchor=None,first_slew_time=.04):
        start=time.perf_counter();self.target.query_time=t;s=rolling_contact(s);M,J=s['M'],s['Jc'];v0=s['v'];q0=s['q']
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
        D=np.vstack(objective);d=np.concatenate(bias);H=None;g=None
        G=np.vstack(blocks);lo=np.concatenate(lower);hi=np.concatenate(upper)
        construction_ms=1000*(time.perf_counter()-start);st=time.perf_counter()
        levels=[]
        for frame in range(2):
            Cs=[];ds=[]
            for k in range(self.N):
                offset=114*k+12*frame
                if self.mode=='allocation':
                    # Differential IK: J v_cmd = Ad_(T^-1 Td) Vd - K log(Td^-1 T).
                    # Current release reference only; no future task samples.
                    Ck=D[offset+6:offset+12].copy();dk=d[offset+6:offset+12].copy()
                    name=('base','ee')[frame]
                    cur=pose(anchor[0]) if frame==0 else pin.SE3(self.plant.evaluate(anchor[0],anchor[1])['pose'])
                    target=self.target.at(t)[name+'_pose'];E=pin.SE3(target).inverse()*cur
                    dk+=np.sqrt(np.r_[np.full(3,20.),np.full(3,8.)])*10*pin.log6(E).vector
                else:
                    Ck=D[offset:offset+12];dk=d[offset:offset+12]
                Cs.append(Ck);ds.append(dk)
            levels.append((np.vstack(Cs),np.concatenate(ds)))
        ii=np.concatenate([np.arange(114*k+24,114*(k+1)) for k in range(self.N)])
        levels.append((D[ii],d[ii]))
        z,solver_info=self.hierarchy.solve(levels,G,lo,hi,np.tile(prev_tau/self.scale,self.N))
        iterations=solver_info['hierarchy_iterations'];solve_ms=1000*(time.perf_counter()-st)
        res={key:solver_info['hierarchy_kkt'] for key in ['qp_primal_relative','qp_dual_relative','qp_normal_relative']}
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
