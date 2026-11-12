"""Predictive base/legs + arm GMPC with explicit internal wrench consensus.

Models are re-evaluated each feedback cycle, then frozen over the horizon.
SE(3) errors are linearized in a shared measured tangent frame. This is a
linearized geometric MPC, not a multi-iteration nonlinear SQP implementation.
"""
import time,pathlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import pinocchio as pin
import osqp
from controller import Costs,Constraints
from model import se3,log,contact,equilibrium


class Parts:
    def __init__(self,root):
        self.full=pin.buildModelFromUrdf(str(pathlib.Path(root)/'models/b2_z1.urdf'),pin.JointModelFreeFlyer())
        self.indices=[np.arange(18),np.r_[np.arange(6),np.arange(18,24)]]
        self.models=[];self.data=[]
        for part in range(2):
            m=pin.Model(self.full)
            for j in range(1,m.njoints):
                if j==1:
                    I=self.full.inertias[j];m.inertias[j]=pin.Inertia(.5*I.mass,I.lever,.5*I.inertia)
                elif (m.joints[j].idx_v>=18)!=(part==1):m.inertias[j]=pin.Inertia.Zero()
            self.models.append(m);self.data.append(m.createData())

    def evaluate(self,q,v):
        out=[]
        for m,d,idx in zip(self.models,self.data,self.indices):
            M=np.array(pin.crba(m,d,q))
            h=np.array(pin.nonLinearEffects(m,d,q,v))
            out.append((M[np.ix_(idx,idx)].copy(),h[idx].copy()))
        return out


class QPSolver:
    def __init__(self,H,g,A,l,u,warm=None,tol=2e-7):
        self.rows=1/np.maximum(1.,np.linalg.norm(A,axis=1))
        self.solver=osqp.OSQP()
        self.solver.setup(P=sp.csc_matrix(np.triu((H+H.T)/2)),q=g,
            A=sp.csc_matrix(A*self.rows[:,None]),l=l*self.rows,u=u*self.rows,
            eps_abs=tol,eps_rel=tol,max_iter=12000,polishing=False,
            warm_starting=True,verbose=False,check_termination=10)
        if warm is not None:self.solver.warm_start(x=warm)
    def solve(self,g=None):
        if g is not None:self.solver.update(q=g)
        r=self.solver.solve()
        if r.info.status_val not in (1,2):raise RuntimeError('Local QP: '+r.info.status)
        self.dual=self.rows*r.y
        return r.x,r.info


class SplitGMPC:
    def __init__(self,plant,ref,root,mode='distributed',N=10,dt=.03,iterations=100,rho=30.,parallel=False,
                 backend='osqp',qp_tol=2e-7,qp_max_iter=3000):
        if N<1 or dt<.02 or iterations<1 or rho<=0 or mode not in ('central','distributed'):
            raise ValueError('Invalid GMPC configuration')
        self.plant=plant;self.ref=ref;self.parts=Parts(root);self.mode=mode
        self.N=N;self.dt=dt;self.iterations=iterations;self.rho=rho
        s=plant.truth();self.home=s['q'].copy();self.pose_home=se3(s['pose'])
        self.base_home=pin.SE3(pin.Quaternion(s['q'][6],*s['q'][3:6]).matrix(),s['q'][:3])
        self.prev_tau=equilibrium(s);self.last=None;self.dual=None
        self.previous_rotation=None;self.warm_rotation=np.eye(3)
        self.pool=ThreadPoolExecutor(max_workers=2) if parallel else None
        self.parallel=parallel
        if backend not in ('osqp','reluqp') or (backend=='reluqp' and mode!='central'):
            raise ValueError('ReLU-QP is implemented for the centralized comparison only')
        self.backend=backend;self.qp_tol=qp_tol;self.qp_max_iter=qp_max_iter

    def close(self):
        if self.pool:self.pool.shutdown()

    def build_part(self,s,t,part,M,h):
        idx=self.parts.indices[part];nv=len(idx);nj=nv-6;nf=12 if part==0 else 0
        nu=nj+nf+6;n=self.N*nu;dt=self.dt
        scale=np.r_[np.full(nj,100. if part==0 else 10.),np.full(nf,100.),np.full(3,200.),np.full(3,50.)]
        S=np.vstack((np.zeros((6,nj)),np.eye(nj)))
        Jc=s['Jc'][:,idx] if part==0 else np.zeros((0,nv))
        force_map=np.column_stack((S,Jc.T,np.vstack((np.eye(6),np.zeros((nj,6))))))
        G=la.solve(M,force_map*scale[None,:],assume_a='pos');a0=-la.solve(M,h,assume_a='pos')
        # Static reference exactly respects the same mass split.
        f=la.lstsq(s['Jc'][:,:6].T,s['h'][:6])[0]
        wrench=h[:6]-(Jc.T@f)[:6] if part==0 else h[:6]
        tau_ref=h[6:]-(Jc.T@f)[6:] if part==0 else h[6:]
        uref=np.r_[tau_ref,f if part==0 else [],wrench]/scale
        cons=Constraints();cost=Costs(n)
        qmap=np.zeros((nv,n));qoff=np.zeros(nv);vmap=np.zeros((nv,n));voff=s['v'][idx].copy()
        base=pin.SE3(pin.Quaternion(s['q'][6],*s['q'][3:6]).matrix(),s['q'][:3])
        Eb=self.base_home.inverse()*base;eb=pin.log6(Eb).vector;Db=pin.Jlog6(Eb)
        home_offset=self.plant.difference(self.home,s['q'])[idx]
        qnow=s['q'][7:19] if part==0 else s['q'][19:]
        limits=slice(0,12) if part==0 else slice(12,18)
        physical=[];controls=[];interfaces=[];interface_offsets=[];accels=[];taskmaps=[]
        mu=.6/np.sqrt(2)
        cone=la.block_diag(*([np.array([[1,0,-mu],[-1,0,-mu],[0,1,-mu],[0,-1,-mu],[0,0,1]])]*4))
        oldU=None
        for k in range(self.N):
            U=np.zeros((nu,n));U[:,k*nu:(k+1)*nu]=np.eye(nu)
            A=G@U;b=a0
            if part==0:
                cons.add(Jc@A+100*Jc@qmap+20*Jc@vmap,
                    Jc@b+s['cdotv']+100*(s['ce']+Jc@qoff)+20*Jc@voff,0.,0.)
                fm=scale[nj:nj+nf,None]*U[nj:nj+nf]
                cons.add(cone@fm,np.zeros(20),np.tile([-np.inf]*4+[10.],4),np.tile([0.]*4+[800.],4))
            tm=scale[:nj,None]*U[:nj]
            cons.add(tm,np.zeros(nj),-.85*s['tau_max'][limits],.85*s['tau_max'][limits])
            slew=np.full(nj,1000. if part==0 else 100.)
            if k==0:cons.add(tm,-self.prev_tau[limits],-.02*slew,.02*slew)
            else:cons.add(tm-scale[:nj,None]*oldU[:nj],np.zeros(nj),-dt*slew,dt*slew)
            vmap=vmap+dt*A;voff=voff+dt*b
            qmap=qmap+dt*vmap;qoff=qoff+dt*voff
            cons.add(qmap[6:],qoff[6:]+qnow,s['q_min'][limits]+.02,s['q_max'][limits]-.02)
            cons.add(vmap[6:],voff[6:],-np.minimum(s['v_max'][limits],4.),np.minimum(s['v_max'][limits],4.))
            cons.add(Db@qmap[:6],eb+Db@qoff[:6],[-.15,-.15,-.08,-.25,-.25,-.25],[.15,.15,.08,.25,.25,.25])
            # Geometric base posture is counted once in the sum of local costs.
            if part==0:
                cost.add(Db@qmap[:6],eb+Db@qoff[:6],np.r_[np.full(3,2000.),np.full(3,1000.)])
                cost.add(vmap[:6],voff[:6],np.full(6,20.))
            else:
                desired=se3(self.ref.pose_at(t+(k+1)*dt));actual=se3(s['pose'])
                E=desired.inverse()*actual;J=s['J'][:,idx];D=pin.Jlog6(E)@J
                target_velocity=(actual.inverse()*desired).action@self.ref.twist(t+(k+1)*dt)
                mult=3. if k==self.N-1 else 1.
                cost.add(D@qmap,pin.log6(E).vector+D@qoff,mult*np.r_[np.full(3,8000.),np.full(3,2000.)])
                cost.add(J@vmap,J@voff-target_velocity,mult*np.r_[np.full(3,100.),np.full(3,30.)])
                taskmaps.append((D@qmap,pin.log6(E).vector+D@qoff))
            cost.add(qmap[6:],qoff[6:]+home_offset[6:],.3 if part==0 else .02)
            cost.add(vmap[6:],voff[6:],.2 if part==0 else .05)
            cost.add(U,-uref,np.r_[np.full(nj,.002),np.full(nf,.0002),np.full(6,.0001)])
            if oldU is None:
                cost.add(U[:nj],-self.prev_tau[limits]/scale[:nj],.2)
            else:cost.add((U-oldU)[:nj],np.zeros(nj),.2)
            # Both subsystems use the same measured body tangent frame.
            vs=np.r_[np.full(3,.2),np.full(3,.5)]
            interfaces.append(np.vstack((vmap[:6]/vs[:,None],(1 if part==0 else -1)*U[-6:])))
            interface_offsets.append(np.r_[voff[:6]/vs,np.zeros(6)])
            physical.append((qmap.copy(),qoff.copy(),vmap.copy(),voff.copy()))
            controls.append(U);accels.append((A,b));oldU=U
        H,g,_,_=cost.arrays();C,l,u=cons.arrays()
        H+=1e-9*np.eye(n)
        return dict(H=H,g=g,A=C,l=l,u=u,E=np.vstack(interfaces),d=np.concatenate(interface_offsets),
            physical=physical,controls=controls,accels=accels,scale=scale,nu=nu,nj=nj,nf=nf,
            M=M,h=h,Jc=Jc,taskmaps=taskmaps,uref=uref)

    def build(self,s,t):
        rotation=pin.Quaternion(s['q'][6],*s['q'][3:6]).matrix()
        self.warm_rotation=np.eye(3) if self.previous_rotation is None else rotation.T@self.previous_rotation
        self.current_rotation=rotation
        models=self.parts.evaluate(s['q'],s['v'])
        return [self.build_part(s,t,i,*models[i]) for i in range(2)]

    def warm(self,parts):
        guesses=[]
        for i,p in enumerate(parts):
            if self.last is None:guesses.append(np.tile(p['uref'],self.N));continue
            a=self.last[i].reshape(self.N,p['nu']).copy();alpha=.02/self.dt
            a[:,-6:-3]=a[:,-6:-3]@self.warm_rotation.T
            a[:,-3:]=a[:,-3:]@self.warm_rotation.T
            a=(1-alpha)*a+alpha*np.vstack((a[1:],a[-1:]));guesses.append(a.ravel())
        return guesses

    def solve(self,parts):
        guesses=self.warm(parts);setup=time.perf_counter();p0,p1=parts
        qp_metrics={}
        if self.mode=='central':
            from central_relu import CentralReLU,residuals
            H=la.block_diag(p0['H'],p1['H']);g=np.r_[p0['g'],p1['g']]
            A=la.block_diag(p0['A'],p1['A']);E=np.column_stack((p0['E'],-p1['E']));b=p1['d']-p0['d']
            A=np.vstack((A,E));l=np.r_[p0['l'],p1['l'],b];u=np.r_[p0['u'],p1['u'],b]
            if self.backend=='reluqp':
                solver=CentralReLU(H,g,A,l,u,np.r_[*guesses],tol=self.qp_tol,max_iter=self.qp_max_iter)
            else:solver=QPSolver(H,g,A,l,u,np.r_[*guesses],tol=self.qp_tol)
            setup_ms=1000*(time.perf_counter()-setup);ts=time.perf_counter();z,info=solver.solve()
            qp_metrics=residuals(H,g,A,l,u,z,solver.dual)
            count=info.iter
            # Enforce identical, explicitly evaluated residual criteria. OSQP's
            # internal scaling may stop earlier than these original-QP metrics.
            if self.backend=='osqp':
                for refinement in range(1,5):
                    if max(qp_metrics[k] for k in ['qp_primal_relative','qp_dual_relative','qp_normal_relative'])<=self.qp_tol:break
                    solver.solver.update_settings(eps_abs=self.qp_tol*10.**(-refinement),eps_rel=self.qp_tol*10.**(-refinement))
                    z,info=solver.solve();count+=info.iter
                    qp_metrics=residuals(H,g,A,l,u,z,solver.dual)
            qp_metrics.update(qp_iterations=info.iter,
                qp_converged=int(max(qp_metrics[k] for k in ['qp_primal_relative','qp_dual_relative','qp_normal_relative'])<=self.qp_tol),
                preprocessing_ms=getattr(solver,'preprocessing_ms',0.),
                kernel_setup_ms=getattr(solver,'kernel_setup_ms',0.))
            qp_metrics['qp_iterations']=count
            if not qp_metrics['qp_converged']:
                raise RuntimeError(f'{self.backend} did not reach common QP tolerance: {qp_metrics}')
            result=[z[:len(p0['g'])],z[len(p0['g']):]];iterations=1;dual_residual=0.;local_iterations=count
        else:
            rho=self.rho
            solvers=[QPSolver(p['H']+rho*p['E'].T@p['E'],p['g'],p['A'],p['l'],p['u'],w)
                     for p,w in zip(parts,guesses)]
            setup_ms=1000*(time.perf_counter()-setup);ts=time.perf_counter()
            result=guesses;Y=[p['E']@w+p['d'] for p,w in zip(parts,result)];consensus=(Y[0]+Y[1])/2
            if self.dual is None:dual=[np.zeros_like(consensus),np.zeros_like(consensus)]
            else:
                dual=[];alpha=.02/self.dt
                for d in self.dual:
                    a=d.reshape(self.N,12).copy()
                    for j in range(4):a[:,3*j:3*j+3]=a[:,3*j:3*j+3]@self.warm_rotation.T
                    a=(1-alpha)*a+alpha*np.vstack((a[1:],np.zeros((1,12))));a[-1]=0;dual.append(a.ravel())
            local_iterations=0;dual_residual=np.inf
            for iterations in range(1,self.iterations+1):
                gradients=[p['g']+rho*p['E'].T@(p['d']-consensus+d) for p,d in zip(parts,dual)]
                if self.pool:
                    pending=[self.pool.submit(solver.solve,g) for solver,g in zip(solvers,gradients)]
                    solved=[p.result() for p in pending]
                else:solved=[solver.solve(g) for solver,g in zip(solvers,gradients)]
                result=[a for a,b in solved];local_iterations+=sum(b.iter for a,b in solved)
                Y=[p['E']@w+p['d'] for p,w in zip(parts,result)]
                old=consensus;consensus=(Y[0]+dual[0]+Y[1]+dual[1])/2
                dual=[d+y-consensus for d,y in zip(dual,Y)]
                primal=float(np.max(np.abs(Y[0]-Y[1])))
                dual_residual=max(float(np.max(np.abs(rho*p['E'].T@(consensus-old)))) for p in parts)
                if primal<1e-5 and dual_residual<1e-4:break
            self.dual=dual
        solve_ms=1000*(time.perf_counter()-ts)
        Y=[p['E']@w+p['d'] for p,w in zip(parts,result)]
        primal=float(np.max(np.abs(Y[0]-Y[1])))
        violations=[float(max(0,np.max(p['l']-p['A']@w),np.max(p['A']@w-p['u']))) for p,w in zip(parts,result)]
        self.last=[r.copy() for r in result]
        self.previous_rotation=self.current_rotation.copy() if hasattr(self,'current_rotation') else None
        return result,dict(setup_ms=setup_ms,solve_ms=solve_ms,iterations=iterations,
            interface_scaled=primal,dual_residual=dual_residual,local_primal=max(violations),local_iterations=local_iterations,
            admm_converged=int(primal<1e-5 and dual_residual<1e-4),**qp_metrics)

    def control(self,s,t):
        start=time.perf_counter();parts=self.build(s,t);model_ms=1000*(time.perf_counter()-start)
        result,info=self.solve(parts);u=[p['scale']*(p['controls'][0]@w) for p,w in zip(parts,result)]
        tau=np.r_[u[0][:12],u[1][:6]];self.prev_tau=tau.copy()
        a=[p['accels'][0][0]@w+p['accels'][0][1] for p,w in zip(parts,result)]
        acc=np.r_[a[0],a[1][6:]];force=u[0][12:24]
        T,a0,F,f0=contact(s);actual_acc=T@tau+a0
        # Diagnostics use the full model independently of local OCP constraints.
        full_res=s['M']@acc+s['h']-np.r_[np.zeros(6),tau]-s['Jc'].T@force
        defects=[];wrench_res=[];vel_res=[];nonlinear=[];pose_res=[]
        for k in range(self.N):
            uk=[p['scale']*(p['controls'][k]@w) for p,w in zip(parts,result)]
            ak=[p['accels'][k][0]@w+p['accels'][k][1] for p,w in zip(parts,result)]
            af=np.r_[ak[0],ak[1][6:]];tk=np.r_[uk[0][:12],uk[1][:6]]
            defects.append(s['M']@af+s['h']-np.r_[np.zeros(6),tk]-s['Jc'].T@uk[0][12:24])
            wrench_res.append(uk[0][-6:]+uk[1][-6:])
            vk=[p['physical'][k][2]@w+p['physical'][k][3] for p,w in zip(parts,result)]
            vel_res.append(vk[0][:6]-vk[1][:6])
            dqk=[p['physical'][k][0]@w+p['physical'][k][1] for p,w in zip(parts,result)]
            pose_res.append(dqk[0][:6]-dqk[1][:6])
            if k==0:state=s
            else:
                prevq=[p['physical'][k-1][0]@w+p['physical'][k-1][1] for p,w in zip(parts,result)]
                prevv=[p['physical'][k-1][2]@w+p['physical'][k-1][3] for p,w in zip(parts,result)]
                qpred=self.plant.integrate(s['q'],np.r_[prevq[0],prevq[1][6:]])
                state=self.plant.evaluate(qpred,np.r_[prevv[0],prevv[1][6:]])
            nonlinear.append(state['M']@af+state['h']-np.r_[np.zeros(6),tk]-state['Jc'].T@uk[0][12:24])
        info.update(model_ms=model_ms,controller_ms=1000*(time.perf_counter()-start),
            full_dynamics_max=float(np.max(np.abs(defects))),first_dynamics_max=float(np.max(np.abs(full_res))),
            wrench_force_max=float(np.max(np.abs(np.array(wrench_res)[:,:3]))),
            wrench_moment_max=float(np.max(np.abs(np.array(wrench_res)[:,3:]))),
            interface_velocity_max=float(np.max(np.abs(vel_res))),
            interface_pose_increment_max=float(np.max(np.abs(pose_res))),
            nonlinear_base_force_max=float(np.max(np.abs(np.array(nonlinear)[:,:3]))),
            nonlinear_base_moment_max=float(np.max(np.abs(np.array(nonlinear)[:,3:6]))),
            nonlinear_joint_torque_max=float(np.max(np.abs(np.array(nonlinear)[:,6:]))),
            first_acceleration_defect=float(np.max(np.abs(acc-actual_acc))),
            contact_acceleration_residual=float(np.max(np.abs(s['Jc']@acc+s['cdotv']+100*s['ce']+20*s['Jc']@s['v']))))
        self.info=info;self.debug=dict(tau=tau.copy(),acc=acc.copy(),actual_acc=actual_acc,force=force,interface_force=u[0][-6:])
        self.last_parts=parts;self.last_result=result
        self.info['controller_ms']=1000*(time.perf_counter()-start)
        return tau,acc
