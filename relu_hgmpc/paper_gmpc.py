"""Paper first-order SE(3) error prediction for the existing B2+Z1 OCP.

Source: local Hierarchical Geometric... manuscript, (13)-(18), Assumption 1.
All twists use Pinocchio [linear; angular], the permutation of paper notation.
This is a frozen-coefficient task-space model, NOT full SQP or a full q/v
Taylor expansion. Four-foot/contact/interface constraints extend the fixed-base
paper. The original weighted task hierarchy is retained for this comparison.
"""
import time
import numpy as np
import scipy.linalg as la
import pinocchio as pin
from split_gmpc import SplitGMPC
from controller import Costs, Constraints
from model import se3, ad


class ReferenceCache:
    """Exact memoization at requested times; no reference approximation.

    Preparation is explicitly measured and included in amortized total cost.
    Cache misses are evaluated and charged in the feedback path.
    """
    def __init__(self,ref,duration,control_dt,N,dt,geometric=True):
        self.ref=ref;self.poses={};self.twists={};self.misses=0;self.error_blocks={};self.dt=dt
        start=time.perf_counter()
        times=set()
        for k in range(round(duration/control_dt)):
            for j in range(N+1):times.add(round(k*control_dt+j*dt,9))
        for t in sorted(times):
            self.poses[t]=ref.pose_at(t);self.twists[t]=ref.twist(t)
            if geometric:
                a=ad(self.twists[t]);self.error_blocks[t]=(np.eye(6)-dt*a,a)
        self.precompute_ms=1000*(time.perf_counter()-start)
        self.bytes=sum(v.nbytes for v in self.poses.values())+sum(v.nbytes for v in self.twists.values())
        self.bytes+=sum(a.nbytes+b.nbytes for a,b in self.error_blocks.values())
    def pose_at(self,t):
        key=round(t,9)
        if key not in self.poses:self.misses+=1;self.poses[key]=self.ref.pose_at(t)
        return self.poses[key]
    def twist(self,t):
        key=round(t,9)
        if key not in self.twists:self.misses+=1;self.twists[key]=self.ref.twist(t)
        return self.twists[key]
    def error_terms(self,t,dt):
        key=round(t,9)
        if dt!=self.dt:raise ValueError('Reference cache step mismatch')
        if key not in self.error_blocks:
            self.misses+=1;a=ad(self.twist(t));self.error_blocks[key]=(np.eye(6)-dt*a,a)
        return self.error_blocks[key]


def task_coefficients(M,C,gravity,J,dJ,v,force_map):
    """Paper H,F,b; includes the measured velocity-nullspace contribution."""
    MinvJT=la.solve(M,J.T,assume_a='pos')
    task_mass=J@MinvJT
    Jbar=la.solve(task_mass,MinvJT.T,assume_a='pos').T
    MinvC=la.solve(M,C,assume_a='pos')
    F=J@la.solve(M,force_map,assume_a='pos')
    K=dJ-J@MinvC
    vn=v-Jbar@(J@v)
    H=K@Jbar
    b=K@vn-J@la.solve(M,gravity,assume_a='pos')
    return dict(H=H,F=F,b=b,Jbar=Jbar,vn=vn,K=K)


class PaperGMPC(SplitGMPC):
    def make_costs(self,n):
        return Costs(n)

    def add_tracking_cost(self,cost,A,b,weight):
        cost.add(A,b,weight)

    def cost_metadata(self,cost):
        return {}

    def __init__(self,*args,geometry_only=False,reuse_static=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.geometry_only=geometry_only;self.reuse_static=reuse_static
        self.extra_data=[m.createData() for m in self.parts.models]
        self.kin_data=self.parts.full.createData();self.static={}
        self.task_id=self.parts.full.getFrameId('tool_tip')
        start=time.perf_counter()
        if reuse_static:
            for part in [0,1]:self.static[part]=self.make_static(part)
        self.static_precompute_ms=1000*(time.perf_counter()-start)

    def make_static(self,part):
        nj=12 if part==0 else 6;nf=12 if part==0 else 0;nu=nj+nf+6;n=self.N*nu
        scale=np.r_[np.full(nj,100. if part==0 else 10.),np.full(nf,100.),np.full(3,200.),np.full(3,50.)]
        S=np.vstack((np.zeros((6,nj)),np.eye(nj)));Us=[]
        for k in range(self.N):
            U=np.zeros((nu,n));U[:,k*nu:(k+1)*nu]=np.eye(nu);Us.append(U)
        mu=.6/np.sqrt(2)
        cone=la.block_diag(*([np.array([[1,0,-mu],[-1,0,-mu],[0,1,-mu],[0,-1,-mu],[0,0,1]])]*4))
        return scale,S,Us,cone

    def build(self,s,t):
        start=time.perf_counter();self.coefficients={}
        if not self.geometry_only:
            pin.computeJointJacobiansTimeVariation(self.parts.full,self.kin_data,s['q'],s['v'])
            pin.updateFramePlacements(self.parts.full,self.kin_data)
            self.dJ=np.array(pin.getFrameJacobianTimeVariation(self.parts.full,self.kin_data,self.task_id,pin.LOCAL))
            for part,(m,d,idx) in enumerate(zip(self.parts.models,self.extra_data,self.parts.indices)):
                C=np.array(pin.computeCoriolisMatrix(m,d,s['q'],s['v']))[np.ix_(idx,idx)].copy()
                gravity=np.array(pin.computeGeneralizedGravity(m,d,s['q']))[idx].copy()
                self.coefficients[part]=(C,gravity)
        self.derivatives_ms=1000*(time.perf_counter()-start)
        result=super().build(s,t)
        return result

    def build_part(self,s,t,part,M,h):
        idx=self.parts.indices[part];nv=len(idx);nj=nv-6;nf=12 if part==0 else 0
        nu=nj+nf+6;n=self.N*nu;dt=self.dt
        scale,S,Us,cone=self.static[part] if self.reuse_static else self.make_static(part)
        Jc=s['Jc'][:,idx] if part==0 else np.zeros((0,nv))
        force_map=np.column_stack((S,Jc.T,np.vstack((np.eye(6),np.zeros((nj,6))))))
        G=la.solve(M,force_map*scale[None,:],assume_a='pos');a0=-la.solve(M,h,assume_a='pos')
        v0=s['v'][idx]
        C,gravity=(np.zeros_like(M),h) if self.geometry_only else self.coefficients[part]
        Av=-la.solve(M,C,assume_a='pos')
        # h(q,v0)=C(q,v0)v0+gravity(q). Retain velocity coupling, rather
        # than treating h as a constant acceleration bias at future nodes.
        assert np.max(np.abs(C@v0+gravity-h))<1e-7
        f=la.lstsq(s['Jc'][:,:6].T,s['h'][:6])[0]
        wrench=h[:6]-(Jc.T@f)[:6] if part==0 else h[:6]
        tau_ref=h[6:]-(Jc.T@f)[6:] if part==0 else h[6:]
        uref=np.r_[tau_ref,f if part==0 else [],wrench]/scale
        cons=Constraints();cost=self.make_costs(n)
        qmap=np.zeros((nv,n));qoff=np.zeros(nv);vmap=np.zeros((nv,n));voff=v0.copy()
        base=pin.SE3(pin.Quaternion(s['q'][6],*s['q'][3:6]).matrix(),s['q'][:3])
        eb=pin.log6(self.base_home.inverse()*base).vector
        home_offset=self.plant.difference(self.home,s['q'])[idx]
        qnow=s['q'][7:19] if part==0 else s['q'][19:]
        limits=slice(0,12) if part==0 else slice(12,18)
        physical=[];controls=[];interfaces=[];interface_offsets=[];accels=[];taskmaps=[];vstarts=[]
        oldU=None;task=None
        if part==1:
            J=s['J'][:,idx]
            emap=np.zeros((6,n));eoff=pin.log6(se3(self.ref.pose_at(t)).inverse()*se3(s['pose'])).vector
            Vmap=np.zeros((6,n));Voff=J@v0
            if not self.geometry_only:
                task=task_coefficients(M,C,gravity,J,self.dJ[:,idx],v0,force_map*scale[None,:])
                AV=np.eye(6)+dt*task['H'];BV=dt*task['F'];bV=dt*task['b']
        for k,U in enumerate(Us):
            vstarts.append((vmap.copy(),voff.copy()))
            A=G@U+Av@vmap;b=a0+Av@(voff-v0)
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
            # Static base reference has Vd=0: dot(phi_b)=v_b. Both virtual
            # subsystems use the same first-order base-error coordinates.
            cons.add(qmap[:6],eb+qoff[:6],[-.15,-.15,-.08,-.25,-.25,-.25],[.15,.15,.08,.25,.25,.25])
            if part==0:
                cost.add(qmap[:6],eb+qoff[:6],np.r_[np.full(3,2000.),np.full(3,1000.)])
                cost.add(vmap[:6],voff[:6],np.full(6,20.))
            else:
                vd=self.ref.twist(t+k*dt)
                # Paper (22)-(24): forward Euler uses V_i and Vd_i together.
                # Advance phi before V; using V_(i+1) here with Vd_i would
                # introduce a reference time-staggering error.
                Aphi=self.ref.error_terms(t+k*dt,dt)[0] if self.reuse_static else np.eye(6)-dt*ad(vd)
                emap=Aphi@emap+dt*Vmap
                eoff=Aphi@eoff+dt*(Voff-vd)
                if self.geometry_only:Vmap=J@vmap;Voff=J@voff
                else:
                    Vmap=AV@Vmap+BV@U
                    Voff=AV@Voff+bV
                vdnext=self.ref.twist(t+(k+1)*dt)
                adnext=self.ref.error_terms(t+(k+1)*dt,dt)[1] if self.reuse_static else ad(vdnext)
                edmap=-adnext@emap+Vmap;edoff=-adnext@eoff+Voff-vdnext
                mult=3. if k==self.N-1 else 1.
                self.add_tracking_cost(cost,emap,eoff,mult*np.r_[np.full(3,8000.),np.full(3,2000.)])
                self.add_tracking_cost(cost,edmap,edoff,mult*np.r_[np.full(3,100.),np.full(3,30.)])
                taskmaps.append((emap.copy(),eoff.copy()))
            cost.add(qmap[6:],qoff[6:]+home_offset[6:],.3 if part==0 else .02)
            cost.add(vmap[6:],voff[6:],.2 if part==0 else .05)
            cost.add(U,-uref,np.r_[np.full(nj,.002),np.full(nf,.0002),np.full(6,.0001)])
            if oldU is None:cost.add(U[:nj],-self.prev_tau[limits]/scale[:nj],.2)
            else:cost.add((U-oldU)[:nj],np.zeros(nj),.2)
            vs=np.r_[np.full(3,.2),np.full(3,.5)]
            interfaces.append(np.vstack((vmap[:6]/vs[:,None],(1 if part==0 else -1)*U[-6:])))
            interface_offsets.append(np.r_[voff[:6]/vs,np.zeros(6)])
            physical.append((qmap.copy(),qoff.copy(),vmap.copy(),voff.copy()))
            controls.append(U);accels.append((A,b));oldU=U
        H,g,_,_=cost.arrays();H+=1e-9*np.eye(n);AA,l,u=cons.arrays()
        return dict(H=H,g=g,A=AA,l=l,u=u,E=np.vstack(interfaces),d=np.concatenate(interface_offsets),
            physical=physical,controls=controls,accels=accels,scale=scale,nu=nu,nj=nj,nf=nf,
            M=M,h=h,Jc=Jc,taskmaps=taskmaps,uref=uref,C=C,gravity=gravity,vstarts=vstarts,task=task,
            **self.cost_metadata(cost))

    def control(self,s,t):
        start=time.perf_counter();tau,acc=super().control(s,t)
        defects=[]
        for p,z in zip(self.last_parts,self.last_result):
            for k,U in enumerate(p['controls']):
                u=p['scale']*(U@z);a=p['accels'][k][0]@z+p['accels'][k][1]
                v=p['vstarts'][k][0]@z+p['vstarts'][k][1]
                force=p['Jc'].T@u[p['nj']:p['nj']+p['nf']]
                applied=np.r_[u[-6:],u[:p['nj']]]+force
                defects.append(p['M']@a+p['C']@v+p['gravity']-applied)
        self.info['affine_dynamics_max']=float(np.max(np.abs(np.concatenate(defects))))
        self.info['derivatives_ms']=self.derivatives_ms
        self.info['controller_ms']=1000*(time.perf_counter()-start)
        return tau,acc
