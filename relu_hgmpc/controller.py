import time
import numpy as np
import scipy.linalg as la
from model import contact,equilibrium,fixed_linearization,discretize,condense,condense_varying,log,ad
from relu_solver import ReLUQP

class Costs:
    def __init__(self,n): self.n=n;self.a=[];self.b=[]
    def add(self,A,b,weight):
        w=np.sqrt(np.broadcast_to(weight,(len(b),)))
        self.a.append(w[:,None]*A);self.b.append(w*b)
    def arrays(self):
        C=np.vstack(self.a);d=np.concatenate(self.b)
        return 2*C.T@C,2*C.T@d,C,d

class Constraints:
    def __init__(self):self.a=[];self.l=[];self.u=[]
    def add(self,A,b,lo,hi):
        self.a.append(A);self.l.append(np.broadcast_to(lo,b.shape)-b);self.u.append(np.broadcast_to(hi,b.shape)-b)
    def arrays(self):return np.vstack(self.a),np.concatenate(self.l),np.concatenate(self.u)

class Controller:
    def __init__(self,plant,ref,mode,N=10,dt=.03,iterations=100):
        self.plant=plant;self.ref=ref;self.mode=mode;self.N=N;self.dt=dt;self.iters=iterations
        self.initial=plant.truth();self.home=self.initial['q'].copy();self.tau0=equilibrium(self.initial)
        self.scale=np.r_[np.full(12,100.),np.full(6,10.)];self.Su=np.diag(self.scale)
        self.last_u=None;self.solver=None;self.index=0
        A,B,self.fx,self.T,self.a0,self.F,self.f0=fixed_linearization(plant,self.home,self.tau0)
        self.A,self.B,self.c=discretize(A,B@self.Su,np.r_[np.zeros(24),self.T@self.tau0+self.a0],dt)
        self.Q=np.diag(np.r_[np.full(3,5000.),np.full(3,3000.),np.full(12,30.),np.full(6,600.),np.full(6,60.),np.full(12,2.),np.full(6,8.)])
        self.R=.08*np.eye(18)
        self.P=la.solve_discrete_are(self.A,self.B,self.Q,self.R)
        self.K=la.solve(self.R+self.B.T@self.P@self.B,self.B.T@self.P@self.A)
        if np.max(np.abs(la.eigvals(self.A-self.B@self.K)))>=1:raise RuntimeError('Unstable LQR preconditioner')
        self.maps=condense(self.A,self.B,self.c,N,self.K)

    def build(self,s,t):
        N=self.N;nu=N*18; geo=self.mode=='hgmpc'
        if geo:
            T,a0,F,f0=contact(s);J=s['J'];Vd=self.ref.twist(t)
            # Frozen contact-consistent inverse dynamics. The contact feedback
            # acts only in constrained directions, so J*T torque nullspace is
            # invariant across the prediction. General nonlinear derivatives
            # would require preserving the complete condensed task map instead.
            MinvJ=la.solve(s['M'],s['Jc'].T,assume_a='pos')
            Kc=MinvJ@la.inv(s['Jc']@MinvJ)
            Aq=-100*Kc@s['Jc'];Av=-20*Kc@s['Jc']
            Ac=np.zeros((54,54));Bc=np.zeros((54,18));cc=np.zeros(54)
            Ac[:24,24:48]=np.eye(24);Ac[24:48,:24]=Aq;Ac[24:48,24:48]=Av
            Ac[48:,48:]=-ad(Vd);Ac[48:,24:48]=J
            Bc[24:48]=T@self.Su;cc[24:48]=a0-Av@s['v']+T@self.tau0;cc[48:]=-Vd
            K=np.column_stack((self.K,np.zeros((18,6))))
            models=[]
            for k in range(N):
                vd=self.ref.twist(t+k*self.dt)
                Ac[48:,48:]=-ad(vd);cc[48:]=-vd
                models.append(discretize(Ac,Bc,cc,self.dt))
            states,inputs=condense_varying(models,K)
            x=np.r_[np.zeros(24),s['v'],log(np.linalg.inv(self.ref.pose_at(t))@s['pose'])]
            offset=self.plant.difference(self.home,s['q'])
            # Actual frozen force affine approximation consistent with Aq/Av.
            force_x=np.zeros((12,54))
            Fc=la.inv(s['Jc']@MinvJ)
            force_x[:,:24]=-100*Fc@s['Jc'];force_x[:,24:48]=-20*Fc@s['Jc']
            fbase=f0-force_x[:,24:48]@s['v']
        else:
            T,a0,F,fbase=self.T,self.a0,self.F,self.f0
            J=self.initial['J'];states,inputs=self.maps
            x=np.r_[self.plant.difference(self.home,s['q']),s['v']]
            offset=np.zeros(24);force_x=self.fx
        nstate=len(x); cons=Constraints();cost=Costs(nu);physical=[];up=[];tasks=[]
        prev=(np.zeros((nstate,nu)),x)
        mu=.6/np.sqrt(2)
        cone=la.block_diag(*([np.array([[1,0,-mu],[-1,0,-mu],[0,1,-mu],[0,-1,-mu],[0,0,1]])]*4))
        fl=np.tile([-np.inf]*4+[10.],4);fu=np.tile([0.]*4+[800.],4)
        for k,((E,S,h),(Ue,Um,Uh)) in enumerate(zip(states,inputs)):
            xb=E@x+h;ub=Ue@x+Uh
            pm,pb=prev
            # Torque variables are deviations from the shared stance equilibrium.
            tauM=self.Su@Um;taub=self.tau0+self.scale*ub
            cons.add(tauM,taub,-.85*s['tau_max'],.85*s['tau_max'])
            cons.add(S[6:24],xb[6:24]+s['q'][7:] if geo else xb[6:24]+self.home[7:],s['q_min']+.02,s['q_max']-.02)
            vm=np.minimum(s['v_max'],4.)
            cons.add(S[30:48],xb[30:48],-vm,vm)
            cons.add(S[:6],xb[:6]+offset[:6],[-.15,-.15,-.08,-.25,-.25,-.25],[.15,.15,.08,.25,.25,.25])
            fm=F@tauM+force_x@pm;fb=F@taub+fbase+force_x@pb
            cons.add(cone@fm,cone@fb,fl,fu)
            physical.append((S,xb));up.append((Um,ub))
            if geo:
                vd=self.ref.twist(t+(k+1)*self.dt)
                cm=J@S[24:48]-ad(vd)@S[48:]
                cb=J@xb[24:48]-ad(vd)@xb[48:]-vd
                mult=3 if k==N-1 else 1
                cost.add(S[48:],xb[48:],mult*np.r_[np.full(3,8000),np.full(3,2000)])
                cost.add(cm,cb,mult*np.r_[np.full(3,100),np.full(3,30)])
                # Only task-effective input cost: no full-torque primary penalty.
                accM=J@T@tauM
                accb=J@(T@taub+a0)+s['jdotv']
                vdot=(self.ref.twist(t+(k+1)*self.dt+1e-3)-vd)/1e-3
                cost.add(accM,accb-vdot,.002)
                tasks.extend([S[48:],cm,accM])
            else:
                qr,vr=self.ref.joint(t+(k+1)*self.dt)
                xr=np.r_[self.plant.difference(self.home,qr),vr]
                qcost=self.P if k==N-1 else self.Q
                L=la.cholesky(qcost,lower=True).T
                cost.add(L@S,L@(xb-xr),1.)
                cost.add(Um,ub,.08)
            prev=(S,xb)
        H,g,_,_=cost.arrays();C,l,u=cons.arrays()
        return dict(H=H,g=g,C=C,l=l,u=u,physical=physical,up=up,tasks=tasks,
                    T=T,a0=a0,J=J,offset=offset,x=x)

    def control(self,s,t):
        start=time.perf_counter();p=self.build(s,t)
        setup_ms=0.;qp_ms=0.
        if self.solver is None or self.mode=='hgmpc':
            self.solver=ReLUQP(p['H'],p['C']);setup_ms+=self.solver.setup_ms
        # Shift physical torque sequence then invert the LQR input transform.
        U=np.vstack([a for a,b in p['up']]);ub=np.concatenate([b for a,b in p['up']])
        warm=None
        if self.last_u is not None:
            shifted=np.r_[self.last_u[18:],self.last_u[-18:]]
            warm=la.solve_triangular(U,shifted-ub,lower=True)
        w=self.solver.solve(p['g'],p['l'],p['u'],self.iters,xwarm=warm)
        primary_info=self.solver.info.copy();qp_ms+=primary_info['qp_ms']
        hierarchy=0.;sec_info={};pri_w=w.copy()
        if self.mode=='hgmpc':
            Z=la.block_diag(*([la.null_space(p['J']@p['T']@self.Su,rcond=1e-9)]*self.N))
            D=la.solve_triangular(U,Z,lower=True)
            sec=Costs(D.shape[1]);weight=np.r_[np.full(3,10000),np.full(3,6000),np.full(12,30),np.full(6,.2)]
            for k,((S,b),(Um,bm)) in enumerate(zip(p['physical'],p['up'])):
                # Posture regulation keeps the base near home and permits the arm
                # to move. Joint bounds and contact constraints remain hard.
                sec.add(S[:24]@D,S[:24]@w+b[:24]+p['offset'],weight)
                sec.add(S[24:48]@D,S[24:48]@w+b[24:48],np.r_[np.full(6,60),np.full(12,2),np.full(6,.5)])
                sec.add(Um@D,Um@w+bm,.02)
                if k:
                    pm,pb=p['up'][k-1]
                    sec.add((Um-pm)@D,(Um-pm)@w+bm-pb,.05)
            hs,gs,_,_=sec.arrays();secsolver=ReLUQP(hs,p['C']@D)
            setup_ms+=secsolver.setup_ms
            z=secsolver.solve(gs,p['l']-p['C']@w,p['u']-p['C']@w,self.iters,xwarm=np.zeros(D.shape[1]))
            sec_info=secsolver.info.copy();qp_ms+=sec_info['qp_ms']
            w=w+D@z
            hierarchy=float(np.max(np.abs(np.vstack(p['tasks'])@(w-pri_w))))
        un=U@w+ub;self.last_u=un.copy();self.index+=1
        torque=self.tau0+self.scale*un[:18]
        acc=p['T']@torque+p['a0']
        violation=float(max(0.,np.max(p['l']-p['C']@w),np.max(p['C']@w-p['u'])))
        self.last_problem=p;self.last_solution=w;self.primary_solution=pri_w
        self.info=dict(controller_ms=1000*(time.perf_counter()-start),setup_ms=setup_ms,qp_ms=qp_ms,
            primal=violation,primary_dual=primary_info['scaled_dual'],primary_primal=primary_info['scaled_primal'],
            secondary_dual=sec_info.get('scaled_dual',0.),hierarchy=hierarchy,
            n=p['H'].shape[0],m=p['C'].shape[0],qps=2 if self.mode=='hgmpc' else 1)
        return torque,acc
