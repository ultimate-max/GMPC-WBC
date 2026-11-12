"""Contact-consistent model and reference; Pinocchio order is [linear; angular]."""
import numpy as np
import scipy.linalg as la
import pinocchio as pin

def se3(T): return pin.SE3(T[:3,:3],T[:3,3])
def log(T): return pin.log6(se3(T)).vector
def ad(v): return np.asarray(pin.Motion(v).action)

def contact(s):
    M,J=s['M'],s['Jc']; S=np.vstack((np.zeros((6,18)),np.eye(18)))
    K=np.block([[M,-J.T],[J,np.zeros((12,12))]])
    rhs=np.vstack((S,np.zeros((12,18))))
    rhs0=np.r_[-s['h'],-s['cdotv']-100*s['ce']-20*J@s['v']]
    out=la.solve(K,np.column_stack((rhs,rhs0)))
    return out[:24,:18],out[:24,18],out[24:,:18],out[24:,18]

def equilibrium(s):
    J=s['Jc']; jb=J[:,:6].T
    f=jb.T@la.solve(jb@jb.T,s['h'][:6])
    return s['h'][6:]-J[:,6:].T@f

def fixed_linearization(plant,home,tau):
    s=plant.evaluate(home,np.zeros(24)); T,a0,F,f0=contact(s)
    A=np.zeros((48,48));A[:24,24:]=np.eye(24)
    force_x=np.zeros((12,48)); eps=1e-5
    for k in range(48):
        dq=np.zeros(24);v=np.zeros(24)
        if k<24:dq[k]=eps
        else:v[k-24]=eps
        vals=[]
        for sign in (1,-1):
            st=plant.evaluate(plant.integrate(home,sign*dq),sign*v)
            t,b,f,c=contact(st);vals.append((t@tau+b,f@tau+c))
        A[24:,k]=(vals[0][0]-vals[1][0])/(2*eps)
        force_x[:,k]=(vals[0][1]-vals[1][1])/(2*eps)
    B=np.vstack((np.zeros((24,18)),T))
    return A,B,force_x,T,a0,F,f0

def discretize(A,B,c,dt):
    n=len(A); m=B.shape[1]
    aug=np.zeros((n+m+1,n+m+1));aug[:n,:n]=A;aug[:n,n:n+m]=B;aug[:n,-1]=c
    e=la.expm(dt*aug)
    return e[:n,:n],e[:n,n:n+m],e[:n,-1]

def condense(A,B,c,N,K=None):
    nx,nu=B.shape
    if K is None:K=np.zeros((nu,nx))
    Ac=A-B@K
    E=np.eye(nx);S=np.zeros((nx,N*nu));h=np.zeros(nx)
    states=[];inputs=[]
    for k in range(N):
        um=-K@S;um[:,k*nu:(k+1)*nu]+=np.eye(nu)
        inputs.append((-K@E,um,-K@h))
        S=Ac@S;S[:,k*nu:(k+1)*nu]+=B
        E=Ac@E;h=Ac@h+c
        states.append((E.copy(),S.copy(),h.copy()))
    return states,inputs

def condense_varying(models,K):
    nx,nu=models[0][1].shape;N=len(models)
    E=np.eye(nx);S=np.zeros((nx,N*nu));h=np.zeros(nx)
    states=[];inputs=[]
    for k,(A,B,c) in enumerate(models):
        Ac=A-B@K
        um=-K@S;um[:,k*nu:(k+1)*nu]+=np.eye(nu)
        inputs.append((-K@E,um,-K@h))
        S=Ac@S;S[:,k*nu:(k+1)*nu]+=B
        E=Ac@E;h=Ac@h+c
        states.append((E.copy(),S.copy(),h.copy()))
    return states,inputs

class Reference:
    def __init__(self,plant,home,pose,scale=1):
        self.plant=plant;self.home=home.copy();self.pose=se3(pose);self.scale=scale
        self.cache={}
    def pose_at(self,t):
        z=max(0.,t-.5);r=min(z,1.);r=r*r*r*(10+r*(-15+6*r));p=2*np.pi*.2*z
        d=self.scale*r*np.array([.025*np.sin(p),.015*(1-np.cos(p)),.010*np.sin(p),.06*np.sin(p),.04*np.sin(.7*p),.08*np.sin(p)])
        return (self.pose*pin.exp6(d)).homogeneous
    def twist(self,t):
        eps=1e-4
        return log(np.linalg.inv(self.pose_at(t))@self.pose_at(t+eps))/eps
    def joint(self,t):
        key=round(t,6)
        if key in self.cache:return self.cache[key]
        q=self.home.copy();goal=se3(self.pose_at(t))
        for _ in range(30):
            s=self.plant.evaluate(q,np.zeros(24));err=pin.log6(se3(s['pose']).inverse()*goal).vector
            if np.linalg.norm(err)<1e-8:break
            J=s['J'][:,18:];step=J.T@la.solve(J@J.T+1e-6*np.eye(6),err)
            q[19:]+=np.clip(step,-.1,.1)
            q[7:]=np.clip(q[7:],s['q_min']+.01,s['q_max']-.01)
        if np.linalg.norm(err)>2e-3:raise RuntimeError(f'IK reference infeasible at {t}: {np.linalg.norm(err)}')
        v=np.zeros(24);v[18:]=np.linalg.lstsq(s['J'][:,18:],self.twist(t),rcond=1e-8)[0]
        self.cache[key]=(q,v);return q,v
