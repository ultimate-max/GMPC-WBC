"""Offline common feasible standing trajectory, independent of either WBC.

Analytic smooth base/arm motion; legs integrated using spherical-foot rolling
kinematics. No-slip velocity and acceleration constraints are solved explicitly.
Base body derivatives use centered log increments (h=1 ms). Every sample has
an independently checked inverse-dynamics feasible force/torque solution.
This is a reference replay experiment, NOT an online MPC implementation.
"""
import argparse
import hashlib
import json
import pathlib
import time
import numpy as np
import scipy.linalg as la
from scipy.integrate import solve_ivp
import pinocchio as pin
import quadprog
import native
from geometric_wbc import pose,cone_matrix,rolling_contact


def smooth(t):
    u=np.clip(t-.5,0,1)
    return u**3*(10-15*u+6*u*u)


class CommonReference:
    def __init__(self,plant,case,duration=12.):
        self.plant=plant;self.case=case
        s=plant.truth();self.home=s['q'].copy();self.base0=pose(self.home)
        self.scale=2. if case=='double' else 1.
        self.offset=np.array([.02,-.01,.01,.10,-.10,.10]) if case=='offset' else np.zeros(6)
        initial=self.initial_configuration()
        self.initial=initial.copy()
        self.ode=solve_ivp(self.rolling_velocity,(0.,duration+.01),initial[7:19],
            method='DOP853',rtol=2e-11,atol=2e-12,max_step=.04,dense_output=True)
        if not self.ode.success:raise RuntimeError(self.ode.message)

    def base_pose(self,t):
        z=t-.5;r=smooth(t);w=2*np.pi*.2
        d=self.scale*r*np.array([.015*np.sin(w*z),.010*np.sin(.7*w*z),.010*np.sin(w*z),
            .08*np.sin(.8*w*z),.10*np.sin(w*z),.08*np.sin(.7*w*z)])+(1-r)*self.offset
        return self.base0*pin.exp6(d)

    def arm(self,t):
        z=t-.5;w=2*np.pi*.2
        return self.home[19:]+self.scale*smooth(t)*np.array([.12,.12,.12,.10,.10,.15])*np.sin(w*z*np.array([.7,1.,.8,.9,.6,1.1]))

    def initial_configuration(self):
        t=0.
        q=self.home.copy();T=self.base_pose(t)
        q[:3]=T.translation;q[3:7]=pin.Quaternion(T.rotation).coeffs();q[19:]=self.arm(t)
        for _ in range(25):
            s=self.plant.evaluate(q,np.zeros(24))
            if np.max(np.abs(s['ce']))<2e-12:break
            dq=la.solve(s['Jc'][:,6:18],-s['ce'])
            q[7:19]+=np.clip(dq,-.15,.15)
        else:raise RuntimeError(f'Contact IK failed at {t}')
        return q

    def base_derivatives(self,t,h=.00002):
        T=self.base_pose(t)
        dp=pin.log6(T.inverse()*self.base_pose(t+h)).vector
        dm=pin.log6(T.inverse()*self.base_pose(t-h)).vector
        return (dp-dm)/(2*h),(dp+dm)/(h*h)

    def rolling_velocity(self,t,legs):
        q=self.home.copy();T=self.base_pose(t)
        q[:3]=T.translation;q[3:7]=pin.Quaternion(T.rotation).coeffs()
        q[7:19]=legs;q[19:]=self.arm(t)
        s=self.plant.evaluate(q,np.zeros(24));vb,_=self.base_derivatives(t)
        return la.solve(s['Jc'][:,6:18],-s['Jc'][:,:6]@vb)

    def configuration(self,t):
        q=self.home.copy();T=self.base_pose(t)
        q[:3]=T.translation;q[3:7]=pin.Quaternion(T.rotation).coeffs()
        q[7:19]=self.ode.sol(t);q[19:]=self.arm(t)
        return q

    def sample(self,t,h=.001):
        q=self.configuration(t)
        T=self.base_pose(t)
        dp=pin.log6(T.inverse()*self.base_pose(t+h)).vector
        dm=pin.log6(T.inverse()*self.base_pose(t-h)).vector
        v=np.zeros(24);a=np.zeros(24)
        v[:6]=(dp-dm)/(2*h);a[:6]=(dp+dm)/(h*h)
        v[18:]=(self.arm(t+h)-self.arm(t-h))/(2*h)
        a[18:]=(self.arm(t+h)+self.arm(t-h)-2*self.arm(t))/(h*h)
        s=self.plant.evaluate(q,np.zeros(24))
        v[6:18]=la.solve(s['Jc'][:,6:18],-s['Jc'][:,:6]@v[:6])
        s=rolling_contact(self.plant.evaluate(q,v))
        a[6:18]=la.solve(s['Jc'][:,6:18],-s['cdotv']-s['Jc'][:,:6]@a[:6])
        rhs=s['M']@a+s['h'];Jt=s['Jc'].T
        cone=cone_matrix()
        C=np.vstack((Jt[:6],cone,-Jt[6:]))
        low=np.r_[rhs[:6],np.tile([-np.inf]*4+[10.],4),-.85*s['tau_max']-rhs[6:]]
        high=np.r_[rhs[:6],np.tile([0.]*4+[800.],4),.85*s['tau_max']-rhs[6:]]
        # Force QP in scaled coordinates, including actuator inequalities.
        Cs=C*100.;rows=1/np.maximum(la.norm(Cs,axis=1),1e-12)
        Cs=rows[:,None]*Cs;lo=rows*low;hi=rows*high
        il=np.flatnonzero(np.isfinite(lo[6:]))+6;iu=np.flatnonzero(np.isfinite(hi[6:]))+6
        Q=np.vstack((Cs[:6],Cs[il],-Cs[iu])).T
        b=np.r_[lo[:6],lo[il],-hi[iu]]
        target=np.tile([0.,0.,float(s['h'][2])/4],4)/100.
        try:sol=quadprog.solve_qp(np.eye(12),target,Q,b,meq=6)[0]
        except ValueError as exc:
            from scipy.optimize import linprog
            lp=linprog(np.zeros(12),A_ub=-Q[:,6:].T,b_ub=-b[6:],A_eq=Q[:,:6].T,b_eq=b[:6],bounds=[(None,None)]*12,method='highs')
            raise RuntimeError(f'Force feasibility failed at t={t}: max a={np.max(np.abs(a))}; independent LP={lp.message}') from exc
        f=100*sol;tau=(rhs-Jt@f)[6:]
        defect=max(0,np.max(low-C@f),np.max(C@f-high))
        cp=float(np.max(np.abs(s['ce'])));cv=float(np.max(np.abs(s['Jc']@v)))
        ca=float(np.max(np.abs(s['Jc']@a+s['cdotv'])))
        qmargin=float(min(np.min(q[7:]-s['q_min']-.02),np.min(s['q_max']-.02-q[7:])))
        vmargin=float(np.min(np.minimum(s['v_max'],4.)-np.abs(v[6:])))
        amargin=float(np.min(np.r_[np.full(12,50.),np.full(6,30.)]-np.abs(a[6:])))
        if defect>1e-7 or cp>1e-9 or cv>1e-9 or ca>1e-8 or qmargin<0 or vmargin<0 or amargin<0:
            raise RuntimeError(f'Reference feasibility failed at {t}: {defect}, {cp}, {cv}, {ca}, {qmargin}, {vmargin}')
        return dict(q=q,v=v,a=a,base_pose=T.homogeneous,base_twist=v[:6],base_acc=a[:6],
            ee_pose=s['pose'],ee_twist=s['J']@v,ee_acc=s['J']@a+s['jdotv'],force=f,tau=tau,
            checks=np.array([defect,cp,cv,ca,qmargin,vmargin,amargin]))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',choices=['normal','double','offset'],required=True)
    ap.add_argument('--duration',type=float,default=12.);ap.add_argument('--dt',type=float,default=.005)
    args=ap.parse_args();root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/geometric_wbc'
    out.mkdir(parents=True,exist_ok=True);p=native.Plant(str(root));start=time.perf_counter()
    r=CommonReference(p,args.case,args.duration);times=np.arange(round(args.duration/args.dt)+1)*args.dt
    samples=[r.sample(float(t)) for t in times]
    values={k:np.array([s[k] for s in samples]) for k in samples[0]}
    np.savez_compressed(out/f'reference_{args.case}.npz',time=times,home=r.home,**values)
    checks=values['checks'];digest=hashlib.sha256()
    for k in sorted(values):digest.update(values[k].tobytes())
    summary=dict(args=vars(args),samples=len(times),precompute_ms=1000*(time.perf_counter()-start),
        reference_sha256=digest.hexdigest(),checked_sample_max=checks[:,:4].max(axis=0).tolist(),
        min_joint_position_margin=float(checks[:,4].min()),min_joint_velocity_margin=float(checks[:,5].min()),
        min_joint_acceleration_margin=float(checks[:,6].min()),
        source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        meaning='Offline common trajectory; no online MPC. Contact and dynamics feasibility checked at every 5 ms sample, not a continuous-time proof.')
    (out/f'reference_{args.case}.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
