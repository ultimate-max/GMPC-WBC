"""Centralized EE-anchor > nominal base + regularized SV > smoothness.

Reuses the existing frozen-affine full dynamics, contact elimination, horizon,
hard bounds, condensation, and certified lexicographic QP. Only task rows change.
No interface force duplication and no distributed optimizer is introduced.
"""
import time
import numpy as np
import pinocchio as pin
from shared_upper import SharedUpper
from anchor_qp import AnchorQP
from anchored_geometry import mount_transform,pulled_back_linearize
from body_arm_tasks import manipulability_linearization
from geometric_wbc import pose


class AnchorHierarchy:
    def __init__(self,owner):self.owner=owner;self.qp=AnchorQP()
    def solve(self,levels,G,lo,hi,initial):
        o=self.owner;s=o.state;t=o.time;start=time.perf_counter()
        m,grad=manipulability_linearization(o.model,s['q'],eps=.02)
        sigma=float(np.linalg.svd(s['J'][:,18:],compute_uv=False)[-1])
        def smooth(x):
            x=np.clip(x,0.,1.);return x*x*(3.-2*x)
        # A single closure task; these normalized motion weights distribute
        # effort, not independent base/arm endpoint set-points.
        arm_weight=.2+.6*(1.-smooth((sigma-.02)/.06))
        mobility=np.r_[np.full(3,np.sqrt(1-arm_weight)/.1),np.full(3,np.sqrt(1-arm_weight)/.3),np.full(6,np.sqrt(arm_weight)/.5)]
        sv_weight=smooth((.03-sigma)/.02)
        now=o.target.at(t);E0=pin.SE3(now['ee_pose']).inverse()*pin.SE3(s['pose'])
        e0=pin.log6(E0).vector
        task_distance=max(np.linalg.norm(e0[:3])/.02,np.linalg.norm(e0[3:])/.1)
        base_gate=.02*smooth((task_distance-.1)/.9)
        primary=[];bias=[];secondary_C=[np.sqrt(base_gate)*levels[0][0]]
        secondary_d=[np.sqrt(base_gate)*levels[0][1]]
        wv=np.sqrt(np.r_[np.full(3,20.),np.full(3,8.)])
        base_lim=np.r_[np.full(3,.1),np.full(3,.3)]
        for k in range(o.N):
            g=G[134*k:134*(k+1)];l=lo[134*k:134*(k+1)]
            P=np.vstack((g[128:134],g[92:110]))
            p=np.r_[-base_lim-l[128:134],-.15-l[92:110]]
            target=o.target.at(t+(k+1)*o.dt)
            E=pin.SE3(target['base_pose']).inverse()*pose(s['q'])
            bv=E.inverse().action@target['base_twist']
            V=np.vstack((levels[0][0][12*k+6:12*k+12]/wv[:,None],g[110:128]))
            v=np.r_[levels[0][1][12*k+6:12*k+12]/wv+bv,
                    -np.minimum(s['v_max'],4.)-l[110:128]]
            if o.error_mode=='anchor':
                e,J,bt=pulled_back_linearize(s,o.C,pin.SE3(target['ee_pose']),target['ee_twist'])
            else:
                E=pin.SE3(target['ee_pose']).inverse()*pin.SE3(s['pose']);L=pin.Jlog6(E)
                e=pin.log6(E).vector;J=L@s['J'];bt=-L@E.inverse().action@target['ee_twist']
            wp=np.sqrt((3. if k==o.N-1 else 1.)*np.r_[np.full(3,4000.),np.full(3,1000.)])
            primary.extend((wp[:,None]*(J@P),wv[:,None]*(J@V)))
            bias.extend((wp*(e+J@p),wv*(J@v+bt)))
            idx=np.r_[np.arange(6),np.arange(18,24)]
            secondary_C.append(mobility[:,None]*P[idx]);secondary_d.append(mobility*p[idx])
            if sv_weight>0:
                secondary_C.append((-.5*np.sqrt(sv_weight)*grad@P[18:])[None,:])
                secondary_d.append(np.array([np.sqrt(sv_weight)*(1.-.5*grad@p[18:])]))
        secondary=(np.vstack(secondary_C),np.concatenate(secondary_d))
        newlevels=[(np.vstack(primary),np.concatenate(bias)),secondary,levels[2]]
        assembly_ms=1000*(time.perf_counter()-start)
        x,info=self.qp.solve(newlevels,G,lo,hi,initial)
        info.update(anchor_task_construction_ms=assembly_ms,regularized_manipulability=m,
                    arm_motion_weight=arm_weight,base_motion_weight=1-arm_weight,
                    base_nominal_gate=base_gate,singularity_preference_weight=sv_weight)
        self.last_levels=newlevels
        return x,info


class AnchoredMPC(SharedUpper):
    def __init__(self,plant,root,target,error_mode='anchor'):
        super().__init__(plant,root,target,'mpc')
        self.C=mount_transform(self.model,plant.truth()['q'])
        self.error_mode=error_mode;self.hierarchy=AnchorHierarchy(self)
    def solve(self,s,t,*args,**kwargs):
        self.state=s;self.time=t
        return super().solve(s,t,*args,**kwargs)
