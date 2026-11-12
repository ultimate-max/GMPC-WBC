"""Matched upper: EE pose/rate > base pose/rate > redundancy and smoothness.

The body task has its own lexicographic level. Unlike AnchoredMPC, neither
mobility penalties nor an error-dependent gate can trade against that level.
Dynamics, hard bounds, N=10 and h=40 ms remain those of SharedUpper.
"""
import time
import numpy as np
import pinocchio as pin
from shared_upper import SharedUpper
from anchor_qp import AnchorQP
from anchored_geometry import mount_transform,pulled_back_linearize
from body_arm_tasks import manipulability_linearization
from geometric_wbc import pose

class StrictTasks:
    def __init__(self,owner):self.owner=owner;self.qp=AnchorQP()
    def solve(self,levels,G,lo,hi,initial):
        o=self.owner;s=o.state;t=o.time;start=time.perf_counter()
        m,gradient=manipulability_linearization(o.model,s['q'],eps=.02)
        C=[];d=[];Cs=[];ds=[];damping_rows=[];damping_lo=[];damping_hi=[]
        wv=np.sqrt(np.r_[np.full(3,20.),np.full(3,8.)])
        limits=np.r_[np.full(3,.1),np.full(3,.3)]
        for k in range(o.N):
            g=G[134*k:134*(k+1)];l=lo[134*k:134*(k+1)]
            P=np.vstack((g[128:134],g[92:110]))
            p=np.r_[-limits-l[128:134],-.15-l[92:110]]
            target=o.target.at(t+(k+1)*o.dt)
            Eb=pin.SE3(target['base_pose']).inverse()*pose(s['q'])
            vb=Eb.inverse().action@target['base_twist']
            V=np.vstack((levels[0][0][12*k+6:12*k+12]/wv[:,None],g[110:128]))
            v=np.r_[levels[0][1][12*k+6:12*k+12]/wv+vb,-np.minimum(s['v_max'],4.)-l[110:128]]
            if o.joint_damping:
                eta=o.joint_damping;q=s['q'][7:]
                damping_rows.append(V[6:]+eta*P[6:])
                damping_lo.append(-eta*(q-s['q_min']-.02)-v[6:]-eta*p[6:])
                damping_hi.append(eta*(s['q_max']-.02-q)-v[6:]-eta*p[6:])
            e,J,b=pulled_back_linearize(s,o.C,pin.SE3(target['ee_pose']),target['ee_twist'])
            wp=np.sqrt((3. if k==o.N-1 else 1.)*np.r_[np.full(3,4000.),np.full(3,1000.)])
            C.extend((wp[:,None]*(J@P),wv[:,None]*(J@V)))
            d.extend((wp*(e+J@p),wv*(J@v+b)))
            # A soft preference only, below BOTH pose tracking levels.
            Cs.append(-.5*gradient@P[18:]);ds.append(1.-.5*gradient@p[18:])
        tasks=[(np.vstack(C),np.concatenate(d)),levels[0],
               (np.vstack((levels[2][0],Cs)),np.r_[levels[2][1],ds])]
        assembly_ms=1000*(time.perf_counter()-start)
        self.last_tasks=tasks
        if damping_rows:
            G=np.vstack((G,*damping_rows));lo=np.r_[lo,np.concatenate(damping_lo)];hi=np.r_[hi,np.concatenate(damping_hi)]
        try:z,info=self.qp.solve(tasks,G,lo,hi,initial)
        except Exception:
            np.savez_compressed('/tmp/strict_gmpc_failed.npz',G=G,lo=lo,hi=hi,initial=initial,
                **{f'C{i}':a for i,(a,b) in enumerate(tasks)},**{f'd{i}':b for i,(a,b) in enumerate(tasks)})
            raise
        info.update(task_construction_ms=assembly_ms,regularized_manipulability=m,
                    task_order='EE > base > redundancy/smoothness',joint_damping=o.joint_damping)
        return z,info

class StrictGMPC(SharedUpper):
    def __init__(self,plant,root,target,joint_damping=0.):
        super().__init__(plant,root,target,'mpc')
        self.C=mount_transform(self.model,plant.truth()['q'])
        self.joint_damping=joint_damping
        self.hierarchy=StrictTasks(self)
    def solve(self,s,t,*args,**kwargs):
        self.state=s;self.time=t
        plan=super().solve(s,t,*args,**kwargs)
        # Independent physical-coordinate check, after native prediction and
        # Lie-group retraction: do not certify only the assembled matrix.
        violation=0.
        if self.joint_damping:
            for q,v,a in self.debug['states'][1:]:
                violation=max(violation,float(np.max(v[6:]-self.joint_damping*(s['q_max']-.02-q[7:]))),
                    float(np.max(-v[6:]-self.joint_damping*(q[7:]-s['q_min']-.02))))
            if violation>2e-5:raise RuntimeError(f'Physical joint damping check: {violation}')
        self.last_info['joint_damping_violation']=violation
        return plan
