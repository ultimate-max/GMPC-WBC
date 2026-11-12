"""Paper's SNS allocator, with either original adapted WBC or shared WBC.

6D standing extension on B2+Z1, not author code or their 3D experiments.
The shared-WBC adapter retains allocation, activation, saturation and nullspace
SV logic; converting its commands to C2 plans is explicitly an adaptation.
"""
import numpy as np
import pinocchio as pin
from prioritized import PrioritizedController,rpy_rate_map,rpy_rate_map_dot
from online_wbc_mpc import Plan

class EEReference:
    def __init__(self,target):self.target=target
    def pose_at(self,t):return self.target.at(t)['ee_pose']
    def twist(self,t):return self.target.at(t)['ee_twist']

class PaperRateController(PrioritizedController):
    """50 Hz unchanged allocation; original WBC equations evaluated at 200 Hz."""
    def allocate(self,s,t):
        if not hasattr(self,'cached') or round(t/.005)%4==0:
            self.cached=super().allocate(s,t)
        _,command,acc,ff,error=self.cached
        return self.coordinates(s),command,acc,ff,error

class PaperUpper:
    def __init__(self,plant,target):
        self.plant=plant
        self.allocator=PrioritizedController(plant,EEReference(target),task_dim=6,dt=.02)
    def solve(self,s,t,anchor,prev_tau,force,release_delay=.02):
        # Same committed release reference used as the MPC's initial q/v.
        # A 20 ms reference-delay compensation, not a prediction horizon.
        predicted=self.plant.evaluate(anchor[0],anchor[1])
        self.allocator.dt=release_delay
        _,cmd,acmd,_,_=self.allocator.allocate(predicted,t+release_delay)
        # Shared-WBC adaptation: bounded velocity-to-reference conversion.
        # Integrate from the committed reference, rather than repeatedly
        # interpolating a lagging position integral with discontinuous acmd.
        # This adapter is NOT part of the paper's original controller.
        states=[tuple(x.copy() for x in anchor)]
        q,v,_=anchor
        for _ in range(2):
            R=pin.Quaternion(q[6],*q[3:6]).matrix();rpy=pin.rpy.matrixToRpy(R)
            desired=np.zeros(24);desired[:3]=R.T@cmd[:3]
            desired[3:6]=rpy_rate_map(rpy)@cmd[3:6];desired[18:]=cmd[6:]
            a=(desired-v)/.04
            limits=np.r_[np.full(3,2.),np.full(3,3.),np.full(12,50.),np.full(6,8.)]
            a=np.clip(a,-limits,limits)
            q=self.plant.integrate(q,.04*v+.5*.04**2*a);v=v+.04*a
            states.append((q.copy(),v.copy(),a.copy()))
        self.last_info=dict(self.allocator.allocation_info)
        return Plan(self.plant,t+release_delay,.04,states,np.tile(prev_tau,(2,1)),np.tile(force,(2,1)))
