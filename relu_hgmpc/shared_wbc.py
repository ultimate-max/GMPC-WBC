"""Shared execution WBC: physical constraints precede all tracking tasks."""
import numpy as np
import pinocchio as pin
from geometric_wbc import GeometricWBC,pose
class CommonWBC(GeometricWBC):
    """Execution feedback plus the common moving joint-reference envelope."""
    def control(self,s,reference):
        # Joint error can grow in weakly observable task-space directions even
        # with small end-effector error. A common, explicit execution envelope
        # is applied to ALL scheduling policies, above soft pose tasks.
        e=s['q'][19:]-reference['q'][19:];de=s['v'][18:]-reference['v'][18:]
        center=reference['a'][18:]-16*de-64*e
        # Use physical bounds only, identically for both upper methods.
        arm_error=e.copy()
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
            arm_envelope_margin=float(1-np.max(np.abs(arm_error))/.08),
            acceleration_margin=float(np.min(1-np.abs(a[6:])/np.r_[np.full(12,50.),np.full(6,30.)])))
        return tau,a


