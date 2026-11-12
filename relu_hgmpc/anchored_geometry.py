"""Reverse-chain mounting closure. Pinocchio motion order [linear; angular].

Y=T_WA=T_WB C, F=T_AE, D=T_WE_des. Yreq=D F^-1.
E_I=Yreq^-1 Y=F (D^-1 T_WE) F^-1, e_I=Ad_F e.
The arm configuration in F is a decision variable, not a frozen target.
"""
import numpy as np
import pinocchio as pin
from geometric_wbc import pose, task_error, GeometricWBC


def mount_transform(model, q):
    data=model.createData()
    pin.forwardKinematics(model,data,q);pin.updateFramePlacements(model,data)
    return pose(q).inverse()*data.oMf[model.getFrameId('z1_link00')]


def relative_arm(s,C):
    return (pose(s['q'])*C).inverse()*pin.SE3(s['pose'])


def linearize(s,C,D,Vd):
    F=relative_arm(s,C);E=D.inverse()*pin.SE3(s['pose'])
    e=pin.log6(E).vector;L=pin.Jlog6(E)
    Ja=np.zeros_like(s['J']);Ja[:,18:]=s['J'][:,18:]
    Ji=F.action@(L@s['J']-pin.Motion(e).action@Ja)
    bt=-F.action@L@E.inverse().action@Vd
    return F.action@e,Ji,bt


def pulled_back_linearize(s,C,D,Vd):
    """Use the EE metric on the reverse-chain closure, without false rank loss.

    r=Ad_F^-1 log(Yreq^-1 Y). Differentiating Ad_F^-1 is essential:
    J_r=Ad_F^-1 J_I + ad_r J_arm. This cancels the configuration-dependent
    chart term. Algebraically identical to log(D^-1 YF), as it must be.
    """
    # Evaluate the algebraically simplified expression. Multiplying and then
    # cancelling adjoints introduces roundoff in near-dependent task locks.
    E=D.inverse()*pin.SE3(s['pose']);L=pin.Jlog6(E)
    return pin.log6(E).vector,L@s['J'],-L@E.inverse().action@Vd


def error_dynamics(s,C,D,Vd,Ad):
    F=relative_arm(s,C)
    e,de,J,b=task_error(pin.SE3(s['pose']),s['J']@s['v'],s['jdotv'],s['J'],D,Vd,Ad,'se3')
    Ja=np.zeros_like(s['J']);Ja[:,18:]=s['J'][:,18:]
    w=Ja@s['v'];aw=pin.Motion(w).action
    dw0=s['jdotv']+aw@(s['J']@s['v']-w)
    return (F.action@e,F.action@(de+aw@e),
            F.action@(J-pin.Motion(e).action@Ja),
            F.action@(b+2*aw@de+pin.Motion(dw0).action@e+aw@aw@e))


class AnchoredWBC(GeometricWBC):
    def __init__(self,plant,C):
        super().__init__(plant,'se3',dt=.005)
        self.C=C;self.ee_first=True
    def anchored_error(self,s,D,Vd,Ad):
        return error_dynamics(s,self.C,D,Vd,Ad)
