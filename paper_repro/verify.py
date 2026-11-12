"""Nontrivial checks of reduction, frames and generated derivatives."""
from pathlib import Path
import numpy as np
import casadi as cs
import pinocchio as pin
ROOT=Path(__file__).resolve().parents[1]; G=ROOT/'paper_repro/generated'
names=['front','rear','arm']; model=pin.buildModelFromUrdf(str(ROOT/'models/b2_z1.urdf'),pin.JointModelFreeFlyer());data=model.createData()
home=np.loadtxt(ROOT/'models/q_home.txt');rng=np.random.default_rng(13)
full=cs.Function.load(str(G/'central_step.casadi'))
terms=[cs.Function.load(str(G/(n+'_terms.casadi'))) for n in names]
steps=[cs.Function.load(str(G/(n+'_step.casadi'))) for n in names]
costs=[cs.Function.load(str(G/(n+'_task_cost.casadi'))) for n in ['central']+names]
peak_dyn=peak_pin=peak_cost=peak_jac=0.
for trial in range(12):
    q=pin.integrate(model,home,rng.normal(0,.03,24));v=rng.normal(0,.1,24);u=rng.normal(0,2,30);u[[20,23,26,29]]+=194
    x=np.r_[q,v];p=np.array([.55,0,.8]);xs=[];us=[];hs=[];ws=[]
    for i in range(3):
        xs.append(np.r_[q[:7],q[7+6*i:13+6*i],v[:6],v[6+6*i:12+6*i]])
        us.append(np.r_[u[6*i:6*i+6],u[18+6*i:24+6*i] if i<2 else []])
        h,w=terms[i](xs[-1],us[-1]);hs.append(np.array(h));ws.append(np.array(w).ravel())
    merged=np.zeros(49);cost=0
    for i in range(3):
        par=np.r_[p,(sum(hs)-hs[i]).flatten(order='F'),sum(ws)-ws[i],v[:6],xs[i],us[i],1.]
        z=np.array(steps[i](xs[i],us[i],par)).ravel()
        merged[:7]+=z[:7]/3;merged[25:31]+=z[13:19]/3
        merged[7+6*i:13+6*i]=z[7:13];merged[31+6*i:37+6*i]=z[19:25]
        cost+=float(costs[i+1](xs[i],us[i],par))
    y=np.array(full(x,u,p)).ravel();peak_dyn=max(peak_dyn,np.max(np.abs(y-merged)))
    peak_cost=max(peak_cost,abs(float(costs[0](x,u,p))-cost))
    # Independent numerical Pinocchio forward dynamics with world-velocity transform.
    R=pin.Quaternion(q[3:7]).matrix();T=np.eye(24);T[:3,:3]=R.T;T[3:6,3:6]=R.T
    vp=T@v;tdv=np.zeros(24);tdv[:3]=-R.T@np.cross(v[3:6],v[:3])
    pin.computeJointJacobians(model,data,q);pin.updateFramePlacements(model,data);J=[]
    for foot in ['FL_foot','FR_foot','RL_foot','RR_foot']:
        j=pin.getFrameJacobian(model,data,model.getFrameId(foot),pin.LOCAL_WORLD_ALIGNED)
        J.append((j[:3]-pin.skew(np.array([0,0,-.032]))@j[3:])@T)
    force=np.r_[np.zeros(6),u[:18]]+np.vstack(J).T@u[18:]
    a=T.T@(pin.aba(model,data,q,vp,T@force)-tdv)
    peak_pin=max(peak_pin,np.max(np.abs(y[25:]-(v+.01*a))))
    if trial==0:
        sx=cs.MX.sym('x',49);su=cs.MX.sym('u',30);sp=cs.MX.sym('p',3)
        jac=cs.Function('test_jacobian',[sx,su,sp],[cs.jacobian(full(sx,su,sp),cs.vertcat(sx,su))])
        direction=rng.normal(size=79);direction/=np.linalg.norm(direction);eps=1e-6
        fd=(np.array(full(x+eps*direction[:49],u+eps*direction[49:],p))-np.array(full(x-eps*direction[:49],u-eps*direction[49:],p))).ravel()/(2*eps)
        peak_jac=np.max(np.abs(np.array(jac(x,u,p))@direction-fd))
print(dict(samples=12,decomposition_max_abs=peak_dyn,pinocchio_max_abs=peak_pin,task_cost_max_abs=peak_cost,autodiff_direction_max_abs=peak_jac))
assert peak_dyn<1e-8 and peak_pin<1e-8 and peak_cost<1e-8 and peak_jac<1e-5
