"""Pinocchio/CasADi dynamics and acados SQP-RTI for the paper-aligned benchmark.

State [p_world, quaternion_xyzw, joints, v_world, omega_world, joint_rates].
Local interface elimination uses a Schur complement; no wrench decisions.
"""
import os, sys, json
from pathlib import Path
import numpy as np
import casadi as cs
import pinocchio as pin
import pinocchio.casadi as cp
ROOT=Path(__file__).resolve().parents[1]
from partition import prepare
prepare(ROOT)
os.environ['ACADOS_SOURCE_DIR']=str(ROOT/'third_party/acados')
from acados_template import AcadosOcp, AcadosOcpSolver
OUT=ROOT/'paper_repro/generated'; OUT.mkdir(parents=True,exist_ok=True)
N=50; DT=.01
HOME=np.loadtxt(ROOT/'models/q_home.txt')
FOOTS=['FL_foot','FR_foot','RL_foot','RR_foot']

def rotation(q):
    # Quaternion on the unit sphere, propagated multiplicatively.
    a=q[3:6]; w=q[6]
    return cs.SX.eye(3)+2*w*cs.skew(a)+2*cs.skew(a)@cs.skew(a)

def integrate(q,v,dt):
    z=v[3:6]*dt; t2=cs.dot(z,z); t=cs.sqrt(t2+1e-30)
    fac=cs.if_else(t2<1e-8,.5-t2/48+t2*t2/3840,cs.sin(t/2)/t)
    dq=fac*z; dw=cs.if_else(t2<1e-8,1-t2/8+t2*t2/384,cs.cos(t/2))
    a=q[3:6]; w=q[6]
    qa=dw*a+w*dq+cs.cross(dq,a); qw=dw*w-cs.dot(dq,a)
    quat=cs.vertcat(qa,qw); quat=quat/cs.sqrt(cs.dot(quat,quat))
    return cs.vertcat(q[:3]+dt*v[:3],quat,q[7:]+dt*v[6:])

def construct(name):
    full=name=='central'; idx={'front':0,'rear':1,'arm':2}.get(name,0)
    path=ROOT/'models/b2_z1.urdf' if full else ROOT/'paper_repro/models'/(name+'.urdf')
    model=pin.buildModelFromUrdf(str(path),pin.JointModelFreeFlyer())
    cm=cp.Model(model); cd=cm.createData(); nq=model.nq; nv=model.nv; nj=nv-6
    feet=FOOTS if full else (FOOTS[2*idx:2*idx+2] if idx<2 else [])
    nc=len(feet); nu=nj+3*nc; nx=nq+nv
    qh=HOME.copy() if full else np.r_[HOME[:7],HOME[7+6*idx:13+6*idx]]
    x=cs.SX.sym('x',nx); u=cs.SX.sym('u',nu); q=x[:nq]; v=x[nq:]
    R=rotation(q); T=cs.diagcat(R.T,R.T,cs.SX.eye(nj)); vp=T@v
    tdv=cs.vertcat(-R.T@cs.cross(v[3:6],v[:3]),cs.SX.zeros(nv-3))
    cp.computeJointJacobians(cm,cd,q); cp.updateFramePlacements(cm,cd)
    jac=[]; pos=[]
    for foot in feet:
        fid=model.getFrameId(foot)
        jj=cp.getFrameJacobian(cm,cd,fid,pin.LOCAL_WORLD_ALIGNED)
        # Physical point on the bottom of the 32 mm spherical foot.
        jac.append((jj[:3,:]-cs.skew(cs.DM([0,0,-.032]))@jj[3:,:])@T)
        pos.append(cd.oMf[fid].translation)
    J=cs.vertcat(*jac) if nc else cs.SX.zeros(0,nv)
    feetpos=cs.vertcat(*pos) if nc else cs.SX.zeros(0,1)
    tip=cd.oMf[model.getFrameId('tool_tip')].translation if full or idx==2 else cs.SX.zeros(0,1)
    Mpin=cp.crba(cm,cd,q); Mpin=cs.triu(Mpin)+cs.triu(Mpin,False).T
    hp=cp.nonLinearEffects(cm,cd,q,vp)
    M=T.T@Mpin@T; h=T.T@(hp+Mpin@tdv)
    rhs=cs.vertcat(cs.SX.zeros(6),u[:nj])+J.T@u[nj:]-h
    # Common parameters: target xyz. Local: remote H, remote momentum,
    # consensus target z - scaled_dual, previous state (proximal regularizer).
    p=cs.SX.sym('p',3 if full else 3+36+6+6+nx+nu+1)
    if full:
        taupin=T@ (cs.vertcat(cs.SX.zeros(6),u[:nj])+J.T@u[nj:])
        a=T.T@(cp.aba(cm,cd,q,vp,taupin)-tdv)
        vn=v+DT*a
        terms=None
    else:
        D=M[6:,6:]; B=M[:6,6:]
        DinvBt=cs.solve(D,B.T); Dinvg=cs.solve(D,rhs[6:])
        H=M[:6,:6]-B@DinvBt; b=rhs[:6]-B@Dinvg
        momentum=H@v[:6]+DT*b
        otherH=cs.reshape(p[3:39],6,6); othermoment=p[39:45]
        alpha=p[-1]
        vbase=(1-alpha)*v[:6]+cs.solve(H+otherH,alpha*H@v[:6]+DT*b+othermoment)
        aj=Dinvg-DinvBt@(vbase-v[:6])/DT
        vn=cs.vertcat(vbase,v[6:]+DT*aj)
        terms=cs.Function(name+'_terms',[x,u],[cs.densify(H),cs.densify(momentum)])
    xn=cs.vertcat(integrate(q,vn,DT),vn)
    # The sum of local task costs is exactly the centralized task cost at consensus.
    posture=cs.vertcat(q[:3]-qh[:3],cp.log3(R),q[7:]-qh[7:],v)
    wy=np.r_[[100,100,5000],[300]*3,[1]*nj,[10]*6,[.5]*nj]
    if not full: wy[:6]/=3; wy[6+nj:12+nj]/=3
    fd=pin.Data(model); pin.framesForwardKinematics(model,fd,qh)
    foot_home=np.concatenate([fd.oMf[model.getFrameId(f)].translation for f in feet]) if nc else np.zeros(0)
    y=cs.vertcat(posture,feetpos-foot_home,tip-p[:3] if full or idx==2 else cs.SX.zeros(0,1))
    weights=np.r_[wy,[1000]*(3*nc),[2000]*(3 if full or idx==2 else 0)]
    ye=y; we=4*weights
    y=cs.vertcat(y,u); weights=np.r_[weights,[1e-4]*nj,[1e-6]*(3*nc)]
    task_cost=cs.Function(name+'_task_cost',[x,u,p],[.5*cs.dot(y,cs.DM(weights)*y)])
    if not full:
        # Standard scaled consensus convention: ||v-z+y||^2, y += v-z.
        uscale=cs.DM(np.r_[[.01]*nj,[.005]*(3*nc)])
        y=cs.vertcat(y,v[:6]-p[45:51],x-p[51:51+nx],uscale*(u-p[51+nx:51+nx+nu])); weights=np.r_[weights,[200]*6,[100]*nx,[0]*nu]
        # The trajectory proximal norm includes its terminal state as well.
        ye=cs.vertcat(ye,v[:6]-p[45:51],x-p[51:51+nx]);we=np.r_[we,[200]*6,[100]*nx]
    ocp=AcadosOcp(); ocp.model.name='rti_'+name; ocp.name='rti_'+name
    ocp.model.x=x; ocp.model.u=u; ocp.model.p=p; ocp.model.disc_dyn_expr=xn
    ocp.parameter_values=np.zeros(p.numel())
    ocp.cost.cost_type='NONLINEAR_LS'; ocp.cost.cost_type_e='NONLINEAR_LS'
    ocp.model.cost_y_expr=y; ocp.model.cost_y_expr_e=ye
    ocp.cost.W=np.diag(weights); ocp.cost.W_e=np.diag(we)
    ocp.cost.yref=np.zeros(y.numel()); ocp.cost.yref_e=np.zeros(ye.numel())
    # Physical joint bounds; no free-base box constraint.
    bix=np.r_[np.arange(7,nq),np.arange(nq+6,nx)].astype(int)
    ocp.constraints.idxbx=bix; ocp.constraints.lbx=np.r_[model.lowerPositionLimit[7:],-model.velocityLimit[6:]]
    ocp.constraints.ubx=np.r_[model.upperPositionLimit[7:],model.velocityLimit[6:]]
    ocp.constraints.idxbu=np.arange(nj); ocp.constraints.lbu=-model.effortLimit[6:]; ocp.constraints.ubu=model.effortLimit[6:]
    friction=[]; low=[]; high=[]
    for i in range(nc):
        f=u[nj+3*i:nj+3*i+3]; mu=.6/np.sqrt(2)
        friction += [mu*f[2]+f[0],mu*f[2]-f[0],mu*f[2]+f[1],mu*f[2]-f[1],f[2]]
        low += [0]*5; high += [1e6]*4+[800]
    fc=cs.vertcat(*friction) if nc else cs.SX.zeros(0,1)
    slip=J@v; z=feetpos[2::3]
    if nc:
        # Contact velocity is softened as in Eq.6d. Ground height is hard after x0.
        ocp.model.con_h_expr=cs.vertcat(fc,slip,z)
        ocp.constraints.lh=np.r_[low,np.zeros(3*nc),np.zeros(nc)]
        ocp.constraints.uh=np.r_[high,np.zeros(3*nc),np.zeros(nc)]
        ocp.constraints.idxsh=np.arange(5*nc,8*nc)
        ocp.cost.zl=ocp.cost.zu=np.ones(3*nc)*100
        ocp.cost.Zl=ocp.cost.Zu=np.ones(3*nc)*1e4
        ocp.model.con_h_expr_0=fc
        ocp.constraints.lh_0=np.array(low); ocp.constraints.uh_0=np.array(high)
        ocp.constraints.idxsh_0=np.array([],dtype=int)
        ocp.cost.zl_0=ocp.cost.zu_0=ocp.cost.Zl_0=ocp.cost.Zu_0=np.zeros(0)
    ocp.constraints.x0=np.r_[qh,np.zeros(nv)]
    so=ocp.solver_options
    so.N_horizon=N; so.tf=N*DT; so.integrator_type='DISCRETE'; so.nlp_solver_type='SQP_RTI'
    so.hessian_approx='GAUSS_NEWTON'; so.qp_solver='PARTIAL_CONDENSING_HPIPM'; so.qp_solver_cond_N=10
    so.qp_solver_iter_max=50; so.qp_solver_tol_stat=so.qp_solver_tol_eq=so.qp_solver_tol_ineq=so.qp_solver_tol_comp=1e-6
    so.qp_solver_warm_start=1; so.levenberg_marquardt=1e-6; so.print_level=0
    so.ext_fun_compile_flags='-O3'; so.hpipm_mode='ROBUST'
    ocp.code_export_directory=str(OUT/name)
    diag=cs.Function(name+'_step',[x,u,p],[cs.densify(xn)])
    # Exact full-state dynamics evaluation for common residual checks.
    funcs=[diag,task_cost]
    if terms: funcs.append(terms)
    os.chdir(OUT); cg=cs.CodeGenerator('diag_'+name+'.c',{'with_header':True})
    for f in funcs: cg.add(f); f.save(str(OUT/(f.name()+'.casadi')))
    cg.generate()
    meta=dict(nx=nx,nu=nu,np=p.numel(),nq=nq,nv=nv,nj=nj,nc=nc,qhome=qh.tolist(),foot_home=foot_home.tolist())
    (OUT/(name+'_meta.json')).write_text(json.dumps(meta,indent=2))
    AcadosOcpSolver(ocp,json_file=str(OUT/(name+'.json')))
    print('GENERATED',name,meta,flush=True)

if __name__=='__main__':
    for name in (sys.argv[1:] or ['central','front','rear','arm']): construct(name)
