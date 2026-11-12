"""Independent identities and approximation-order checks for paper GMPC."""
import pathlib,json
import numpy as np
import pinocchio as pin
import scipy.linalg as la
import native
from model import Reference,se3,ad
from paper_gmpc import PaperGMPC,ReferenceCache,task_coefficients

root=pathlib.Path(__file__).resolve().parents[1];p=native.Plant(str(root));s=p.truth()
ref=Reference(p,s['q'],s['pose']);c=PaperGMPC(p,ref,root,mode='central')
rng=np.random.default_rng(11102026);records=[]
for i in range(6):
    q=p.integrate(s['q'],rng.normal(size=24)*.025);v=rng.normal(size=24)*.2
    st=p.evaluate(q,v);parts=c.build(st,1.2)
    for part,a in enumerate(parts):
        idx=c.parts.indices[part];vel=v[idx]
        assert np.max(np.abs(a['C']@vel+a['gravity']-a['h']))<1e-8
        if part!=1:continue
        J=st['J'][:,idx];coeff=a['task'];u=rng.normal(size=a['nu'])*.2
        B=np.column_stack((np.vstack((np.zeros((6,6)),np.eye(6))),np.vstack((np.eye(6),np.zeros((6,6))))))*a['scale'][None,:]
        acceleration=la.solve(a['M'],B@u-a['h'],assume_a='pos')
        exact=c.dJ[:,idx]@vel+J@acceleration
        affine=coeff['H']@(J@vel)+coeff['F']@u+coeff['b']
        error=float(np.max(np.abs(exact-affine)));assert error<1e-8
        nullerr=float(np.max(np.abs(J@coeff['vn'])));assert nullerr<1e-9
        eps=1e-6
        plus=p.evaluate(p.integrate(q,eps*v),v);minus=p.evaluate(p.integrate(q,-eps*v),v)
        dJfd=(plus['J']-minus['J'])/(2*eps)
        jerr=float(np.max(np.abs(dJfd-c.dJ)));assert jerr<1e-7
        records.append(dict(task_acceleration_identity=error,velocity_nullspace_identity=nullerr,jacobian_time_derivative_error=jerr))
# Exact log-error vector field versus paper first-order model: residual O(eps^2).
axis=rng.normal(size=6);dvel=rng.normal(size=6);vd=rng.normal(size=6)
errors=[]
for eps in [.02,.01,.005,.0025]:
    phi=eps*axis;V=vd+eps*dvel;E=pin.exp6(phi)
    exact=pin.Jlog6(E)@(V-E.inverse().action@vd)
    linear=-ad(vd)@phi+V-vd
    errors.append(float(la.norm(exact-linear)))
ratios=np.array(errors[:-1])/errors[1:];assert np.all((ratios>3.8)&(ratios<4.2)),ratios
# Exact reference and constant-block reuse must not change the mathematical QP.
cached=ReferenceCache(ref,2.,.02,10,.03)
reuse=PaperGMPC(p,cached,root,mode='central',reuse_static=True)
a=c.build(s,1.2);b=reuse.build(s,1.2);differences={}
# Independently evaluate the manuscript's 12D block transition and C_i
# coordinate transformation, including the time-varying reference offset.
# This catches using next-step V with current-step Vd in the pose update.
arm=a[1];idx=c.parts.indices[1];tc=arm['task'];dt=c.dt
x=np.r_[pin.log6(se3(ref.pose_at(1.2)).inverse()*se3(s['pose'])).vector,s['J'][:,idx]@s['v'][idx]]
z=rng.normal(size=c.N*arm['nu'])*.01;step_errors=[]
for k in range(c.N):
    vd=ref.twist(1.2+k*dt);vn=ref.twist(1.2+(k+1)*dt)
    CI=np.block([[np.eye(6),np.zeros((6,6))],[-ad(vd),np.eye(6)]])
    CN=np.block([[np.eye(6),np.zeros((6,6))],[-ad(vn),np.eye(6)]])
    d=np.r_[np.zeros(6),vd];dn=np.r_[np.zeros(6),vn]
    Ac=np.block([[-ad(vd),np.eye(6)],[np.zeros((6,6)),tc['H']]])
    Bd=dt*np.vstack((np.zeros_like(tc['F']),tc['F']));Ad=np.eye(12)+dt*Ac
    hd=dt*np.r_[-vd,tc['b']];u=arm['controls'][k]@z
    y=CI@x-d
    yn=CN@Ad@la.solve(CI,y)+CN@Bd@u+CN@Ad@la.solve(CI,d)+CN@hd-dn
    x=la.solve(CN,yn+dn)
    emap,eoff=arm['taskmaps'][k]
    step_errors.append(float(np.max(np.abs(x[:6]-(emap@z+eoff)))))
assert max(step_errors)<1e-10,step_errors
for key in ['H','g','A','l','u','E','d']:
    err=0.
    for x,y in zip(a,b):
        finite=np.isfinite(x[key]);assert np.array_equal(finite,np.isfinite(y[key]))
        err=max(err,float(np.max(np.abs(x[key][finite]-y[key][finite]))/max(1.,la.norm(x[key][finite]))))
    differences[key]=err;assert err<1e-9,(key,err)
tau,acc=c.control(s,0.)
assert c.info['first_acceleration_defect']<1e-6,c.info
assert c.info['affine_dynamics_max']<1e-6,c.info
assert c.info['qp_converged']==1,c.info
result=dict(identities=records,error_field_residuals=errors,error_halving_ratios=ratios.tolist(),
    paper_discrete_prediction_errors=step_errors,reuse_QP_relative_differences=differences,first_cycle=c.info)
out=root/'results/paper_gmpc';out.mkdir(parents=True,exist_ok=True)
(out/'verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
