import json,pathlib
import numpy as np
import scipy.linalg as la
import native
from prioritized import distribute,shaped_bounds,hierarchy,mp,solve_wbc_qp,PrioritizedController,rpy_rate_map,rpy_rate_map_dot
from model import Reference

rng=np.random.default_rng(20261109);n=12
rpy=np.array([.2,-.3,.1]);rates=np.array([.1,.2,-.1]);eps=1e-6
eerr=np.max(np.abs((rpy_rate_map(rpy+eps*rates)-rpy_rate_map(rpy-eps*rates))/(2*eps)-rpy_rate_map_dot(rpy,rates)))
assert eerr<1e-8,eerr
# Saturated arm must cause base activation and preserve the attainable task.
J=rng.normal(size=(3,n));lo=-np.ones(n)*.03;hi=-lo
ff=np.array([.1,-.1,.08]);v,info=distribute(J,ff,np.zeros(3),lo,hi,rng.normal(size=n)*2)
assert info['base_active']==1 and info['saturations']>0,info
assert info['distribution_task_residual']<1e-7 and info['velocity_violation']<1e-8,info
assert info['tertiary_task_change']<1e-9 and info['frozen_dof_change']<1e-9,info
# Deliberately infeasible unscaled command: phi must decrease, not violate bounds.
w,scaled=distribute(J,100*ff,np.zeros(3),lo,hi,np.zeros(n))
assert 0<scaled['phi']<1 and scaled['velocity_violation']<1e-8,scaled
# Tertiary projection must really move while preserving fixed coordinates/task.
J2=np.column_stack((np.zeros((3,6)),np.eye(3),np.eye(3)))
v2,ter=distribute(J2,np.ones(3)*.1,np.zeros(3),-np.ones(n),np.ones(n),rng.normal(size=n))
assert ter['base_active']==0 and np.linalg.norm(v2[6:])>.1 and np.max(np.abs(v2[:6]))<1e-10
assert ter['tertiary_task_change']<1e-9 and ter['frozen_dof_change']<1e-9,ter
# Counterexamples for interpreting the printed equations literally.
free=np.r_[np.zeros(6),np.ones(6)];fixed=np.r_[np.ones(6)*.02,np.zeros(6)];Js=J*free
literal=mp(Js)@ff+(np.eye(n)-mp(Js)@Js)@fixed
intended=fixed+mp(Js)@(ff-J@fixed)
bad7=float(np.linalg.norm(J@literal-ff));good7=float(np.linalg.norm(J@intended-ff))
assert bad7>1e-3 and good7<1e-10
P=np.eye(n)-mp(J)@J;D=np.diag(1-free);z=P@rng.normal(size=n)
printed17=(np.eye(n)-mp(D@P))@z
Z=la.null_space(np.vstack((J,D)));good17=Z@Z.T@z
assert np.linalg.norm(D@printed17)<1e-10 and np.linalg.norm(printed17-good17)<1e-10
# Actual B2+Z1 hierarchy preserves all preceding tasks during recursion.
root=pathlib.Path(__file__).resolve().parents[1];plant=native.Plant(str(root));s=plant.truth()
jrot=np.eye(24)[3:6];jpos=np.eye(24)[:3];jarm=np.eye(24)[18:]
tasks=[]
for jac in (s['Jc'],jrot,jpos,jarm):
    target=rng.normal(size=len(jac))*.1;tasks.append((jac,target,target,target))
dq,dv,a,err=hierarchy(s['M'],tasks)
assert err<1e-8,err
assert max(np.max(np.abs(j@a-d)) for j,_,_,d in tasks)<1e-8
# Braking limits shrink near a hard stop.
low,high=shaped_bounds(np.array([.99]),np.array([-1.]),np.array([1.]),np.array([4.]),np.array([8.]),.02)
assert abs(high[0]-.4)<1e-10
result=dict(rpy_derivative_error=float(eerr),saturation_test=info,task_scaling_test=scaled,tertiary_test=ter,
    printed_eq7_task_error=bad7,corrected_eq7_error=good7,
    eq17_fixed_dof_change=float(np.linalg.norm(D@printed17)),eq17_intersection_projection_difference=float(np.linalg.norm(printed17-good17)),
    wbc_hierarchy_error=err)
# Test actual Z1 configurations, not just a synthetic Jacobian.
ref=Reference(plant,s['q'],s['pose']);ctrl=PrioritizedController(plant,ref,3)
best=(np.inf,None)
for angle in np.linspace(s['q_min'][14]+.02,s['q_max'][14]-.02,35):
    q=s['q'].copy();q[21]=angle
    st=plant.evaluate(q,np.zeros(24));h,sigma=ctrl.manipulability(st)
    if sigma<best[0]:best=(sigma,st)
sigma,st=best;_,_,jac=ctrl.chart(st)
_,near=distribute(jac,np.array([.01,0,0]),np.zeros(3),np.full(12,-.2),np.full(12,.2),np.zeros(12),force_base=sigma<.1)
assert sigma<.1 and near['base_active']==1 and near['distribution_task_residual']<1e-8,(sigma,near)
result['b2z1_near_singular']=dict(sigma=sigma,**near)
# Regress the actual active-constraint QP that stalled the iterative backend.
failed=root/'results/prioritized/failure_6d.npz'
if failed.exists():
    from scipy.optimize import minimize,LinearConstraint
    data=np.load(failed);A,low,upp=data['A'],data['l'],data['u']
    sol,primal,dual,iterations=solve_wbc_qp(A,low,upp)
    scale=1/np.sqrt(np.r_[np.full(6,2e4),np.full(12,2e-4)])
    G=A*scale;E=1/np.maximum(np.linalg.norm(G,axis=1),1e-10);G=E[:,None]*G
    oracle=minimize(lambda x:.5*x@x,np.zeros(18),jac=lambda x:x,method='SLSQP',
        constraints=[LinearConstraint(G[:6],E[:6]*low[:6],E[:6]*upp[:6]),LinearConstraint(G[6:],E[6:]*low[6:],E[6:]*upp[6:])],
        options={'maxiter':1000,'ftol':1e-8})
    gap=abs(.5*np.sum((sol/scale)**2)-oracle.fun)
    assert oracle.success and gap<1e-6 and primal<1e-7 and dual<1e-7,(gap,primal,dual)
    result['active_wbc_qp']=dict(oracle_objective_gap=gap,primal=primal,dual=dual,iterations=iterations)
out=root/'results'/'prioritized';out.mkdir(parents=True,exist_ok=True)
(out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
