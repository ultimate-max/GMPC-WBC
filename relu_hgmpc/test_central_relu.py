"""Physical-QP equivalence, active constraints, and snapshot solver agreement."""
import ast,json,pathlib,time
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import osqp,native
from central_relu import CentralReLU,residuals
from model import Reference
from split_gmpc import SplitGMPC

root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/central_relu';out.mkdir(exist_ok=True)

def method(path,name):
    module=ast.parse(path.read_text())
    return ast.dump(next(n for n in ast.walk(module) if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
for name in ['build_part','build','warm']:
    assert method(root/'relu_hgmpc/split_gmpc.py',name)==method(out/'backup/split_gmpc.py',name),name

def reference(H,g,A,l,u):
    rows=1/np.maximum(1.,np.linalg.norm(A,axis=1));s=osqp.OSQP()
    s.setup(P=sp.csc_matrix(np.triu((H+H.T)/2)),q=g,A=sp.csc_matrix(rows[:,None]*A),
        l=rows*l,u=rows*u,eps_abs=1e-10,eps_rel=1e-10,polishing=True,max_iter=50000,verbose=False)
    r=s.solve();assert r.info.status_val in (1,2),r.info.status
    return r.x,rows*r.y

rng=np.random.default_rng(112026);tests=[]
for duplicate in [False,True]:
    n=12;B=rng.normal(size=(n,n));H=B.T@B+np.eye(n);g=rng.normal(size=n)*5
    E=rng.normal(size=(3,n));A=np.vstack((E,np.eye(n)));l=np.r_[np.zeros(3),np.full(n,-.2)];u=np.r_[np.zeros(3),np.full(n,.2)]
    if duplicate:A=np.vstack((A,E[0]));l=np.r_[l,0.];u=np.r_[u,0.]
    xr,dr=reference(H,g,A,l,u);solver=CentralReLU(H,g,A,l,u,np.zeros(n),tol=1e-8,max_iter=6000,rho=.1)
    x,info=solver.solve();err=float(np.max(np.abs(x-xr)))
    assert info.converged and err<1e-5,(info,err)
    assert np.any(np.abs(x)>.199),'Test must exercise active inequalities'
    tests.append(dict(duplicate_equality=duplicate,control_error=err,**vars(info)))

p=native.Plant(str(root));s=p.truth();ref=Reference(p,s['q'],s['pose'])
trace=np.load(root/'results/split_gmpc/central_12s_trace.npz');snapshots=[]
for index in [0,60,200,400,599]:
    state=p.evaluate(trace['q'][index],trace['v'][index]);t=index*.02
    base=SplitGMPC(p,ref,root,mode='central',backend='osqp');gpu=SplitGMPC(p,ref,root,mode='central',backend='reluqp')
    if index:base.prev_tau=trace['tau'][index-1].copy();gpu.prev_tau=base.prev_tau.copy()
    parts=base.build(state,t);other=gpu.build(state,t)
    for a,b in zip(parts,other):
        for key in ['H','g','A','l','u','E','d']:assert np.array_equal(a[key],b[key]),key
    xc,ic=base.solve(parts);xg,ig=gpu.solve(other)
    obj=lambda xs:sum(.5*x@a['H']@x+a['g']@x for a,x in zip(parts,xs))
    gap=float(abs(obj(xc)-obj(xg)));delta=float(max(np.max(np.abs(x-y)) for x,y in zip(xc,xg)))
    assert gap<1e-3,(t,gap)
    assert ig['qp_converged'] and ic['qp_converged']
    snapshots.append(dict(time=t,objective_gap=gap,max_normalized_control_difference=delta,osqp=ic,reluqp=ig))
result=dict(model_constraints_weights_warm_start_unchanged=True,random_active_constraint_tests=tests,snapshots=snapshots)
(out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
