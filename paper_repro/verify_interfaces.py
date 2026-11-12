"""Independent signed-interface KKT, edge-ADMM, and common-mode tests."""
from pathlib import Path
import json
import numpy as np
import pinocchio as pin
import casadi as cs
from scipy.linalg import block_diag
ROOT=Path(__file__).resolve().parents[1];G=ROOT/'paper_repro/generated'
names=['front','rear','arm'];rng=np.random.default_rng(2718)
models=[pin.buildModelFromUrdf(str(ROOT/'paper_repro/models'/f'{n}.urdf'),pin.JointModelFreeFlyer()) for n in names]
homes=[np.array(json.loads((G/f'{n}_meta.json').read_text())['qhome']) for n in names]
terms=[cs.Function.load(str(G/f'{n}_terms.casadi')) for n in names]
steps=[cs.Function.load(str(G/f'{n}_step.casadi')) for n in names]
Jbase=np.c_[np.eye(6),np.zeros((6,6))]
C=np.block([[Jbase,-Jbase,np.zeros((6,12))],[Jbase,np.zeros((6,12)),-Jbase]])
peak=reciprocity=constraint=0.
for sample in range(24):
    matrices=[];free=[];xs=[];us=[];hs=[];ws=[]
    for i,m in enumerate(models):
        q=pin.integrate(m,homes[i],rng.normal(0,.05,12));v=rng.normal(0,.2,12)
        u=rng.normal(0,2,12 if i<2 else 6);d=m.createData()
        R=pin.Quaternion(q[3:7]).matrix();T=block_diag(R.T,R.T,np.eye(6))
        vp=T@v;tdv=np.r_[-R.T@np.cross(v[3:6],v[:3]),np.zeros(9)]
        Mp=pin.crba(m,d,q);hp=pin.nonLinearEffects(m,d,q,vp)
        M=T.T@Mp@T;h=T.T@(hp+Mp@tdv)
        pin.computeJointJacobians(m,d,q);pin.updateFramePlacements(m,d)
        force=np.r_[np.zeros(6),u[:6]]
        feet=[['FL_foot','FR_foot'],['RL_foot','RR_foot'],[]][i]
        for k,f in enumerate(feet):
            j=pin.getFrameJacobian(m,d,m.getFrameId(f),pin.LOCAL_WORLD_ALIGNED)
            jj=(j[:3]-pin.skew(np.array([0,0,-.032]))@j[3:])@T
            force+=jj.T@u[6+3*k:9+3*k]
        matrices.append(M);free.append(v+.01*np.linalg.solve(M,force-h))
        xs.append(np.r_[q,v]);us.append(u)
        H,w=terms[i](xs[-1],u);hs.append(np.array(H));ws.append(np.array(w).ravel())
    M=block_diag(*matrices);vfree=np.concatenate(free);MinvCt=np.linalg.solve(M,C.T)
    impulse=-np.linalg.solve(C@MinvCt,C@vfree)
    expected=vfree+MinvCt@impulse;actual=[]
    for i in range(3):
        p=np.r_[np.zeros(3),(sum(hs)-hs[i]).flatten(order='F'),sum(ws)-ws[i],np.zeros(6),xs[i],us[i],1.]
        actual.append(np.array(steps[i](xs[i],us[i],p)).ravel()[13:])
    actual=np.concatenate(actual)
    peak=max(peak,float(np.max(np.abs(actual-expected))))
    constraint=max(constraint,float(np.linalg.norm(C@actual)))
    impulses=(C.T@impulse).reshape(3,12)
    reciprocity=max(reciprocity,float(np.linalg.norm(impulses[:,:6].sum(axis=0))))
assert peak<1e-9 and constraint<1e-9 and reciprocity<1e-9

# Pairwise residual r_ij = v_i - v_j, antisymmetric scaled multipliers.
# Completing the square in +y'r requires (r+y/rho)^2, not (r-y/rho)^2.
target=np.array([-1.,2.,5.]);rho=2.;sigma=10.
def iterate(sign):
    v=np.zeros(3);edge=np.zeros((3,3));aggregate=np.zeros(3);algebra=0.
    for _ in range(400):
        nxt=np.empty(3)
        for i in range(3):
            other=[j for j in range(3) if j!=i]
            nxt[i]=(target[i]+rho*sum(v[j]-sign*edge[i,j] for j in other)+sigma*v[i])/(1+2*rho+sigma)
            reduced=(target[i]+2*rho*((sum(v)-v[i])/2-sign*aggregate[i])+sigma*v[i])/(1+2*rho+sigma)
            algebra=max(algebra,abs(nxt[i]-reduced))
        edge+=nxt[:,None]-nxt[None,:]
        aggregate+=1.5*(nxt-nxt.mean());v=nxt
    return float(np.max(np.abs(v-target.mean()))),algebra
correct,algebra=iterate(1);incorrect,_=iterate(-1)
assert correct<1e-9 and algebra<1e-12 and incorrect>1

# Three local solvers can each predict zero velocity using stale remote inputs,
# while the jointly applied inputs diverge in a common direction (Jacobi -2 mode).
u=np.ones(3)*1e-3
for _ in range(6):
    old=u.copy();u=-(old.sum()-old)
    predicted=(u+(old.sum()-old))/3
    assert np.linalg.norm(predicted[:,None]-predicted[None,:])<1e-14
assert abs(u.mean())>50e-3
print(json.dumps(dict(nonconsensus_samples=24,schur_vs_signed_kkt_max_abs=peak,
    interface_constraint_norm=constraint,net_interface_impulse_norm=reciprocity,
    edge_admm_solution_error=correct,aggregate_edge_update_error=algebra,
    wrong_sign_solution_error=incorrect,zero_consensus_common_mode_actual_velocity=float(u.mean())),indent=2))
