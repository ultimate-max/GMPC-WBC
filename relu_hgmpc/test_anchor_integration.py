import importlib.util,pathlib,json
import numpy as np
import pinocchio as pin
import native
from anchored_geometry import linearize,relative_arm,pulled_back_linearize,mount_transform
from anchored_mpc import AnchoredMPC
from run_anchor_comparison import Goal
from prioritized import PrioritizedController
from paper_anchor_baseline import EEReference
from geometric_wbc import rolling_contact

root=pathlib.Path(__file__).resolve().parents[1];p=native.Plant(str(root));s=p.truth();target=Goal(p,root,'normal')
model=pin.buildModelFromUrdf(str(root/'models/b2_z1.urdf'),pin.JointModelFreeFlyer());C=mount_transform(model,s['q'])
rng=np.random.default_rng(726);errs=[]
for _ in range(10):
    ss=p.evaluate(p.integrate(s['q'],rng.normal(size=24)*.03),rng.normal(size=24)*.1)
    D=pin.SE3(ss['pose'])*pin.exp6(rng.normal(size=6)*.2);vd=rng.normal(size=6)*.1
    ei,ji,bi=linearize(ss,C,D,vd);F=relative_arm(ss,C);ei=F.inverse().action@ei
    ja=np.zeros_like(ss['J']);ja[:,18:]=ss['J'][:,18:]
    explicit=(ei,F.inverse().action@ji+pin.Motion(ei).action@ja,F.inverse().action@bi)
    simplified=pulled_back_linearize(ss,C,D,vd)
    errs.extend(float(np.max(abs(a-b))) for a,b in zip(explicit,simplified))
assert max(errs)<1e-12
# Refactoring SNS into allocate() must preserve the original complete control.
spec=importlib.util.spec_from_file_location('old_paper',root/'relu_hgmpc/prioritized.pre_anchor.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
c0=old.PrioritizedController(p,EEReference(target),task_dim=6)
c1=PrioritizedController(p,EEReference(target),task_dim=6)
baseline=[]
for t in (0.,.02,.04):
    a=c0.control(rolling_contact(s),t);b=c1.control(rolling_contact(s),t)
    baseline.append(max(float(np.max(abs(x-y))) for x,y in zip(a,b)))
assert max(baseline)<1e-12
# Default WBC task order remains unchanged for existing experiments.
spec=importlib.util.spec_from_file_location('old_wbc',root/'relu_hgmpc/geometric_wbc.pre_anchor.py')
oldw=importlib.util.module_from_spec(spec);spec.loader.exec_module(oldw)
from geometric_wbc import GeometricWBC
w0=oldw.GeometricWBC(p,'se3');w1=GeometricWBC(p,'se3');legacy=[]
for t in (0.,.02,.04):
    aa=w0.control(s,target.at(t));bb=w1.control(s,target.at(t))
    legacy.append(max(float(np.max(abs(x-y))) for x,y in zip(aa,bb)))
assert max(legacy)<1e-12
# Anchor and direct formulations share the very same condensed QP and result.
from model import equilibrium
tau=equilibrium(s);a=AnchoredMPC(p,root,target,'anchor');b=AnchoredMPC(p,root,target,'direct')
pa=a.solve(s,.02,tau,np.zeros(24));pb=b.solve(s,.02,tau,np.zeros(24))
qp=max(float(np.max(abs(x-y))) for ls,rs in zip(a.hierarchy.last_levels,b.hierarchy.last_levels) for x,y in zip(ls,rs))
torques=float(np.max(abs(pa.torques-pb.torques)))
assert qp==0. and torques<1e-10
print(json.dumps(dict(pullback_equivalence=max(errs),baseline_refactor=max(baseline),legacy_wbc=max(legacy),qp_rows=qp,torque_equivalence=torques),indent=2))
