"""Separate finite-tolerance solver effects from changes to the physical OCP."""
import pathlib,json
import numpy as np
import scipy.linalg as la
import native
from model import Reference
from paper_gmpc import ReferenceCache
from osqp_acceleration import AcceleratedGMPC,assemble

root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/osqp_acceleration'
p=native.Plant(str(root));home=p.truth();records=[]
for scale in [1,2]:
    path=root/'results/paper_gmpc'/('aligned_paper_cached'+('_scale2' if scale==2 else '')+'_r1_trace.npz')
    tr=np.load(path);ref=ReferenceCache(Reference(p,home['q'],home['pose'],scale),12.,.02,10,.03)
    gold=AcceleratedGMPC(p,ref,root,mode='central',reuse_static=True,optimization='eliminate',qp_tol=2e-12)
    ctrls={m:AcceleratedGMPC(p,ref,root,mode='central',reuse_static=True,optimization=m,qp_tol=2e-7)
           for m in ['baseline','reuse','eliminate','combined']}
    for index in [80,240,490]:
        s=p.evaluate(tr['q'][index],tr['v'][index]);t=index*.02
        for c in [gold,*ctrls.values()]:c.prev_tau=tr['tau'][index-1].copy();c.last=None
        parts=gold.build(s,t);H,g,A,l,u=assemble(parts)
        answer,gi=gold.solve(parts);xg=np.r_[*answer]
        row=dict(scale=scale,index=index,reference_dual=gi['qp_dual_relative'],variants={})
        for name,c in ctrls.items():
            x,info=c.solve(parts);x=np.r_[*x]
            diff=np.r_[100*(x[:12]-xg[:12]),10*(x[300:306]-xg[300:306])]
            row['variants'][name]=dict(first_torque_difference_Nm=float(np.max(np.abs(diff))),
                primal=info['qp_primal_relative'],dual=info['qp_dual_relative'],normal=info['qp_normal_relative'],
                objective_relative_difference=float(abs((.5*x@H@x+g@x)-(.5*xg@H@xg+g@xg))/max(1.,abs(.5*xg@H@xg+g@xg))),
                iterations=info['qp_iterations'])
        records.append(row)
result=dict(scope='Identical replay states and physical QP; cold primal/dual warm start for each state. Persistent workspace keeps numerical state. This is a finite-accuracy diagnostic, not a controller speed benchmark.',
            reference_tolerance=2e-12,runtime_tolerance=2e-7,records=records)
(out/'accuracy_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
