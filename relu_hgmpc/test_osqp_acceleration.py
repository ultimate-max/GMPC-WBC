"""Independent exact-elimination, same-QP solution, and pattern-growth checks."""
import pathlib,json
import numpy as np
import scipy.linalg as la
import native
from model import Reference
from paper_gmpc import ReferenceCache
from osqp_acceleration import AcceleratedGMPC,ContactReduction,PersistentOSQP,assemble

root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/osqp_acceleration';out.mkdir(parents=True,exist_ok=True)
p=native.Plant(str(root));home=p.truth();rng=np.random.default_rng(20261110)
records=[]
for scale in [1,2]:
    filename='aligned_paper_cached'+('_scale2' if scale==2 else '')+'_r1_trace.npz'
    tr=np.load(root/'results/paper_gmpc'/filename)
    ref=ReferenceCache(Reference(p,home['q'],home['pose'],scale),12.,.02,10,.03)
    controllers={v:AcceleratedGMPC(p,ref,root,mode='central',reuse_static=True,
        optimization=v,qp_tol=2e-11) for v in ['baseline','reuse','eliminate','combined']}
    for index in [0,80,240,490]:
        s=p.evaluate(tr['q'][index],tr['v'][index]);t=index*.02
        for c in controllers.values():
            if index:c.prev_tau=tr['tau'][index-1].copy()
        baseline=controllers['baseline'];parts=baseline.build(s,t);H,g,A,l,u=assemble(parts)
        red=ContactReduction(parts,H,g,A,l,u)
        # Test all free controls, not just an optimizer solution.
        z=rng.normal(size=len(red.gr))*.1;x=red.T@z+red.c
        assert max(red.equality_map_max,red.equality_offset_max)<1e-8
        constant=.5*red.c@H@red.c+g@red.c
        original=.5*x@H@x+g@x;reduced=.5*z@red.Hr@z+red.gr@z+constant
        objective_error=abs(original-reduced)/max(1.,abs(original));assert objective_error<1e-11
        bounderr=max(np.max(np.abs((A@x)[red.ineq]-(red.Ar@z+A[red.ineq]@red.c))),0.)
        assert bounderr<1e-8
        baseline.last=None;full,info=baseline.solve(parts);gold=np.r_[*full]
        row=dict(scale=scale,index=index,map_error=red.equality_map_max,offset_error=red.equality_offset_max,
                 objective_transform_error=objective_error,inequality_transform_error=float(bounderr),variants={})
        for variant,c in controllers.items():
            if variant=='baseline':continue
            # Same matrices, reference state, previous input and warm sequence.
            c.last=None;solution,metrics=c.solve(parts);got=np.r_[*solution]
            error=float(la.norm(got-gold,np.inf))
            torque_error=float(np.max(np.abs(np.r_[100*(got[:12]-gold[:12]),10*(got[300:306]-gold[300:306])])) )
            gap=float(abs((.5*got@H@got+g@got)-(.5*gold@H@gold+g@gold))/max(1.,abs(.5*gold@H@gold+g@gold)))
            assert torque_error<.005,(variant,index,torque_error)
            assert gap<1e-7,(variant,index,gap)
            row['variants'][variant]=dict(first_torque_difference_Nm=torque_error,x_max_difference=error,
                objective_relative_difference=gap,primal=metrics['qp_primal_relative'],
                dual=metrics['qp_dual_relative'],normal=metrics['qp_normal_relative'],
                setup_ms=metrics['setup_ms'],solve_ms=metrics['solve_ms'])
        records.append(row);print(json.dumps(row),flush=True)

# Explicitly exercise a structural zero becoming nonzero, followed by reuse.
engine=PersistentOSQP(1e-9);patterns=[]
for i in range(3):
    H=np.eye(3);H[0,1]=H[1,0]=0. if i==0 else .2
    A=np.eye(3);A[0,2]=0. if i==0 else .3
    g=np.array([-.1,.2,-.3]);l=-np.ones(3);u=np.ones(3)
    engine.prepare(H,g,A,l,u,np.zeros(3));x,y,info=engine.solve()
    assert np.max(np.abs(x+la.solve(H,g)))<1e-7
    patterns.append(dict(rebuilt=engine.rebuilt,growth=engine.pattern_growth))
assert [r['rebuilt'] for r in patterns]==[1,1,0],patterns
result=dict(reference_check_tolerance=2e-11,closed_loop_tolerance=2e-7,records=records,pattern_transition=patterns)
(out/'verification.json').write_text(json.dumps(result,indent=2))
print('All exact-QP and persistent-pattern checks passed')
