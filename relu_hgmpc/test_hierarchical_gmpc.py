"""Model parity, both-level KKT, and nontrivial primary-preserving motion."""
import pathlib,json,copy
import numpy as np
import native
from model import Reference
from paper_gmpc import PaperGMPC,ReferenceCache
from paper_gmpc_weighted_legacy import PaperGMPC as LegacyGMPC
from hierarchical_gmpc import HierarchicalGMPC,hierarchy_costs,RESIDUAL_KEYS
from osqp_acceleration import assemble,AcceleratedGMPC

root=pathlib.Path(__file__).resolve().parents[1]
out=root/'results/hierarchical_gmpc';out.mkdir(parents=True,exist_ok=True)
p=native.Plant(str(root));home=p.truth()
ref=ReferenceCache(Reference(p,home['q'],home['pose']),12.,.02,10,.03)
kwargs=dict(plant=p,ref=ref,root=root,mode='central',N=10,dt=.03,backend='osqp',qp_tol=2e-7,reuse_static=True)
old=LegacyGMPC(**kwargs);plain=PaperGMPC(**kwargs);c=HierarchicalGMPC(**kwargs)
trace=np.load(root/'results/osqp_acceleration/combined_scale1_r1_trace.npz')
records=[]
for index in [0,80,240,490]:
    s=p.evaluate(trace['q'][index],trace['v'][index]);t=index*.02
    for obj in [old,plain,c]:
        obj.prev_tau=trace['tau'][index-1].copy() if index else obj.prev_tau
        obj.last=None
    original=old.build(s,t);parts0=plain.build(s,t);parts=c.build(s,t)
    for before,after in [(original,parts0),(parts0,parts)]:
        for a,b in zip(assemble(before),assemble(after)):
            assert np.array_equal(a,b), 'Weighted baseline or physical model changed'
    H,g,A,l,u=assemble(parts);H1,g1,H2,g2,C,d,_=hierarchy_costs(parts)
    assert np.allclose(H,H1+H2,rtol=1e-13,atol=1e-10)
    assert np.allclose(g,g1+g2,rtol=1e-13,atol=1e-11)
    answer,info=c.solve(parts)
    assert max(info[k] for k in RESIDUAL_KEYS)<=c.qp_tol
    assert info['task_lock_relative']<=c.qp_tol
    assert info['secondary_objective_after']<=info['secondary_objective_before']+1e-5*max(1.,abs(info['secondary_objective_before']))
    records.append(dict(index=index,**info));print(json.dumps(records[-1]),flush=True)
    # Deliberately bias a low-priority linear objective; level 1 must not change.
    if index==80:
        C0=C.copy()
        changed=copy.deepcopy(parts)
        changed[1]['secondary_g']=changed[1]['secondary_g']+np.linspace(-.5,.5,len(changed[1]['g']))
        # Use identical fresh workspaces: retained adaptive-rho/scaling state
        # otherwise changes a finite-tolerance primary solve between calls.
        first=HierarchicalGMPC(**kwargs);biased=HierarchicalGMPC(**kwargs)
        first.solve(parts);ans2,info2=biased.solve(changed)
        assert np.array_equal(first.last_primary,biased.last_primary)
        assert info2['task_lock_relative']<=c.qp_tol
        assert np.linalg.norm(first.last_secondary-biased.last_secondary)>1e-5
        full=HierarchicalGMPC(**kwargs,lock_mode='all')
        _,full_info=full.solve(parts)
        assert np.array_equal(first.last_primary,full.last_primary)
        assert abs(full_info['secondary_objective_after']-first.hierarchy_data['secondary_metrics']['qp_objective'])<1e-5*max(1.,abs(full_info['secondary_objective_after']))
        assert np.allclose(first.hierarchy_data['C'],first.hierarchy_data['lock_lift']@first.hierarchy_data['lock_C'],rtol=1e-12,atol=1e-11)
        records.append(dict(index=index,full_lock_comparison=True,**full_info))
        first.close();biased.close();full.close()
        records.append(dict(index=index,secondary_bias_test=True,**info2))
        print('Secondary bias preserves the primary residual',flush=True)
(out/'verification.json').write_text(json.dumps(dict(records=records,weighted_matrix_parity='bitwise',
    physical_constraints='unchanged',tolerance=c.qp_tol),indent=2))
for obj in [old,plain,c]:obj.close()
print('Hierarchy checks passed',flush=True)
