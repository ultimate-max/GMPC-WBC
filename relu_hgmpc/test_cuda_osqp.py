"""CUDA capability, active-bound QP and actual hierarchical-MPC replay audit."""
import pathlib,time,json,sys,hashlib,importlib.metadata as md
import numpy as np
import osqp
import osqp_cuda
import native
from central_relu import residuals
from osqp_acceleration import PersistentOSQP
from hierarchical_gmpc import HierarchicalGMPC,RESIDUAL_KEYS
from model import Reference
from paper_gmpc import ReferenceCache

root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/cuda_osqp';out.mkdir(parents=True,exist_ok=True)
report=dict(osqp=osqp.__version__,cuda_package=md.version('osqp-cu12'),algebras=osqp.algebras_available(),
    cuda_use_float=osqp_cuda.OSQP_USE_FLOAT,cuda_binary=str(osqp_cuda.__file__),
    cuda_binary_sha256=hashlib.sha256(pathlib.Path(osqp_cuda.__file__).read_bytes()).hexdigest(),records=[])
def save():(out/'verification.json').write_text(json.dumps(report,indent=2))
assert 'cuda' in report['algebras']
H=np.diag([2.,4.]);g=np.array([-4.,4.]);A=np.eye(2);l=np.array([0.,-.5]);u=np.array([1.,.5])
for algebra in ('builtin','cuda'):
    eng=PersistentOSQP(2e-7,algebra=algebra);eng.prepare(H,g,A,l,u,np.zeros(2));x,y,info=eng.solve()
    m=residuals(H,g,A,l,u,x,y)
    assert np.max(np.abs(x-np.array([1.,-.5])))<2e-6,(algebra,x)
    assert max(m[k] for k in RESIDUAL_KEYS)<=2e-7,m
    print('Analytic active-bound QP passed',algebra,x,info.iter,flush=True)
    report[algebra+'_analytic']=dict(x=x.tolist(),iterations=info.iter,solve_ms=eng.solve_wall_ms,residuals=m)
save()

class CapturingEngine(PersistentOSQP):
    def prepare(self,H,g,A,l,u,warm):
        self.problem=dict(H=H.copy(),g=g.copy(),A=A.copy(),l=l.copy(),u=u.copy(),warm=warm.copy())
        return super().prepare(H,g,A,l,u,warm)

p=native.Plant(str(root));home=p.truth()
for scale,index in [(1,0),(1,80),(1,200),(1,490),(2,200)]:
    ref=ReferenceCache(Reference(p,home['q'],home['pose'],scale),12.,.02,10,.03)
    trace=np.load(root/f'results/hierarchical_gmpc/hierarchy_null_scale{scale}_r1_trace.npz')
    s=p.evaluate(trace['q'][index],trace['v'][index]);t=index*.02;answers={}
    for algebra in ('builtin','cuda'):
        c=HierarchicalGMPC(plant=p,ref=ref,root=root,mode='central',N=10,dt=.03,backend='osqp',qp_tol=2e-7,reuse_static=True,algebra=algebra)
        c.primary_engine=CapturingEngine(2e-7,algebra=algebra);c.secondary_engine=CapturingEngine(2e-7,algebra=algebra)
        if index:c.prev_tau=trace['tau'][index-1].copy()
        parts=c.build(s,t);start=time.perf_counter()
        try:
            answer,m=c.solve(parts)
            assert max(m[k] for k in RESIDUAL_KEYS)<=2e-7
            assert m['task_lock_relative']<=2e-7
            answers[algebra]=c.last_secondary.copy()
            item=dict(scale=scale,index=index,algebra=algebra,passed=True,wall_ms=1000*(time.perf_counter()-start),**m)
        except Exception as e:
            item=dict(scale=scale,index=index,algebra=algebra,passed=False,wall_ms=1000*(time.perf_counter()-start),error=str(e))
            for level,eng in [('primary',c.primary_engine),('secondary',c.secondary_engine)]:
                if hasattr(eng,'last_result'):
                    raw=eng.last_result.info
                    item[level+'_last_status']=dict(status=raw.status,iterations=raw.iter,
                        primal=float(raw.prim_res),dual=float(raw.dual_res))
        for name,engine in [('primary',c.primary_engine),('secondary',c.secondary_engine)]:
            if hasattr(engine,'problem'):
                np.savez_compressed(out/f'replay_s{scale}_i{index}_{algebra}_{name}.npz',**engine.problem)
        c.close();report['records'].append(item);save()
        print(json.dumps({k:item[k] for k in ('scale','index','algebra','passed','wall_ms','error','qp_iterations') if k in item}),flush=True)
    if len(answers)==2:
        # Same first step physical torque scaling as the actual controls.
        def first_tau(x):
            n0=len(parts[0]['g']);return np.r_[parts[0]['scale'][:12]*x[:12],parts[1]['scale'][:6]*x[n0:n0+6]]
        report['records'][-1]['first_torque_difference_Nm']=float(np.max(np.abs(first_tau(answers['cuda'])-first_tau(answers['builtin']))));save()
print('Replay audit complete',flush=True)
