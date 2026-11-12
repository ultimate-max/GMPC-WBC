"""Read-only replay: is the equality-constrained QP minimizer fully feasible?

This diagnoses an exact fast path. It does NOT change or run the controller.
"""
import json,pathlib,time
import numpy as np
import scipy.linalg as la
import native
from model import Reference,equilibrium
from split_gmpc import SplitGMPC
from central_relu import residuals

root=pathlib.Path(__file__).resolve().parents[1];p=native.Plant(str(root));s=p.truth()
ref=Reference(p,s['q'],s['pose']);c=SplitGMPC(p,ref,root,mode='central')
results=[]
for backend in ['osqp','reluqp']:
    c.prev_tau=equilibrium(s)
    z=np.load(root/f'results/split_gmpc/central_{backend}_12s_r1_trace.npz')
    rows=[]
    for index in np.linspace(0,len(z['q'])-1,60,dtype=int):
        state=p.evaluate(z['q'][index],z['v'][index])
        if index:c.prev_tau=z['tau'][index-1].copy()
        start=time.perf_counter();parts=c.build(state,index*.02);a,b=parts
        H=la.block_diag(a['H'],b['H']);g=np.r_[a['g'],b['g']]
        A=np.vstack((la.block_diag(a['A'],b['A']),np.column_stack((a['E'],-b['E']))))
        l=np.r_[a['l'],b['l'],b['d']-a['d']];u=np.r_[a['u'],b['u'],b['d']-a['d']]
        build_ms=1000*(time.perf_counter()-start);start=time.perf_counter()
        eq=np.isfinite(l)&np.isfinite(u)&(l==u);E=A[eq];be=l[eq]
        scale=1/np.maximum(1.,np.linalg.norm(E,axis=1));Es=scale[:,None]*E;bs=scale*be
        Q,R,piv=la.qr(Es.T,mode='full',pivoting=True)
        rank=int(np.sum(np.abs(np.diag(R))>1e-11*max(1.,np.max(np.abs(np.diag(R))))))
        xe=Q[:,:rank]@la.solve_triangular(R[:rank,:rank].T,bs[piv[:rank]],lower=True)
        Z=Q[:,rank:];Hr=Z.T@H@Z
        x=xe-Z@la.solve((Hr+Hr.T)/2,Z.T@(H@xe+g),assume_a='pos')
        lm=np.zeros(np.sum(eq));lm[piv[:rank]]=la.solve_triangular(R[:rank,:rank],Q[:,:rank].T@(-H@x-g))
        dual=np.zeros(len(l));dual[eq]=scale*lm
        res=residuals(H,g,A,l,u,x,dual)
        cost_ms=1000*(time.perf_counter()-start)
        ai=A[~eq]@x;violation=float(max(0,np.max(l[~eq]-ai),np.max(ai-u[~eq])))
        passed=violation<=1e-9 and max(res[k] for k in ['qp_primal_relative','qp_dual_relative','qp_normal_relative'])<=2e-7
        rows.append(dict(time=float(index*.02),passed=bool(passed),inequality_violation=violation,
            build_ms=build_ms,solve_and_check_ms=cost_ms,**res))
    results.append(dict(backend=backend,samples=len(rows),feasible_count=sum(r['passed'] for r in rows),
        max_inequality_violation=max(r['inequality_violation'] for r in rows),
        max_common_dual=max(r['qp_dual_relative'] for r in rows),
        build_mean_ms=float(np.mean([r['build_ms'] for r in rows])),
        solve_check_mean_ms=float(np.mean([r['solve_and_check_ms'] for r in rows])),rows=rows))
result=dict(replay_only=True,results=results,scope='120 stored states, current gentle stance trajectory; '
    'not a new closed-loop run or worst-case runtime guarantee. Physical controller is unchanged.')
out=root/'results/central_relu/acceleration_audit.json';out.write_text(json.dumps(result,indent=2))
print(json.dumps([{k:v for k,v in r.items() if k!='rows'} for r in results],indent=2))
