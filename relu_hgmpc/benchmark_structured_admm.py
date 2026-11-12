"""Paired replay of immutable, original GMPC QPs; no closed-loop deployment.

All acceptance tests are evaluated on the original H,g,G,l,u. Common measured
construction time is added once per problem (and once per hierarchy snapshot).
No reference solution is used as an initial guess. Disk I/O is outside timing.
"""
import argparse
import hashlib
import json
import os
import pathlib
import sys
import time
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import osqp

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'relu_hgmpc/structured_admm_native/build'))
from central_relu import residuals
from online_osqp import OnlineOSQP
from shared_hierarchy import HierarchyQP
import structured_admm_native as native_admm

KEYS = ('qp_primal_relative', 'qp_dual_relative', 'qp_normal_relative')
TOL=2e-7

def certify(qp, x, y, tol=None):
    if tol is None:tol=TOL
    rr = residuals(qp['H'], qp['g'], qp['G'], qp['lo'], qp['hi'], x, y)
    rr['accepted'] = bool(np.all(np.isfinite(x)) and np.all(np.isfinite(y))
        and max(rr[k] for k in KEYS) <= tol and rr['qp_primal_raw'] <= tol)
    ref = qp['reference_x']
    value = .5*ref@qp['H']@ref + qp['g']@ref
    rr['objective_reference'] = float(value)
    rr['objective_gap_absolute'] = float(abs(rr['qp_objective']-value))
    rr['objective_gap_relative'] = float(abs(rr['qp_objective']-value)/max(1., abs(value)))
    if 'task_C' in qp and qp['task_C'].ndim==2 and qp['task_C'].shape[1]==len(x) and qp['task_C'].shape[0]:
        rr['task_output_reference_difference'] = float(np.max(np.abs(qp['task_C']@(x-ref))))
    return rr

def dense_csc(a, upper=False):
    nrows, ncols = a.shape
    if upper:
        indices = np.concatenate([np.arange(j+1) for j in range(ncols)])
        indptr = np.r_[0, np.cumsum(np.arange(1, ncols+1))]
        data = np.concatenate([a[:j+1,j] for j in range(ncols)])
    else:
        indices = np.tile(np.arange(nrows),ncols)
        indptr = np.arange(ncols+1)*nrows
        data = a.ravel(order='F')
    return sp.csc_matrix((data,indices,indptr),shape=a.shape)

class FullOSQP:
    """Persistent full-constraint OSQP; numerical updates are always timed."""
    def __init__(self,preconditioned=False): self.workspaces={};self.preconditioned=preconditioned
    def solve(self, qp):
        start=time.perf_counter()
        H,g,G,lo,hi=[qp[k] for k in ('H','g','G','lo','hi')]
        cost_scale=1.;transform=None;coordinate_factor_ms=0.;coordinate_transform_ms=0.
        if self.preconditioned:
            meta=json.loads(str(qp.get('_metadata','{}')))
            shift=0. if meta.get('family')=='weighted' or meta.get('level')==2 else 1e-6
            cost_scale=max(1.,np.max(np.abs(H)),np.max(np.abs(g)))
            hs=H/cost_scale;gs=g/cost_scale
            ct=time.perf_counter();L=la.cholesky(hs+shift*np.eye(len(g)),lower=True)
            coordinate_factor_ms=1000*(time.perf_counter()-ct);ct=time.perf_counter()
            transform=la.solve_triangular(L.T,np.eye(len(g)),lower=False)
            H=transform.T@hs@transform;H=(H+H.T)*.5
            g=transform.T@gs;G=G@transform
            coordinate_transform_ms=1000*(time.perf_counter()-ct)
        norms=np.linalg.norm(G,axis=1)
        scale=np.ones(len(lo));nz=norms>1e-10;scale[nz]=1./norms[nz]
        P=dense_csc(H,True); A=dense_csc(G*scale[:,None])
        low=lo*scale; upp=hi*scale
        prep_ms=1000*(time.perf_counter()-start)
        ts=time.perf_counter(); key=(len(g),len(lo)); reused=key in self.workspaces
        if not reused:
            s=osqp.OSQP(algebra='builtin')
            eps0=min(1e-8,TOL)
            s.setup(P=P,q=g,A=A,l=low,u=upp,eps_abs=eps0,eps_rel=eps0,
                max_iter=20000,check_termination=10,polishing=True,
                warm_starting=True,verbose=False,adaptive_rho=True,
                eps_prim_inf=1e-30,eps_dual_inf=1e-30)
            self.workspaces[key]=s
        else:
            s=self.workspaces[key];s.update(Px=P.data,Ax=A.data,q=g,l=low,u=upp)
            s.update_settings(eps_abs=min(1e-8,TOL),eps_rel=min(1e-8,TOL),max_iter=20000)
        # Shared deterministic cold start: reduced hierarchy bases can rotate.
        s.warm_start(x=np.zeros(len(g)),y=np.zeros(len(lo)))
        update_ms=1000*(time.perf_counter()-ts)
        iterations=0; refine=0; solve_ms=0.; cert_ms=0.; reported_solve=0.;reported_polish=0.;rho_updates=0
        for eps in (min(1e-8,TOL),min(1e-9,TOL*.1),min(1e-10,TOL*.01)):
            if refine:s.update_settings(eps_abs=eps,eps_rel=eps)
            ts=time.perf_counter();sol=s.solve(raise_error=False)
            solve_ms+=1000*(time.perf_counter()-ts);iterations+=sol.info.iter
            reported_solve+=1000*sol.info.solve_time;reported_polish+=1000*sol.info.polish_time
            rho_updates+=int(sol.info.rho_updates)
            ts=time.perf_counter();x=sol.x if transform is None else transform@sol.x;y=sol.y*scale*cost_scale
            rr=certify(qp,x,y);cert_ms+=1000*(time.perf_counter()-ts)
            if rr['accepted'] or sol.info.status_val not in (1,2,7):break
            refine+=1
        return dict(x=x,y=y,iterations=int(iterations),status=sol.info.status,
            preprocess_ms=prep_ms,setup_update_ms=update_ms,iteration_ms=solve_ms,
            coordinate_factor_ms=coordinate_factor_ms,coordinate_transform_ms=coordinate_transform_ms,
            certificate_ms=cert_ms,solver_wall_ms=1000*(time.perf_counter()-start),
            refinements=refine,workspace_reused=reused,osqp_rho_updates=rho_updates,
            osqp_reported_setup_ms=1000*sol.info.setup_time,
            osqp_reported_update_ms=1000*sol.info.update_time,
            osqp_reported_solve_ms=reported_solve,
            osqp_reported_polish_ms=reported_polish,**rr)

class NativeADMM:
    def __init__(self,rho=.1,adaptive=50,preconditioner=-1.):
        self.rho=rho;self.adaptive=adaptive;self.preconditioner=preconditioner
    def solve(self,qp):
        start=time.perf_counter()
        setting={}
        if self.preconditioner!=-1.:
            meta=json.loads(str(qp.get('_metadata','{}')))
            setting['preconditioner_shift']=(0. if meta.get('family')=='weighted' or meta.get('level')==2 else 1e-6) if self.preconditioner==-2. else self.preconditioner
        ans=native_admm.solve(qp['H'],qp['g'],qp['G'],qp['lo'],qp['hi'],
            tol=TOL,primal_abs_tol=TOL,max_iter=20000,rho=self.rho,
            adaptive_interval=self.adaptive,**setting)
        ans['kernel_total_ms']=ans.pop('total_ms')
        ts=time.perf_counter();rr=certify(qp,ans['x'],ans['y'])
        cert_ms=1000*(time.perf_counter()-ts)
        return dict(ans,**dict(certificate_ms=cert_ms,
            solver_wall_ms=1000*(time.perf_counter()-start),**rr))

class ScreenedOSQP:
    """Existing production shortcut, only applicable to SPD QPs."""
    def __init__(self):self.workspaces={}
    def solve(self,qp):
        start=time.perf_counter();key=len(qp['g'])
        if key not in self.workspaces:self.workspaces[key]=OnlineOSQP(screen=True,tol=TOL)
        s=self.workspaces[key]
        # Same no-prior-solution policy as both kernels; this isolates speed of
        # the existing certified screening without leaking reference active sets.
        s.last=None;s.previous_active=np.array([],int);s.last_dual=None
        x,y,it,info=s.solve(qp['H'],qp['g'],qp['G'],qp['lo'],qp['hi'],0.,1.,1)
        ts=time.perf_counter();rr=certify(qp,x,y)
        return dict(x=x,y=y,iterations=it,status='solved',**info,**rr,
            certificate_ms=1000*(time.perf_counter()-ts),solver_wall_ms=1000*(time.perf_counter()-start))

class ExistingHierarchy:
    """Current per-level implementation on the same frozen reference path.

    First two levels: certified least squares or quadprog + exact-objective
    refinement. Last level: certified unconstrained solve or screened OSQP.
    This is explicitly a hybrid baseline, not claimed to be pure OSQP.
    """
    def __init__(self):self.base=HierarchyQP();self.base.qp_tol=TOL;self.screen=ScreenedOSQP()
    def solve(self,qp):
        start=time.perf_counter();C,d=qp['task_C'],qp['task_d']
        meta=json.loads(str(qp['_metadata']))
        if meta['level']==2:
            x=la.solve(C.T@C,-C.T@d,assume_a='pos')
        else:x=la.lstsq(C,-d,cond=1e-10,lapack_driver='gelsy')[0]
        y=np.zeros(len(qp['lo']));rr=certify(qp,x,y);fast=False;count=0;details={}
        if max(rr[k] for k in KEYS)<1e-10 and rr['qp_primal_raw']<1e-9:fast=True
        elif meta['level']==2:
            details=self.screen.solve(qp);x=details['x'];y=details['y'];count=details['iterations']
        else:
            G=qp['G'].copy();lo=qp['lo'].copy();hi=qp['hi'].copy()
            zero=la.norm(G,axis=1)<1e-10
            if np.any(zero) and max(0.,np.max(lo[zero]),np.max(-hi[zero]))>2e-7:
                raise RuntimeError('Inherited row infeasible')
            G[zero]=0;lo[zero]=-np.inf;hi[zero]=np.inf
            lo[(lo>0)&(lo<2e-7)]=0;hi[(hi<0)&(hi>-2e-7)]=0
            x,_,count=self.base.solve_level(meta['level'],C,d,G,lo,hi,np.zeros(len(qp['g'])))
            y=self.base.dual_by_level[meta['level']]
        rr=certify(qp,x,y)
        return dict(x=x,y=y,iterations=count,status='solved',**rr,
            active_face_fastpath=int(fast),solver_wall_ms=1000*(time.perf_counter()-start))

def quantiles(values):
    a=np.array(values,float)
    return dict(mean=float(np.mean(a)),median=float(np.median(a)),p95=float(np.quantile(a,.95)),max=float(np.max(a))) if len(a) else {}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--corpus',default='results/structured_admm/qp_corpus')
    ap.add_argument('--output',default='results/structured_admm/replay')
    ap.add_argument('--repeat',type=int,default=3)
    ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--family',choices=['all','weighted','hierarchy'],default='all')
    ap.add_argument('--split',choices=['all','calibration','heldout'],default='all')
    ap.add_argument('--solvers',default='osqp_full,structured_admm,osqp_screened')
    ap.add_argument('--rho',type=float,default=.1)
    ap.add_argument('--adaptive',type=int,default=50)
    ap.add_argument('--preconditioner',type=float,default=-1.)
    ap.add_argument('--tol',type=float,default=2e-7)
    args=ap.parse_args();global TOL;TOL=args.tol
    folder=ROOT/args.corpus;out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'records.jsonl').exists():raise RuntimeError('Output already has records; choose a new output directory')
    manifest=json.loads((folder/'manifest.json').read_text())
    entries=manifest.get('entries',manifest.get('problems',manifest.get('qps',[])))
    entries=[e for e in entries if (args.family=='all' or e['family']==args.family)
        and (args.split=='all' or e['split']==args.split)]
    if args.limit:entries=entries[:args.limit]
    if not entries:raise RuntimeError('No QPs selected')
    cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu})
    batches=[]
    for e in entries:
        path=pathlib.Path(e['path']);path=path if path.is_absolute() else folder/path
        with np.load(path,allow_pickle=False) as f:qp={k:f[k] for k in f.files if k!='meta_json'}
        from capture_structured_qps import digest_arrays
        assert digest_arrays({k:qp[k] for k in ('H','g','G','lo','hi')})==e['qp_sha256']
        qp['_metadata']=np.array(json.dumps(e))
        batches.append((e,qp))
    selected=args.solvers.split(',');rows=[]
    for rep in range(args.repeat):
        solvers={'osqp_full':FullOSQP(),'osqp_preconditioned':FullOSQP(True),
            'structured_admm':NativeADMM(args.rho,args.adaptive,args.preconditioner),
            'osqp_screened':ScreenedOSQP(),'existing_hierarchy':ExistingHierarchy()}
        # One tiny unrelated warmup, with no benchmark matrix/factor preloaded.
        warm=dict(H=np.eye(2),g=np.array([-2.,1.]),G=np.eye(2),lo=-np.ones(2),hi=np.ones(2),reference_x=np.array([1.,-1.]))
        for name in selected:
            try:solvers[name].solve(warm)
            except Exception:pass
        for index,(e,qp) in enumerate(batches):
            order=selected[(index+rep)%len(selected):]+selected[:(index+rep)%len(selected)]
            for name in order:
                row=dict(repeat=rep,solver=name,**e)
                if name=='existing_hierarchy' and e['family']!='hierarchy':
                    row.update(status='not applicable: weighted problem',applicable=False)
                elif name=='osqp_screened' and e['family']=='hierarchy' and e.get('level',0)<2:
                    row.update(status='not applicable: semidefinite hierarchy objective',applicable=False)
                else:
                    begin=time.perf_counter()
                    try:
                        res=solvers[name].solve(qp)
                        row.update({k:v for k,v in res.items() if k not in ('x','y')})
                        row['applicable']=True
                        # The corpus stores solver-independent construction only.
                        row['total_ms']=e.get('state_evaluation_ms',0.)+e.get('construction_ms',0.)+row['solver_wall_ms']
                    except Exception as exc:
                        row.update(status=f'exception: {exc}',accepted=False,applicable=True,
                            solver_wall_ms=1000*(time.perf_counter()-begin))
                rows.append(row)
                with (out/'records.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                print(rep,index,name,e['family'],e.get('level',''),row.get('accepted'),
                    round(row.get('solver_wall_ms',0),3),row.get('iterations'),flush=True)
    summary={}
    for family in sorted(set(e['family'] for e in entries)):
        for name in selected:
            rr=[r for r in rows if r['solver']==name and r['family']==family and r.get('applicable')]
            ok=[r for r in rr if r.get('accepted')]
            summary[family+'/'+name]=dict(count=len(rr),accepted=len(ok),failed=len(rr)-len(ok),
                solver_wall_ms=quantiles([r['solver_wall_ms'] for r in ok]),
                total_ms=quantiles([r['total_ms'] for r in ok]),
                iterations=quantiles([r['iterations'] for r in ok]),
                max_original_residual=max([max(r[k] for k in KEYS) for r in ok],default=None),
                max_raw_violation=max([r['qp_primal_raw'] for r in ok],default=None),
                max_objective_gap=max([r['objective_gap_relative'] for r in ok],default=None))
    cycles=[]
    for rep in range(args.repeat):
        for name in selected:
            for snapshot in sorted(set(e['snapshot_id'] for e in entries)):
                for family in sorted(set(e['family'] for e in entries)):
                    rr=[r for r in rows if r['repeat']==rep and r['solver']==name and r['snapshot_id']==snapshot
                        and r['family']==family and r.get('applicable')]
                    expected=3 if family=='hierarchy' else 1
                    if len(rr)!=expected:continue
                    common=rr[0].get('state_evaluation_ms',0.)+rr[0]['snapshot_common_construction_ms']
                    if family=='hierarchy':
                        common+=sum(r.get('level_construction_ms',0.)+r.get('hierarchy_lock_construction_ms',0.) for r in rr)
                    ok=all(r.get('accepted') for r in rr)
                    cycles.append(dict(repeat=rep,solver=name,snapshot_id=snapshot,family=family,
                        case=rr[0]['case'],split=rr[0]['split'],accepted=ok,
                        common_construction_ms=common,solver_ms=sum(r['solver_wall_ms'] for r in rr),
                        total_ms=common+sum(r['solver_wall_ms'] for r in rr)))
    cycle_summary={}
    for family in sorted(set(e['family'] for e in entries)):
        for name in selected:
            cr=[c for c in cycles if c['family']==family and c['solver']==name]
            ok=[c for c in cr if c['accepted']]
            cycle_summary[family+'/'+name]=dict(count=len(cr),accepted=len(ok),
                total_ms=quantiles([c['total_ms'] for c in ok]),
                solver_ms=quantiles([c['solver_ms'] for c in ok]),
                attempted_total_ms=quantiles([c['total_ms'] for c in cr]))
    result=dict(args=vars(args),cpu_affinity=cpu,summary=summary,records=rows,
        cycles=cycles,cycle_summary=cycle_summary,
        versions=dict(numpy=np.__version__,osqp=osqp.__version__),
        native_build=dict(blas_enabled=bool(native_admm.blas_enabled),native_arch_enabled=bool(native_admm.native_arch_enabled)),
        osqp_infeasibility_thresholds=dict(eps_prim_inf=1e-30,eps_dual_inf=1e-30,
            reason='Inherited hierarchy rows can contain 1e-15 roundoff inconsistency. Prevent an exact-infeasibility certificate from terminating a solve accepted to common physical tolerance. Original matrices and bounds remain unchanged; all outputs require independent original-QP certification.'),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [pathlib.Path(__file__),ROOT/'relu_hgmpc/structured_admm_native/solver.cpp']},
        corpus_manifest_sha256=hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest(),
        tolerances=dict(original_primal_dual_normal=TOL,physical_primal_abs=TOL),
        timing='Immutable QP replay. total_ms adds common measured matrix construction, never reference solve or disk I/O. Solver wall includes input conversion, scaling, numeric factorization, iterations, adapt/refactor, polish where enabled, and independent original-QP certification. It does not include MPC trajectory recovery, WBC, or simulation.',
        warm_start='Zero primal/dual for all kernels; existing screened path reset. OSQP workspace, symbolic structure and adapted rho may persist by dimensions; numerical values always updated. Native rho starts from configured value on each QP. No reference solution/active-set leak.',
        gate='No closed-loop deployment unless every held-out QP passes common accuracy and end-to-end QP compute improves against optimized existing baseline.')
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(cycle_summary,indent=2),flush=True)

if __name__=='__main__':main()
