"""Offline two-block ADMM probe of the FINAL WBC priority level only.

Higher-priority task outputs come from the certified centralized controller.
This isolates interface splitting with identical constraints and a known
feasible hierarchy. It is NOT a complete distributed hierarchy or closed loop.
Local OSQP factors are reused. No new simulation or concurrent timing work.
"""
import json,pathlib,time
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import osqp
import native
from geometric_wbc import pose,rolling_contact
from dual_wbc import CoordinatedWBC,inf

root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/dual_wbc'
def stack(cost):return np.vstack([a for a,b in cost]),np.concatenate([b for a,b in cost])

def assemble(ctrl,s,r):
    start=time.perf_counter();s=rolling_contact(s)
    b=ctrl.body.dynamics(s);a=ctrl.arm.dynamics(s)
    scale=ctrl.scale;ws=np.r_[np.full(3,100.),np.full(3,10.)];asc=np.full(6,10.)
    Ab=np.c_[b.T*scale[:12],b.W*ws];Aa=np.c_[a.T*scale[12:],a.W*ws]
    Bb=np.c_[b.F*scale[:12],b.FW*ws];Ub=np.c_[np.eye(12),np.zeros((12,6))];Ua=np.c_[np.eye(6),np.zeros((6,6))]
    gb,lb,ub=ctrl.body.joint_bounds(s,Ab,b.a0,Ub*scale[:12,None],ctrl.prev_tau,ctrl.dt)
    ga,ll,ua=ctrl.arm.joint_bounds(s,Aa,a.a0,Ua*scale[12:,None],ctrl.prev_tau,ctrl.dt)
    gb=np.vstack((gb,ctrl.cone@Bb));lb=np.r_[lb,np.tile([-np.inf]*4+[10.],4)-ctrl.cone@b.f0]
    ub=np.r_[ub,np.tile([0.]*4+[800.],4)-ctrl.cone@b.f0]
    eta=ctrl.joint_damping;h=ctrl.dt
    def damp(G,lo,hi,A,bias,joints):
        q=s['q'][7:][joints];v=s['v'][6:][joints];c=h+.5*eta*h*h
        return np.vstack((G,c*A[6:])),np.r_[lo,-eta*(q-s['q_min'][joints]-.02)-(1+eta*h)*v-c*bias[6:]],np.r_[hi,eta*(s['q_max'][joints]-.02-q)-(1+eta*h)*v-c*bias[6:]]
    gb,lb,ub=damp(gb,lb,ub,Ab,b.a0,np.arange(12));ga,ll,ua=damp(ga,ll,ua,Aa,a.a0,np.arange(12,18))
    target_a=r['a']+100*ctrl.plant.difference(s['q'],r['q'])+20*(r['v']-s['v'])
    cb=ctrl.body.smooth(Ab,b.a0,Ub,target_a,ctrl.prev_tau/scale,r['tau']/scale,ctrl.torque_smoothing)
    ca=ctrl.arm.smooth(Aa,a.a0,Ua,target_a,ctrl.prev_tau/scale,r['tau']/scale,ctrl.torque_smoothing)
    cb.append((.1*Bb,.1*(b.f0-r['force'])))
    Cb,db=stack(cb);Ca,da=stack(ca)
    # Both copies store force ON BODY. Arm's dynamics already use its minus.
    EB=np.vstack(((b.Ji@Ab)/asc[:,None],np.c_[np.zeros((6,12)),np.eye(6)]))
    EA=np.vstack((-(a.Ji@Aa)/asc[:,None],np.c_[np.zeros((6,6)),-np.eye(6)]))
    rhs=np.r_[(a.Ji@a.a0-b.Ji@b.a0)/asc,np.zeros(6)]
    Ct,dt,_=ctrl.body.task(s,r,Ab,b.a0);Ce,de,_=ctrl.arm.task(s,r,Aa,a.a0,ctrl.C)
    return dict(body=(Cb,db,gb,lb,ub,EB,Ct),arm=(Ca,da,ga,ll,ua,EA,Ce),c=rhs,
                wscale=ws,ascale=asc,assembly_ms=1000*(time.perf_counter()-start))

class LocalQP:
    def __init__(self,data,lock,rho,cost_scale):
        C,d,G,lo,hi,E,L=data
        # Eliminate the exact task-output locks before the iterative solve.
        # These tiny equalities are ill-conditioned if left as opposing bounds.
        self.x0=la.lstsq(L,lock,cond=1e-12)[0];self.N=la.null_space(L,rcond=1e-12)
        d=d+C@self.x0;C=C@self.N
        self.H=2*C.T@C/cost_scale;self.g=2*C.T@d/cost_scale
        self.Econst=E@self.x0;self.E=E@self.N
        self.rawG=G@self.N;self.rawlo=lo-G@self.x0;self.rawhi=hi-G@self.x0
        keep=la.norm(self.rawG,axis=1)>1e-10
        assert max(0.,np.max(self.rawlo[~keep],initial=0),np.max(-self.rawhi[~keep],initial=0))<2e-7
        self.rawG=self.rawG[keep];self.rawlo=self.rawlo[keep];self.rawhi=self.rawhi[keep]
        norm=np.maximum(la.norm(self.rawG,axis=1),1e-12);self.G=self.rawG/norm[:,None]
        self.lo=self.rawlo/norm;self.hi=self.rawhi/norm
        self.qp=osqp.OSQP();self.qp.setup(P=sp.csc_matrix(np.triu(self.H+rho*self.E.T@self.E)),q=self.g,
            A=sp.csc_matrix(self.G),l=self.lo,u=self.hi,verbose=False,
            eps_abs=1e-9,eps_rel=1e-9,max_iter=10000,polishing=False,check_termination=10)
        self.norm=norm;self.last=None
    def solve(self,offset,rho):
        self.qp.update(q=self.g+rho*self.E.T@(offset+self.Econst));result=self.qp.solve(raise_error=False)
        if result.info.status_val not in (1,2):raise RuntimeError(result.info.status)
        self.last=result;return self.x0+self.N@result.x
    def kkt(self,x,dual):
        x=self.N.T@(x-self.x0)
        y=self.last.y/self.norm
        hx=self.H@x;ex=self.E.T@dual;gx=self.rawG.T@y
        return inf(hx+self.g+ex+gx)/max(1.,inf(hx),inf(self.g),inf(ex),inf(gx))

def run_case(case,t,rho,maxiter=1000):
    tr=np.load(out/f'dual_{case}_damped_trace.npz');k=round(t/.005)
    p=native.Plant(str(root));p.fast_reference=True;ctrl=CoordinatedWBC(p,root,'dual',joint_damping=5.)
    s=p.evaluate(tr['q'][k],tr['v'][k]);q,v,a=[tr[f'reference_{x}'][k] for x in ('q','v','a')];rs=p.evaluate(q,v)
    r=dict(q=q,v=v,a=a,base_pose=pose(q).homogeneous,base_twist=v[:6],base_acc=a[:6],
        ee_pose=rs['pose'],ee_twist=rs['J']@v,ee_acc=rs['J']@a+rs['jdotv'],tau=tr['reference_tau'][k],force=tr['reference_force'][k])
    prev=tr['tau'][k-1].copy();ctrl.prev_tau=prev.copy()
    tc,ac=ctrl.control(s,r);central_ms=ctrl.info['controller_ms'];problem=ctrl.last_problem
    wc=ctrl.interface['wrench'].copy();ctrl.prev_tau=prev.copy();blocks=assemble(ctrl,s,r)
    xstar=np.r_[tc[:12]/ctrl.scale[:12],wc/blocks['wscale']]
    zstar=np.r_[tc[12:]/ctrl.scale[12:],wc/blocks['wscale']]
    EB=blocks['body'][5];EA=blocks['arm'][5];c=blocks['c']
    assert inf(EB@xstar+EA@zstar-c)<1e-8
    # Lifting equivalence checks include inequalities and the exact smooth cost.
    residuals=[]
    for data,xx in ((blocks['body'],xstar),(blocks['arm'],zstar)):
        C,d,G,l,h,E,L=data
        assert max(0.,np.max(l-G@xx),np.max(G@xx-h))<2e-5
        residuals.append(C@xx+d)
    full_res=problem['tasks'][2][0]@(tc/ctrl.scale)+problem['tasks'][2][1]
    assert abs(sum(x@x for x in residuals)-full_res@full_res)<1e-6*max(1.,full_res@full_res)
    stamp=time.perf_counter()
    scale=max(1.,*(inf(2*data[0].T@data[0]) for data in (blocks['body'],blocks['arm'])))
    b=LocalQP(blocks['body'],blocks['body'][6]@xstar,rho,scale)
    ar=LocalQP(blocks['arm'],blocks['arm'][6]@zstar,rho,scale)
    setup_ms=1000*(time.perf_counter()-stamp)
    # Warm start from previous CONTROL, not the centralized optimal decision.
    wm=problem['maps']['W']@prev+problem['maps']['w0']
    x=np.r_[prev[:12]/ctrl.scale[:12],wm/blocks['wscale']];z=np.r_[prev[12:]/ctrl.scale[12:],wm/blocks['wscale']]
    b.qp.warm_start(x=b.N.T@(x-b.x0));ar.qp.warm_start(x=ar.N.T@(z-ar.x0));u=np.zeros(12);history=[];converged=False;failure=None
    stamp=time.perf_counter()
    try:
        for it in range(1,maxiter+1):
            old=z.copy();x=b.solve(EA@z-c+u,rho);z=ar.solve(EB@x-c+u,rho)
            primal=EB@x+EA@z-c;u+=primal;dual=rho*EB.T@EA@(z-old)
            stationarity=max(b.kkt(x,rho*u),ar.kkt(z,rho*u))
            torque=ctrl.scale*np.r_[x[:12],z[:6]];normalized=torque/ctrl.scale
            G,lo,hi=[problem[key] for key in ('G','lo','hi')]
            bound=max(0.,float(np.max(lo-G@normalized)),float(np.max(G@normalized-hi)))
            lock=max(inf(C@normalized-C@(tc/ctrl.scale)) for C,d in problem['tasks'][:2])
            record=dict(iteration=it,acc_consensus=inf(primal[:6]*blocks['ascale']),
                force_consensus=inf(primal[6:]*blocks['wscale']),primal_scaled=inf(primal),
                dual=inf(dual),stationarity=stationarity,full_bound=bound,hierarchy_lock=lock,
                torque_difference=inf(torque-tc),elapsed_ms=1000*(time.perf_counter()-stamp))
            converged=record['acc_consensus']<1e-7 and record['force_consensus']<1e-7 and stationarity<2e-7 and bound<2e-5 and lock<2e-7
            if it in (1,5,10,20,50,100,200,500,maxiter) or converged:history.append(record)
            if converged:break
    except Exception as exc:failure=str(exc)
    runtime=1000*(time.perf_counter()-stamp)
    return dict(case=case,time=t,rho=rho,converged=converged,iterations=it,failure=failure,history=history,
        total_ms=blocks['assembly_ms']+setup_ms+runtime,assembly_ms=blocks['assembly_ms'],setup_ms=setup_ms,
        iteration_and_verification_ms=runtime,central_entire_wbc_ms=central_ms,
        limitation='Final level only. Exact higher-level outputs supplied by central oracle, excluded from ADMM timing. Not a deployable hierarchical controller.')

if __name__=='__main__':
    results=[]
    for case,t in (('normal',1.),('normal',8.),('hold',2.6),('double',6.)):
        for rho in (.1,1.,10.):
            row=run_case(case,t,rho);results.append(row)
            print({k:row[k] for k in ('case','time','rho','converged','iterations','total_ms','failure')},flush=True)
            (out/'admm_probe.json').write_text(json.dumps(dict(scope='Offline last-priority convex QP only; no closed-loop ADMM claim',results=results),indent=2))
