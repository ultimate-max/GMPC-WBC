"""Offline falsification tests for geometric/incremental GMPC reuse.

Never changes the production controller. Uses existing states, plus passive
frame transformations and explicitly labelled synthetic joint perturbations.
The fixed-active-set calculation is an equality-QP diagnostic, not NMPC proof.
"""
import json, pathlib, time, hashlib, platform
import numpy as np
import scipy.linalg as la
import pinocchio as pin
import native
from model import Reference, equilibrium, se3
from split_gmpc import SplitGMPC

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/geometry_reuse_research'
OUT.mkdir(parents=True, exist_ok=True)


def assemble(parts):
    a,b=parts
    H=la.block_diag(a['H'],b['H']); g=np.r_[a['g'],b['g']]
    A=np.vstack((la.block_diag(a['A'],b['A']),np.c_[a['E'],-b['E']]))
    l=np.r_[a['l'],b['l'],b['d']-a['d']]; u=np.r_[a['u'],b['u'],b['d']-a['d']]
    eq=np.isfinite(l)&np.isfinite(u)&(l==u)
    return dict(H=H,g=g,A=A,l=l,u=u,eq=eq)


def rel(a,b):
    return float(la.norm(a-b)/max(1.,la.norm(b)))


def spectrum(a):
    s=la.svdvals(a)
    if s[0]<1e-12:
        return dict(top=float(s[0]),rank_1e_8=0,rank_1percent=0,singular_values=s.tolist())
    return dict(top=float(s[0]),rank_1e_8=int(np.sum(s>s[0]*1e-8)),
                rank_1percent=int(np.sum(s>s[0]*.01)),singular_values=s.tolist())


def transform_q(q,H):
    T=H*pin.SE3(pin.Quaternion(q[6],*q[3:6]).matrix(),q[:3])
    out=q.copy(); out[:3]=T.translation; out[3:7]=pin.Quaternion(T.rotation).coeffs()
    return out


class MovedReference:
    def __init__(self,ref,H): self.ref=ref; self.H=H
    def pose_at(self,t): return (self.H*se3(self.ref.pose_at(t))).homogeneous
    def twist(self,t): return self.ref.twist(t)


def covariance_test(plant,ctrl,trace):
    i=300; q=trace['q'][i]; v=trace['v'][i]; t=i*.02
    s=plant.evaluate(q,v); ctrl.prev_tau=trace['tau'][i-1].copy()
    parts=ctrl.build(s,t); old=assemble(parts); results=[]
    home,base_home,ref=ctrl.home.copy(),ctrl.base_home,ctrl.ref
    for angle in [0.,.37,1.2]:
        R=pin.exp3(np.array([0.,0.,angle])); W=pin.SE3(R,np.array([1.3,-.7,0.]))
        R4=la.block_diag(*([R]*4)); moved=plant.evaluate(transform_q(q,W),v)
        # Native contact anchors are stored in the initial world frame. Move
        # those anchors as part of this PASSIVE frame change, not a robot motion.
        moved['ce']=R4@s['ce']
        ctrl.home=transform_q(home,W); ctrl.base_home=W*base_home; ctrl.ref=MovedReference(ref,W)
        new=assemble(ctrl.build(moved,t))
        P=np.eye(420)
        for k in range(ctrl.N): P[30*k+12:30*k+24,30*k+12:30*k+24]=R4
        transformed=new['A']@P
        for k in range(ctrl.N): transformed[86*k:86*k+12]=R4.T@transformed[86*k:86*k+12]
        friction=np.concatenate([np.arange(86*k+12,86*k+32) for k in range(ctrl.N)])
        other=np.ones(len(old['l']),dtype=bool); other[friction]=False
        # Rotating the pyramid normals too gives A_new = A_old P.T on these rows.
        corrected=transformed.copy();corrected[friction]=old['A'][friction]
        lnew=new['l'].copy();unew=new['u'].copy()
        for k in range(ctrl.N):
            rows=slice(86*k,86*k+12)
            lnew[rows]=R4.T@lnew[rows];unew[rows]=R4.T@unew[rows]
        finite=np.isfinite(old['l']);finiteu=np.isfinite(old['u'])
        row=dict(yaw_rad=angle,M_relative=rel(moved['M'],s['M']),h_relative=rel(moved['h'],s['h']),
            local_tool_J_relative=rel(moved['J'],s['J']),world_contact_J_covariance=rel(moved['Jc'],R4@s['Jc']),
            contact_drift_covariance=rel(moved['cdotv'],R4@s['cdotv']),
            H_covariance_relative=rel(P.T@new['H']@P,old['H']),g_covariance_relative=rel(P.T@new['g'],old['g']),
            A_nonfriction_covariance_relative=rel(transformed[other],old['A'][other]),
            A_fixed_pyramid_relative=rel(transformed[friction],old['A'][friction]),
            A_rotated_pyramid_covariance_relative=rel(corrected,old['A']),
            lower_covariance_relative=rel(lnew[finite],old['l'][finite]),
            upper_covariance_relative=rel(unew[finiteu],old['u'][finiteu]))
        results.append(row)
    ctrl.home=home;ctrl.base_home=base_home;ctrl.ref=ref
    assert max(r['H_covariance_relative'] for r in results)<1e-10
    assert max(r['A_nonfriction_covariance_relative'] for r in results)<1e-10
    return results


def joint_change_test(plant,ctrl,trace):
    i=300;q=trace['q'][i];v=trace['v'][i];ctrl.prev_tau=trace['tau'][i-1].copy()
    s=plant.evaluate(q,v);base=assemble(ctrl.build(s,i*.02));rows=[]
    direction=np.array([.5,-.4,.7,-.3,.4,-.2]);direction/=np.linalg.norm(direction)
    for magnitude in [.02,.1,.3]:
        q2=q.copy();q2[19:]+=magnitude*direction;newstate=plant.evaluate(q2,v)
        new=assemble(ctrl.build(newstate,i*.02))
        delta=new['H']-base['H'];sc=spectrum(delta)
        rows.append(dict(joint_perturbation_norm_rad=magnitude,
            M_relative=rel(newstate['M'],s['M']),H_relative=rel(new['H'],base['H']),
            A_relative=rel(new['A'],base['A']),H_delta_spectrum=sc,
            base_H_relative=rel(new['H'][:300,:300],base['H'][:300,:300]),
            base_A_relative=rel(new['A'][:860,:300],base['A'][:860,:300]),
            arm_H_relative=rel(new['H'][300:,300:],base['H'][300:,300:])))
    return rows


def physical_reduction(plant,ctrl,state):
    """Exact affine parameterization of all 240 current equality constraints.

    Free variables are 18 physical torque coordinates per node, scaled 100/10.
    No approximate null-space reuse: recompute contact dynamics at this state.
    """
    N=ctrl.N;dt=ctrl.dt;J=state['Jc'];nv=24;n=18*N
    K=np.block([[state['M'],-J.T],[J,np.zeros((12,12))]])
    factor=la.lu_factor(K)
    scale=np.r_[np.full(12,100.),np.full(6,10.)]
    S=np.vstack((np.zeros((6,18)),np.eye(18)))
    models=ctrl.parts.evaluate(state['q'],state['v']);Ma,ha=models[1];ia=ctrl.parts.indices[1]
    qmap=np.zeros((24,n));qoff=np.zeros(24);vmap=np.zeros((24,n));voff=state['v'].copy()
    F=np.zeros((42*N,n));c=np.zeros(42*N);wscale=np.r_[np.full(3,200.),np.full(3,50.)]
    for k in range(N):
        U=np.zeros((18,n));U[:,18*k:18*(k+1)]=np.diag(scale)
        rhs=np.vstack((S@U,-100*J@qmap-20*J@vmap))
        off=np.r_[-state['h'],-state['cdotv']-100*(state['ce']+J@qoff)-20*J@voff]
        sol=la.lu_solve(factor,np.c_[rhs,off]);a=sol[:24,:-1];a0=sol[:24,-1]
        f=sol[24:,:-1];f0=sol[24:,-1]
        w=Ma[:6]@a[ia];w0=Ma[:6]@a0[ia]+ha[:6]
        rb=slice(30*k,30*(k+1));ra=slice(30*N+12*k,30*N+12*(k+1))
        F[rb]=np.vstack((U[:12]/100,f/100,-w/wscale[:,None]));c[rb]=np.r_[np.zeros(12),f0/100,-w0/wscale]
        F[ra]=np.vstack((U[12:]/10,w/wscale[:,None]));c[ra]=np.r_[np.zeros(6),w0/wscale]
        vmap+=dt*a;voff+=dt*a0;qmap+=dt*vmap;qoff+=dt*voff
    return F,c


def reduced_reuse_test(plant,ctrl,trace):
    rows=[];cache={};validation=[]
    for current in [100,200,300,400,500]:
        for age in [1,5,25,50]:
            for index in [current-age,current]:
                if index in cache:continue
                state=plant.evaluate(trace['q'][index],trace['v'][index]);ctrl.prev_tau=trace['tau'][index-1].copy()
                start=time.perf_counter();qp=assemble(ctrl.build(state,index*.02));build_ms=1000*(time.perf_counter()-start)
                start=time.perf_counter();F,c=physical_reduction(plant,ctrl,state)
                H=F.T@qp['H']@F;H=(H+H.T)/2;g=F.T@(qp['H']@c+qp['g'])
                reduction_ms=1000*(time.perf_counter()-start)
                E=qp['A'][qp['eq']];be=qp['l'][qp['eq']]
                eqmap=float(np.max(np.abs(E@F)));eqoff=float(np.max(np.abs(E@c-be)))
                assert eqmap<1e-7 and eqoff<1e-7
                cache[index]=(qp,F,c,H,g,build_ms,reduction_ms)
            old=cache[current-age];new=cache[current];qp,F,c,H,g,bms,rms=new
            start=time.perf_counter();fresh_factor=la.cho_factor(H);fresh_solution=-la.cho_solve(fresh_factor,g)
            fresh_factor_solve_ms=1000*(time.perf_counter()-start)
            H0=old[3];start=time.perf_counter();factor=la.cho_factor(H0);factor_ms=1000*(time.perf_counter()-start)
            start=time.perf_counter();yp=-la.cho_solve(factor,g);r=H@yp+g
            d=-la.cho_solve(factor,r);yc=yp+d
            solve_ms=1000*(time.perf_counter()-start)
            ye=-la.solve(H,g,assume_a='pos');xe=F@ye+c;xc=F@yc+c;xp=F@yp+c
            scale=np.r_[np.full(12,100.),np.full(6,10.)];S=np.zeros((18,len(g)));S[:,:18]=np.diag(scale)
            start=time.perf_counter();B=la.cho_solve(factor,H-H0);beta=float(la.norm(B,np.inf))
            bound=float(la.norm(S@B,np.inf)*la.norm(d,np.inf)/(1-beta)) if beta<1 else None
            # Bounds on every coordinate of ye-yc, using componentwise absolute
            # values and the Neumann-series tail. Applies if beta < 1.
            error_bound=None;all_margin_cert=False
            if beta<1:
                e=-B@d
                error_bound=np.abs(e)+np.sum(np.abs(B),axis=1)*(la.norm(e,np.inf)/(1-beta))
                mask=~qp['eq'];ax=qp['A'][mask]@xc
                allowance=np.abs(qp['A'][mask]@F)@error_bound
                all_margin_cert=bool(np.all(ax-allowance>=qp['l'][mask]) and np.all(ax+allowance<=qp['u'][mask]))
            cert_ms=1000*(time.perf_counter()-start)
            err0=float(la.norm(S@(yp-ye),np.inf));err1=float(la.norm(S@(yc-ye),np.inf))
            # Control experiment: recenter on the shifted old optimum and
            # refresh the gradient (standard adjoint/inexact-Newton idea).
            yold=-la.cho_solve(factor,old[4]);oldseq=yold.reshape(ctrl.N,18)
            grid=np.arange(ctrl.N)+age*.02/ctrl.dt
            yw=np.stack([np.interp(grid,np.arange(ctrl.N),oldseq[:,j]) for j in range(18)],axis=1).ravel()
            gradient=H@yw+g;yw1=yw-la.cho_solve(factor,gradient)
            # Energy-scaled sensitivity, not the coordinate-dependent norm of
            # H0^-1 Delta H. Full eigenanalysis is an OFFLINE oracle here.
            L=la.cholesky(H0,lower=True)
            HC=la.solve_triangular(L,H,lower=True)
            HC=la.solve_triangular(L,HC.T,lower=True).T;HC=(HC+HC.T)/2
            ev=la.eigvalsh(HC);beta_energy=float(np.max(np.abs(ev-1)))
            Sw=la.solve_triangular(L,S.T,lower=True).T
            rw1=la.solve_triangular(L,H@yw1+g,lower=True)
            min_eigen=float(ev[0]);max_eigen=float(ev[-1])
            output_bound=float(np.max(la.norm(Sw,axis=1))*la.norm(rw1)/min_eigen)
            warmerr=float(la.norm(S@(yw1-ye),np.inf))
            assert warmerr<=output_bound+1e-5
            # Matrix-free CG under the old Cholesky factor. The minimum
            # eigenvalue used to set its threshold is an expensive ORACLE,
            # so these are feasibility measurements, not an online algorithm.
            output_norm=float(np.max(la.norm(Sw,axis=1)))
            tolerance=.001*min_eigen/output_norm
            def action(w):
                y=la.solve_triangular(L.T,w,lower=False)
                return la.solve_triangular(L,H@y,lower=True)
            start=time.perf_counter()
            rhs=-la.solve_triangular(L,g,lower=True);w=L.T@yw
            residual=rhs-action(w);direction=residual.copy();rr=float(residual@residual);count=0
            while np.sqrt(rr)>tolerance and count<300:
                image=action(direction);alpha=rr/float(direction@image)
                w+=alpha*direction;residual-=alpha*image
                rr_new=float(residual@residual);direction=residual+(rr_new/rr)*direction
                rr=rr_new;count+=1
            ycg=la.solve_triangular(L.T,w,lower=False)
            cg_ms=1000*(time.perf_counter()-start)
            true_residual=la.solve_triangular(L,H@ycg+g,lower=True)
            cg_error=float(la.norm(S@(ycg-ye),np.inf))
            cg_bound=output_norm*la.norm(true_residual)/min_eigen
            if bound is not None:assert err1<=bound+1e-6
            if error_bound is not None:assert np.max(np.abs(ye-yc)-error_bound)<1e-7
            if all_margin_cert:assert inequality_violation(qp,xe)<1e-7
            rows.append(dict(time_s=.02*current,age_cycles=age,beta_inf=beta,
                before_torque_error_Nm=err0,after_torque_error_Nm=err1,after_torque_bound_Nm=bound,
                equalities_map_max=float(np.max(np.abs(qp['A'][qp['eq']]@F))),
                equalities_offset_max=float(np.max(np.abs(qp['A'][qp['eq']]@c-qp['l'][qp['eq']]))),
                exact_inequality_violation=inequality_violation(qp,xe),
                corrected_inequality_violation=inequality_violation(qp,xc),
                predictor_inequality_violation=inequality_violation(qp,xp),
                inequality_margin_certificate=all_margin_cert,
                build_ms=bms,reduction_ms=rms,old_factorization_ms=factor_ms,
                predictor_corrector_ms=solve_ms,full_certificate_ms=cert_ms,
                H_relative=rel(H,H0)))
            rows[-1].update(shifted_warm_one_correction_torque_error_Nm=warmerr,
                shifted_warm_one_correction_bound_Nm=output_bound,
                shifted_warm_one_correction_inequality_violation=inequality_violation(qp,F@yw1+c),
                beta_energy=beta_energy,preconditioned_min_eigen=min_eigen,
                preconditioned_max_eigen=max_eigen,preconditioned_condition=max_eigen/min_eigen)
            rows[-1].update(oracle_cg_iterations=count,oracle_cg_solve_ms=cg_ms,
                oracle_cg_torque_error_Nm=cg_error,oracle_cg_bound_Nm=float(cg_bound),
                oracle_cg_inequality_violation=inequality_violation(qp,F@ycg+c),
                oracle_cg_exact_residual_pass=bool(la.norm(true_residual)<=tolerance),
                cg_target_torque_error_Nm=.001,fresh_factor_solve_ms=fresh_factor_solve_ms,
                fresh_vs_reference_max=float(la.norm(fresh_solution-ye,np.inf)))
            if age==1:
                # Independent QR null-space solution validates the analytic
                # parameterization, including its signs and input scaling.
                E=qp['A'][qp['eq']];be=qp['l'][qp['eq']]
                erow=1/np.maximum(1.,la.norm(E,axis=1));E=erow[:,None]*E;be=erow*be
                Q,R,piv=la.qr(E.T,mode='full',pivoting=True);rank=len(be)
                particular=Q[:,:rank]@la.solve_triangular(R[:rank,:rank].T,be[piv],lower=True)
                Z=Q[:,rank:];Hr=Z.T@qp['H']@Z
                xqr=particular-Z@la.solve((Hr+Hr.T)/2,Z.T@(qp['H']@particular+qp['g']),assume_a='pos')
                err=float(la.norm(xqr-xe,np.inf));assert err<1e-5
                validation.append(dict(time_s=.02*current,x_vs_independent_qr_max=err,
                    reduced_H_condition=float(np.linalg.cond(H)),
                    reduced_equality_solution_stationarity=float(la.norm(H@ye+g,np.inf))))
    return dict(rows=rows,independent_validation=validation)


def qp_at(plant,ctrl,trace,index):
    state=plant.evaluate(trace['q'][index],trace['v'][index])
    ctrl.prev_tau=trace['tau'][index-1].copy() if index else equilibrium(state)
    return assemble(ctrl.build(state,index*.02))


def scaled_kkt(qp,D,row):
    E=qp['A'][qp['eq']]; F=row[:,None]*(E*D[None,:]);n=len(D)
    return np.block([[D[:,None]*qp['H']*D[None,:],F.T],[F,np.zeros((len(F),len(F)))]]),np.r_[-D*qp['g'],row*qp['l'][qp['eq']]]


def inequality_violation(qp,x):
    mask=~qp['eq'];ax=qp['A'][mask]@x
    return float(max(0,np.max(qp['l'][mask]-ax),np.max(ax-qp['u'][mask])))


def reuse_test(plant,ctrl,trace):
    rows=[];cache={}
    for current in [100,200,300,400,500]:
        for age in [1,5,25,50]:
            for index in [current-age,current]:
                if index not in cache:cache[index]=qp_at(plant,ctrl,trace,index)
            old=cache[current-age];new=cache[current];n=len(old['g'])
            # Use the SAME scaling for old and new KKT matrices to make the
            # matrix perturbation and induced norm mathematically consistent.
            D=1/np.sqrt(np.maximum(np.diag(old['H']),1e-4))
            erow=1/np.maximum(1.,la.norm(old['A'][old['eq']]*D[None,:],axis=1))
            K0,b0=scaled_kkt(old,D,erow);K,b=scaled_kkt(new,D,erow)
            start=time.perf_counter();factor=la.lu_factor(K0);factor_ms=1000*(time.perf_counter()-start)
            start=time.perf_counter();pred=la.lu_solve(factor,b);r=K@pred-b
            correction=-la.lu_solve(factor,r);one=pred+correction
            predictor_corrector_ms=1000*(time.perf_counter()-start)
            exact=la.solve(K,b,assume_a='sym');xx=D*exact[:n];xp=D*pred[:n];xc=D*one[:n]
            S=np.zeros((18,len(K)))
            S[:12,:12]=np.diag(100*D[:12]);S[12:,300:306]=np.diag(10*D[300:306])
            # Expensive reference certificate: computes FULL K0^-1 Delta K.
            # This cost is intentionally measured, never hidden in a speedup.
            start=time.perf_counter();B=la.lu_solve(factor,K-K0)
            beta=float(la.norm(B,np.inf));SB=la.norm(S@B,np.inf)
            bound=float(SB*la.norm(correction,np.inf)/(1-beta)) if beta<1 else None
            cert_ms=1000*(time.perf_counter()-start)
            torque_error_pred=float(la.norm(S@(pred-exact),np.inf))
            torque_error_one=float(la.norm(S@(one-exact),np.inf))
            if bound is not None:assert torque_error_one<=bound+1e-6
            Hchange=new['H']-old['H'];sv=spectrum(Hchange)
            rows.append(dict(time_s=current*.02,cache_age_cycles=age,cache_age_s=age*.02,
                H_relative=rel(new['H'],old['H']),A_relative=rel(new['A'],old['A']),
                H_delta_rank_1e_8=sv['rank_1e_8'],H_delta_rank_1percent=sv['rank_1percent'],
                beta_inf=beta,one_correction_torque_bound_Nm=bound,
                first_torque_error_before_Nm=torque_error_pred,first_torque_error_after_Nm=torque_error_one,
                exact_equality_solution_inequality_violation=inequality_violation(new,xx),
                predictor_inequality_violation=inequality_violation(new,xp),
                corrected_inequality_violation=inequality_violation(new,xc),
                predictor_scaled_equality_residual=float(la.norm((K@pred-b)[n:],np.inf)),
                corrected_scaled_equality_residual=float(la.norm((K@one-b)[n:],np.inf)),
                exact_scaled_kkt_residual=float(la.norm(K@exact-b,np.inf)),
                cached_factorization_ms=factor_ms,predictor_corrector_ms=predictor_corrector_ms,
                full_certificate_ms=cert_ms))
    return rows


def main():
    plant=native.Plant(str(ROOT));s0=plant.truth();ref=Reference(plant,s0['q'],s0['pose'])
    ctrl=SplitGMPC(plant,ref,ROOT,mode='central')
    path=ROOT/'results/split_gmpc/central_osqp_12s_r1_trace.npz';trace=np.load(path)
    result=dict(scope='Offline diagnostic only; fixed four-foot contact; no new closed loop; no demonstrated novel algorithm.',
        trace=str(path),trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_sha256={n:hashlib.sha256((ROOT/'relu_hgmpc'/n).read_bytes()).hexdigest() for n in ['split_gmpc.py','model.py','research_geometry_reuse.py']},
        python=platform.python_version(),pinocchio=pin.__version__,N=10,dt=.03,control_dt=.02)
    result['passive_frame_change']=covariance_test(plant,ctrl,trace)
    print('passive_frame_change',json.dumps(result['passive_frame_change']),flush=True)
    result['synthetic_arm_change']=joint_change_test(plant,ctrl,trace)
    print('arm_change',json.dumps([{k:v for k,v in r.items() if k!='H_delta_spectrum'}|{'delta_rank':r['H_delta_spectrum']['rank_1e_8']} for r in result['synthetic_arm_change']]),flush=True)
    result['temporal_reuse']=reuse_test(plant,ctrl,trace)
    result['physical_reduced_reuse']=reduced_reuse_test(plant,ctrl,trace)
    summary=[]
    for age in [1,5,25,50]:
        rows=[r for r in result['temporal_reuse'] if r['cache_age_cycles']==age]
        summary.append(dict(age_cycles=age,samples=len(rows),certified_count=sum(r['beta_inf']<1 for r in rows),
            max_before_Nm=max(r['first_torque_error_before_Nm'] for r in rows),
            max_after_Nm=max(r['first_torque_error_after_Nm'] for r in rows),
            max_beta=max(r['beta_inf'] for r in rows),
            exact_feasible_count=sum(r['exact_equality_solution_inequality_violation']<1e-7 for r in rows)))
    result['summary']=summary
    (OUT/'diagnostics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    print('reduced',json.dumps(result['physical_reduced_reuse'],indent=2),flush=True)


if __name__=='__main__':main()
