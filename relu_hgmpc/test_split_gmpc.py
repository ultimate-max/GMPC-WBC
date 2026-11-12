import pathlib,json,time
import numpy as np
import pinocchio as pin
import scipy.linalg as la
import native
from model import Reference,se3
from split_gmpc import SplitGMPC
root=pathlib.Path(__file__).resolve().parents[1];p=native.Plant(str(root));s=p.truth()
ref=Reference(p,s['q'],s['pose']);ctrl=SplitGMPC(p,ref,root,mode='central',N=10)
rng=np.random.default_rng(7102026);errors=[];acceleration_errors=[]
for i in range(6):
    q=p.integrate(s['q'],rng.normal(size=24)*.02);v=rng.normal(size=24)*.05
    st=p.evaluate(q,v);M=np.zeros((24,24));h=np.zeros(24)
    local_models=ctrl.parts.evaluate(q,v)
    for idx,(Mi,hi) in zip(ctrl.parts.indices,local_models):
        assert np.linalg.eigvalsh(Mi)[0]>1e-8
        M[np.ix_(idx,idx)]+=Mi;h[idx]+=hi
    errors.append(max(np.max(np.abs(M-st['M'])),np.max(np.abs(h-st['h']))))
    tau=rng.normal(size=18)*3;force=rng.normal(size=12)*10
    g=np.r_[np.zeros(6),tau]+st['Jc'].T@force-st['h']
    Mb,hb=local_models[0];Ma,ha=local_models[1]
    gb=np.r_[np.zeros(6),tau[:12]]+st['Jc'][:,:18].T@force-hb
    ga=np.r_[np.zeros(6),tau[12:]]-ha
    C=np.zeros((6,30));C[:,:6]=np.eye(6);C[:,18:24]=-np.eye(6)
    K=np.block([[la.block_diag(Mb,Ma),-C.T],[C,np.zeros((6,6))]])
    sol=la.solve(K,np.r_[gb,ga,np.zeros(6)])
    assembled=np.r_[sol[:18],sol[24:30]]
    acceleration_errors.append(float(np.max(np.abs(assembled-la.solve(st['M'],g)))))
assert max(errors)<1e-8,errors
assert max(acceleration_errors)<1e-8,acceleration_errors
# Pose error derivative against independent Pinocchio integration/FK.
desired=se3(ref.pose_at(1.5));actual=se3(s['pose']);E=desired.inverse()*actual
d=rng.normal(size=24);eps=1e-7
sn=p.evaluate(p.integrate(s['q'],eps*d),s['v'])
fd=(pin.log6(desired.inverse()*se3(sn['pose'])).vector-pin.log6(E).vector)/eps
jerr=float(np.max(np.abs(fd-pin.Jlog6(E)@s['J']@d)));assert jerr<1e-5,jerr
parts=ctrl.build(s,1.0);central,ic=ctrl.solve(parts)
dist=SplitGMPC(p,ref,root,mode='distributed',N=10,iterations=1500,rho=30.)
distributed,idist=dist.solve(parts)
obj=lambda zs:sum(.5*z@a['H']@z+a['g']@z for a,z in zip(parts,zs))
gap=float(abs(obj(central)-obj(distributed)))
controls=float(max(np.max(np.abs(a-b)) for a,b in zip(central,distributed)))
serial=SplitGMPC(p,ref,root,mode='distributed',N=10,iterations=30,rho=30.)
parallel=SplitGMPC(p,ref,root,mode='distributed',N=10,iterations=30,rho=30.,parallel=True)
zs,_=serial.solve(parts);zp,_=parallel.solve(parts);parallel.close()
parallel_error=float(max(np.max(np.abs(a-b)) for a,b in zip(zs,zp)))
assert parallel_error<1e-10,parallel_error
result=dict(split_model_max_error=max(errors),geometric_jacobian_error=jerr,
    interface_kkt_acceleration_error=max(acceleration_errors),
    serial_parallel_difference=parallel_error,
    central=ic,distributed=idist,objective_gap=gap,max_normalized_control_difference=controls)
out=root/'results/split_gmpc';out.mkdir(exist_ok=True,parents=True)
(out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
assert ic['interface_scaled']<1e-4 and idist['interface_scaled']<1e-4,result
assert gap<1e-3 and controls<.02,result
