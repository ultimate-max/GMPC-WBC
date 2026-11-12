"""Independent moving-frame finite-difference and constrained-HQP checks."""
import json,pathlib
import numpy as np
import pinocchio as pin
import native
from geometric_wbc import task_error,GeometricWBC,rolling_contact
from common_wbc_reference import CommonReference,smooth

def error(T,Td,mode):
    E=Td.inverse()*T
    return pin.log6(E).vector if mode=='se3' else np.r_[T.translation-Td.translation,pin.log3(E.rotation)]

rng=np.random.default_rng(20261111);tests={}
for mode in ['product','se3']:
    maximum=np.zeros(4)
    for _ in range(30):
        T=pin.exp6(rng.normal(size=6)*.3);Td=pin.exp6(rng.normal(size=6)*.3)
        V=rng.normal(size=6)*.2;Vd=rng.normal(size=6)*.2
        drift=rng.normal(size=6)*.1;Ad=rng.normal(size=6)*.1
        J=rng.normal(size=(6,8));u=rng.normal(size=8)*.2
        e,ed,Je,b=task_error(T,V,drift,J,Td,Vd,Ad,mode)
        h=2e-4;actual_a=drift+J@u
        ep=error(T*pin.exp6(h*V+.5*h*h*actual_a),Td*pin.exp6(h*Vd+.5*h*h*Ad),mode)
        em=error(T*pin.exp6(-h*V+.5*h*h*actual_a),Td*pin.exp6(-h*Vd+.5*h*h*Ad),mode)
        first=np.max(np.abs((ep-em)/(2*h)-ed))
        second=np.max(np.abs((ep+em-2*e)/(h*h)-(Je@u+b)))
        delta=1e-5
        jac=np.column_stack([(error(T*pin.exp6(delta*J[:,j]),Td,mode)-error(T*pin.exp6(-delta*J[:,j]),Td,mode))/(2*delta) for j in range(8)])
        _,_,_,b2=task_error(T,V,drift,J,Td,Vd,Ad,mode,eps=1e-6)
        maximum=np.maximum(maximum,[first,second,np.max(np.abs(jac-Je)),np.max(np.abs(b-b2))])
    assert maximum[0]<1e-7 and maximum[1]<2e-6 and maximum[2]<1e-8 and maximum[3]<1e-7,(mode,maximum)
    tests[mode]=dict(velocity_derivative_error=maximum[0],acceleration_derivative_error=maximum[1],
        jacobian_error=maximum[2],directional_step_sensitivity=maximum[3])
root=pathlib.Path(__file__).resolve().parents[1];p=native.Plant(str(root));home=p.truth()
reference=CommonReference(p,'normal');r=reference.sample(2.)
for mode in ['product','se3']:
    c=GeometricWBC(p,mode)
    tau,acc=c.control(home,reference.sample(0.))
    tests[mode]['initial_hqp']=c.info
    assert c.info['contact_residual']<1e-8 and c.info['dynamics_residual']<1e-8
    assert c.info['task_lock_max']<1e-8
# Check full configuration velocity/acceleration, independently of contact solve.
t=2.;h=2e-4;q=reference.configuration(t)
qp=reference.configuration(t+h);qm=reference.configuration(t-h)
dp=p.difference(q,qp);dm=p.difference(q,qm)
velocity=float(np.max(np.abs((dp-dm)/(2*h)-r['v'])))
acceleration=float(np.max(np.abs((dp+dm)/(h*h)-r['a'])))
assert velocity<2e-6 and acceleration<3e-5,(velocity,acceleration)
tests['reference']=dict(velocity_fd_error=velocity,acceleration_fd_error=acceleration)
# Independent directional derivative of the spherical no-slip Jacobian.
s=rolling_contact(p.evaluate(r['q'],r['v']));eps=1e-5
plus=p.evaluate(p.integrate(r['q'],eps*r['v']),r['v'])
minus=p.evaluate(p.integrate(r['q'],-eps*r['v']),r['v'])
drift_fd=(plus['Jc']-minus['Jc'])@r['v']/(2*eps)
drift_error=float(np.max(np.abs(drift_fd-s['cdotv'])))
assert drift_error<1e-7,drift_error
tests['reference']['contact_drift_fd_error']=drift_error
for endpoint in [.5,1.5]:
    h=1e-4
    assert abs((smooth(endpoint+h)-smooth(endpoint-h))/(2*h))<1e-6
    assert abs((smooth(endpoint+h)+smooth(endpoint-h)-2*smooth(endpoint))/(h*h))<.002
tests['smooth_ramp_endpoint_checks']=True
# Exercise the numerical zero-row handling and rejection of a real conflict.
c=GeometricWBC(p,'product')
x,res,_=c.solve_level(0,np.eye(1),np.zeros(1),np.array([[1.],[1e-12]]),
    np.array([-1.,5e-9]),np.array([1.,1.]),np.zeros(1))
assert abs(x[0])<1e-8 and res['qp_primal_relative']<2e-7
try:
    c.solve_level(0,np.eye(1),np.zeros(1),np.array([[1e-12]]),np.array([1e-4]),np.array([1.]),np.zeros(1))
except RuntimeError as exc:assert 'Inherited zero-row' in str(exc)
else:raise AssertionError('A genuinely infeasible inherited bound was accepted')
tests['nullspace_zero_row_checks']=True
out=root/'results/geometric_wbc';out.mkdir(parents=True,exist_ok=True)
(out/'verification.json').write_text(json.dumps(tests,indent=2));print(json.dumps(tests,indent=2))
