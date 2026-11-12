"""Independent synthetic certificates for the body/arm graph consensus kernel.

These tests neither load a robot controller nor use a reference optimizer in the
native solve. Analytic KKT solutions verify both coordinate recovery and the
physical constraint gates. A passing kernel is not a robot-level equivalence
or closed-loop stability result.
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent /
                       'body_arm_admm_native' / 'build'))
import body_arm_admm_native as native


def run(q, **kw):
    return native.solve(*q, **kw)


def certify(q, result, tol=5e-7):
    Hb, gb, Gb, lb, ub, Eb, eb, Ha, ga, Ga, la, ua, Ea, ea, w = q
    xb, xa = result['x_b'], result['x_a']
    violations = []
    for H, g, G, l, u, E, x, yg, ye in [
            (Hb, gb, Gb, lb, ub, Eb, xb, result['y_g_b'], result['y_e_b']),
            (Ha, ga, Ga, la, ua, Ea, xa, result['y_g_a'], result['y_e_a'])]:
        ax = G @ x
        violations.extend(np.maximum(0., l-ax))
        violations.extend(np.maximum(0., ax-u))
        assert np.max(np.abs(H@x+g+G.T@yg+E.T@ye), initial=0.) < tol
        # Normal-cone certificate in the exact caller coordinates.
        assert np.max(np.abs(ax-np.clip(ax+yg, l, u)), initial=0.) < tol
    assert max(violations, default=0.) < tol
    assert np.max(np.abs(Eb@xb+eb-Ea@xa-ea), initial=0.) < tol
    assert np.max(np.abs(result['y_e_b']+result['y_e_a']), initial=0.) < tol
    expected = .5*xb@Hb@xb+gb@xb+.5*xa@Ha@xa+ga@xa
    np.testing.assert_allclose(result['objective'], expected, rtol=2e-13, atol=2e-13)
    assert result['converged'], result


def scalar_problem():
    # Unconstrained targets are 0 and 2; consensus optimum is x_b=x_a=1.
    return (np.eye(1), np.array([0.]), np.eye(1), np.array([-3.]), np.array([3.]),
            np.eye(1), np.zeros(1), np.eye(1), np.array([-2.]), np.eye(1),
            np.array([-3.]), np.array([3.]), np.eye(1), np.zeros(1), np.ones(1))


def analytic_problem():
    rng = np.random.default_rng(437)
    n_b, n_a, p = 5, 3, 3
    xb, xa = rng.normal(size=n_b)*.2, rng.normal(size=n_a)*.2
    Eb, Ea = rng.normal(size=(p, n_b)), rng.normal(size=(p, n_a))
    eb = rng.normal(size=p)*.4
    ea = Eb@xb+eb-Ea@xa
    lam = np.array([.13, -.2, .07])
    vals = []
    for n, x, E, sign in [(n_b, xb, Eb, 1.), (n_a, xa, Ea, -1.)]:
        A = rng.normal(size=(n,n));H=A.T@A+.8*np.eye(n)
        G = rng.normal(size=(n+3, n))
        G *= np.geomspace(.01, 100., n+3)[:, None]
        center=G@x;l=center-1.;u=center+1.
        y=np.zeros(n+3);u[0]=center[0];y[0]=.2
        l[1]=center[1];y[1]=-.15
        # Equality row and a structural zero row test raw interval handling.
        l[2]=u[2]=center[2];y[2]=.17
        G[-1]=0.;l[-1]=-1.;u[-1]=1.
        g=-H@x-G.T@y-sign*E.T@lam
        vals.append((H,g,G,l,u,E))
    b,a=vals
    q=(*b,eb,*a,ea,np.array([.2,2.,.7]))
    return q,xb,xa


def test_scalar_consensus():
    q=scalar_problem();r=run(q, adaptive_interval=0, tol=1e-9,
                               interface_tol=1e-9,bound_tol=1e-9)
    certify(q,r);np.testing.assert_allclose(r['x_b'],[1.],atol=1e-8)
    np.testing.assert_allclose(r['x_a'],[1.],atol=1e-8)
    assert r['factorization_count_b']==r['factorization_count_a']==1
    assert r['threads_used']==2
    assert r['blas_threads'] in (-1,1)


def test_scaled_original_duals_and_active_bounds():
    q,xb,xa=analytic_problem();original=[v.copy() for v in q]
    r=run(q,tol=1e-9,interface_tol=1e-9,bound_tol=1e-9,max_iter=30000)
    certify(q,r,2e-7)
    np.testing.assert_allclose(r['x_b'],xb,atol=2e-7)
    np.testing.assert_allclose(r['x_a'],xa,atol=2e-7)
    for actual,expected in zip(q,original):np.testing.assert_array_equal(actual,expected)


def test_psd_local_objectives():
    q=(np.diag([1.,0.]),np.array([-1.,0.]),np.eye(2),-4*np.ones(2),4*np.ones(2),
       np.eye(2),np.zeros(2),np.diag([0.,1.]),np.array([0.,2.]),np.eye(2),
       -4*np.ones(2),4*np.ones(2),np.eye(2),np.zeros(2),np.ones(2))
    r=run(q,tol=1e-9,interface_tol=1e-9,bound_tol=1e-9)
    certify(q,r);np.testing.assert_allclose(r['x_b'],[1.,-2.],atol=2e-8)


def test_parallel_serial_match():
    q,_,_=analytic_problem()
    rp=run(q,parallel=True,tol=1e-9,interface_tol=1e-9,bound_tol=1e-9)
    rs=run(q,parallel=False,tol=1e-9,interface_tol=1e-9,bound_tol=1e-9)
    certify(q,rp);certify(q,rs)
    np.testing.assert_allclose(rp['x_b'],rs['x_b'],atol=1e-12)
    np.testing.assert_allclose(rp['x_a'],rs['x_a'],atol=1e-12)
    assert rs['threads_used']==1 and rp['threads_used']==2


def test_proximal_term_does_not_change_objective():
    q=scalar_problem();r=run(q,sigma=2.,rho=.7,adaptive_interval=0,
                             tol=1e-9,interface_tol=1e-9,bound_tol=1e-9)
    certify(q,r);np.testing.assert_allclose(r['x_b'],[1.],atol=1e-8)


def test_zero_dimension_block():
    # Fixed body interface 1 forces the arm to 1 despite its free target 2.
    q=(np.empty((0,0)),np.empty(0),np.empty((1,0)),np.array([-1.]),np.array([1.]),
       np.empty((1,0)),np.array([1.]),np.array([[2.]]),np.array([-4.]),
       np.empty((0,1)),np.empty(0),np.empty(0),np.eye(1),np.zeros(1),np.ones(1))
    r=run(q,tol=1e-9,interface_tol=1e-9,bound_tol=1e-9)
    certify(q,r);np.testing.assert_allclose(r['x_a'],[1.],atol=1e-8)
    assert r['factorization_count_b']==0


def test_empty_interface_and_constraints():
    q=(np.array([[2.]]),np.array([-4.]),np.empty((0,1)),np.empty(0),np.empty(0),
       np.empty((0,1)),np.empty(0),np.array([[3.]]),np.array([3.]),
       np.empty((0,1)),np.empty(0),np.empty(0),np.empty((0,1)),np.empty(0),np.empty(0))
    r=run(q,tol=1e-9,interface_tol=1e-9,bound_tol=1e-9)
    certify(q,r);np.testing.assert_allclose(r['x_b'],[2.],atol=1e-9)
    np.testing.assert_allclose(r['x_a'],[-1.],atol=1e-9)


def test_iteration_limit_and_infeasible_zero_row():
    q=list(scalar_problem());q[2]=np.zeros((1,1));q[3]=np.ones(1);q[4]=2*np.ones(1)
    r=run(q,max_iter=25,check_interval=5,adaptive_interval=0)
    assert not r['converged'] and r['bound_raw']==1.
    assert r['iterations']==25 and r['status']=='maximum iterations'


def test_interface_scaling_cannot_hide_physical_infeasibility():
    # Both local variables are eliminated, but the physical interfaces disagree.
    # The weighted residual is below tol; only the raw physical gate catches it.
    b=(np.empty((0,0)),np.empty(0),np.empty((0,0)),np.empty(0),np.empty(0),
       np.empty((1,0)))
    q=(*b,np.array([1.]),*b,np.array([0.]),np.array([1e-10]))
    r=run(q,max_iter=20,check_interval=5,adaptive_interval=0)
    assert not r['converged']
    assert r['interface_raw']==1. and r['scaled_interface']==1e-10


def test_warm_primal_and_timing():
    q=scalar_problem()
    r=run(q,warm_x_b=np.array([.9]),warm_x_a=np.array([1.1]),
          tol=1e-9,interface_tol=1e-9,bound_tol=1e-9)
    certify(q,r)
    timing=sum(r[k] for k in ['setup_ms','factor_ms','iteration_ms','coord_ms','check_ms'])
    assert all(r[k]>=0 for k in ['setup_ms','factor_ms','iteration_ms','coord_ms','check_ms'])
    assert timing<=r['total_ms']+1e-6


if __name__=='__main__':
    tests=[(name,fn) for name,fn in globals().copy().items() if name.startswith('test_')]
    passed=[]
    for name,fn in tests:
        fn();passed.append(name);print('PASS',name,flush=True)
    print(json.dumps({'passed':len(passed),'tests':passed,'native':native.__file__},indent=2))
