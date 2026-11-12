"""Independent tests for exact preconditioning and certified row generation."""
import json
import numpy as np
from unittest.mock import patch

import body_arm_admm_solver as solver


OPTIONS = dict(tol=1e-9, interface_tol=1e-9, bound_tol=1e-9,
               max_iter=20000, adaptive_interval=50, parallel=True)


def scalar_problem():
    # Relaxation optimum is 1. Body upper bound makes the full optimum .5.
    b = [np.eye(1), np.array([0.]), np.array([[1.], [1.]]),
         np.array([-3., -3.]), np.array([.5, 4.]), np.eye(1), np.zeros(1)]
    a = [np.eye(1), np.array([-2.]), np.eye(1), np.array([-3.]),
         np.array([3.]), np.eye(1), np.zeros(1)]
    return b+a, np.ones(1)


def certificate(args, w, r, tol=2e-7):
    assert r['converged'], (r['status'], r['screen_history'])
    value=0.
    for off,suffix in [(0,'b'),(7,'a')]:
        H,g,G,l,u,E,e=args[off:off+7]
        x,yg,ye=r['x_'+suffix],r['y_g_'+suffix],r['y_e_'+suffix]
        assert np.max(np.abs(H@x+g+G.T@yg+E.T@ye),initial=0.)<tol
        ax=G@x
        assert np.max(np.abs(ax-np.clip(ax,l,u)),initial=0.)<tol
        assert np.max(np.abs(ax-np.clip(ax+yg,l,u)),initial=0.)<tol
        value+=.5*x@H@x+g@x
    assert np.max(np.abs(args[5]@r['x_b']+args[6]-args[12]@r['x_a']-args[13]),initial=0.)<tol
    assert np.max(np.abs(r['y_e_b']+r['y_e_a']),initial=0.)<tol
    np.testing.assert_allclose(r['objective'],value,atol=1e-12)


def test_violated_omitted_bound_is_added_and_exact_optimum_recovered():
    args,w=scalar_problem()
    r=solver.solve(args,w,OPTIONS)
    certificate(args,w,r)
    np.testing.assert_allclose(r['x_b'],[.5],atol=1e-8)
    np.testing.assert_allclose(r['x_a'],[.5],atol=1e-8)
    assert r['screen_passes']==2
    assert r['screen_history'][0]['rows_b']==0
    assert r['screen_history'][0]['original_bound_raw']>.49
    assert r['rows_used_b']==1 and r['rows_used_a']==0
    assert r['y_g_b'][1]==0. and r['y_g_a'][0]==0.
    np.testing.assert_allclose(r['y_g_b'][0],1.,atol=1e-7)


def test_full_bounds_and_screened_results_agree():
    args,w=scalar_problem()
    full=solver.solve(args,w,OPTIONS,screen_bounds=False)
    screened=solver.solve(args,w,OPTIONS)
    certificate(args,w,full);certificate(args,w,screened)
    assert full['screen_passes']==1 and full['rows_used_b']==2 and full['rows_used_a']==1
    np.testing.assert_allclose(full['x_b'],screened['x_b'],atol=2e-8)
    np.testing.assert_allclose(full['objective'],screened['objective'],atol=2e-8)


def test_input_arrays_and_original_objective_are_unchanged():
    args,w=scalar_problem();before=[a.copy() for a in args]
    r=solver.solve(args,w,dict(OPTIONS,tol=1e-12,certificate_tol=1e-9),preconditioner_shift=100.)
    certificate(args,w,r)
    np.testing.assert_allclose(r['x_b'],[.5],atol=1e-8)
    for actual,expected in zip(args,before):np.testing.assert_array_equal(actual,expected)


def test_psd_objectives_and_empty_local_task():
    args=[np.diag([1.,0.]),np.array([-1.,0.]),np.eye(2),-4*np.ones(2),4*np.ones(2),
          np.eye(2),np.zeros(2),np.diag([0.,1.]),np.array([0.,2.]),np.eye(2),
          -4*np.ones(2),4*np.ones(2),np.eye(2),np.zeros(2)]
    r=solver.solve(args,np.ones(2),OPTIONS)
    certificate(args,np.ones(2),r)
    np.testing.assert_allclose(r['x_b'],[1.,-2.],atol=2e-8)


def test_zero_dimension_block():
    args=[np.empty((0,0)),np.empty(0),np.empty((1,0)),np.array([-1.]),np.array([1.]),
          np.empty((1,0)),np.array([1.]),np.array([[2.]]),np.array([-4.]),
          np.empty((0,1)),np.empty(0),np.empty(0),np.eye(1),np.zeros(1)]
    r=solver.solve(args,np.ones(1),OPTIONS)
    certificate(args,np.ones(1),r)
    np.testing.assert_allclose(r['x_a'],[1.],atol=2e-8)
    assert r['coordinate_factorization_count']==1 and r['factorization_count_b']==0


def test_screen_pass_limit_is_not_success():
    args,w=scalar_problem();r=solver.solve(args,w,OPTIONS,max_screen_passes=1)
    assert not r['converged'] and r['native_converged']
    assert r['status']=='constraint-generation pass limit' and r['bound_raw']>.49
    assert r['screen_passes']==1 and r['rows_used_b']==0


def test_native_iteration_failure_is_preserved():
    args,w=scalar_problem()
    r=solver.solve(args,w,dict(OPTIONS,max_iter=1,tol=1e-14))
    assert not r['converged'] and not r['native_converged']
    assert r['screen_passes']==1 and r['iterations']==1
    assert r['status'].startswith('native failure:')


def test_native_failure_flag_never_discarded_even_at_feasible_optimum():
    args,w=scalar_problem();native=solver.kernel.solve
    def report_failure(*a,**kw):
        r=native(*a,**kw)
        r['converged']=False;r['status']='forced failure flag'
        return r
    with patch.object(solver.kernel,'solve',side_effect=report_failure):
        r=solver.solve(args,w,OPTIONS,screen_bounds=False)
    assert not r['converged'] and r['native_status']=='forced failure flag'
    assert r['bound_raw']<1e-8 and r['local_stationarity']<1e-8


def test_original_coordinate_certificate_does_not_trust_native_success():
    args,w=scalar_problem();native=solver.kernel.solve
    def corrupt_output(*a,**kw):
        r=native(*a,**kw)
        r['x_b']=np.zeros_like(r['x_b']);r['x_a']=np.zeros_like(r['x_a'])
        for key in ('y_g_b','y_g_a','y_e_b','y_e_a'):r[key]=np.zeros_like(r[key])
        r['converged']=True;r['status']='solved'
        return r
    with patch.object(solver.kernel,'solve',side_effect=corrupt_output):
        r=solver.solve(args,w,OPTIONS,screen_bounds=False)
    assert not r['converged'] and r['native_converged']
    assert r['status']=='original-coordinate certificate failed'
    assert r['local_stationarity']>.9


def test_infeasible_zero_row_is_added_then_failure_preserved():
    args,w=scalar_problem();args[2]=np.zeros((1,1));args[3]=np.ones(1);args[4]=2*np.ones(1)
    r=solver.solve(args,w,dict(OPTIONS,max_iter=200))
    assert not r['converged'] and r['screen_passes']==2
    assert r['rows_used_b']==1 and r['bound_raw']==1.
    assert r['status'].startswith('native failure:')


def test_empty_interface_unbounded_intervals_and_warm_coordinates():
    args=[np.array([[2.]]),np.array([-4.]),np.eye(1),np.array([-np.inf]),np.array([np.inf]),
          np.empty((0,1)),np.empty(0),np.array([[3.]]),np.array([3.]),np.empty((0,1)),
          np.empty(0),np.empty(0),np.empty((0,1)),np.empty(0)]
    r=solver.solve(args,np.empty(0),dict(OPTIONS,warm_x_b=np.array([1.5]),warm_x_a=np.array([-.8])))
    certificate(args,np.empty(0),r)
    np.testing.assert_allclose(r['x_b'],[2.],atol=1e-8)
    np.testing.assert_allclose(r['x_a'],[-1.],atol=1e-8)


def test_timing_and_factor_counts_include_all_screen_passes():
    args,w=scalar_problem();r=solver.solve(args,w,dict(OPTIONS,adaptive_interval=0))
    certificate(args,w,r)
    assert r['coordinate_factorization_count']==2
    assert r['native_factorization_count']==4
    assert r['factorization_count']==6
    terms=('setup_ms','factor_ms','iteration_ms','coord_ms','check_ms','screen_ms')
    assert all(r[k]>=0 for k in terms)
    assert sum(r[k] for k in terms)<=r['total_ms']+1e-6
    assert r['kernel_wall_ms']>=r['native_total_ms']


def test_factored_qr_retains_psd_objective_without_eigenvalue_clipping():
    rng=np.random.default_rng(19)
    Q,_=np.linalg.qr(rng.normal(size=(5,5)))
    F=np.diag([1e4,7.,.2])@Q[:3]
    target=np.array([.13,-.07,.11,.04,-.03])
    bias=-F@target
    args=[F.T@F,F.T@bias,np.empty((0,5)),np.empty(0),np.empty(0),np.empty((0,5)),np.empty(0),
          np.eye(1),np.array([-2.]),np.empty((0,1)),np.empty(0),np.empty(0),np.empty((0,1)),np.empty(0)]
    factors=[(F,bias),(np.eye(1),np.array([-2.]))]
    native=solver.kernel.solve;mins=[]
    def observe(*a,**kw):
        mins.append(float(np.linalg.eigvalsh(a[0])[0]))
        return native(*a,**kw)
    with patch.object(solver.kernel,'solve',side_effect=observe):
        r=solver.solve(args,np.empty(0),dict(OPTIONS,tol=1e-11,certificate_tol=1e-8),
                       objective_factors=factors)
    assert r['converged'],r['status']
    assert min(mins)>-1e-12
    np.testing.assert_allclose(F@r['x_b'],F@target,atol=1e-7)
    np.testing.assert_allclose(r['x_a'],[2.],atol=1e-8)
    assert r['objective_factors_used'] and r['objective_factor_H_relative_error']<1e-14


def test_incorrect_objective_factor_is_rejected():
    args,w=scalar_problem()
    try:
        solver.solve(args,w,OPTIONS,objective_factors=[(2*np.eye(1),np.zeros(1)),
                                                       (np.eye(1),np.array([-2.]))])
    except ValueError as error:
        assert 'do not represent' in str(error)
    else:
        raise AssertionError('An objective-changing factor was accepted')


def test_interface_qr_metric_preserves_original_constrained_optimum():
    args,w=scalar_problem()
    factors=[(np.eye(1),np.zeros(1)),(np.eye(1),np.array([-2.]))]
    r=solver.solve(args,w,dict(OPTIONS,tol=1e-11,certificate_tol=1e-9),
                   objective_factors=factors,preconditioner_interface_weight=10.)
    certificate(args,w,r)
    np.testing.assert_allclose(r['x_b'],[.5],atol=1e-8)
    np.testing.assert_allclose(r['x_a'],[.5],atol=1e-8)
    assert r['preconditioner_interface_weight']==10.
    assert r['screen_passes']==2 and r['objective_factors_used']


if __name__=='__main__':
    tests=[(name,fn) for name,fn in globals().copy().items() if name.startswith('test_')]
    for name,fn in tests:
        fn();print('PASS',name,flush=True)
    print(json.dumps({'passed':len(tests),'tests':[name for name,fn in tests]},indent=2))
