"""Structural fairness and strict task-lock checks, independent of performance."""
import pathlib
import numpy as np
import native
from shared_hierarchy import HierarchyQP
from shared_upper import SharedUpper
from shared_wbc import CommonWBC
from online_wbc_mpc import ReferenceMux
from run_online_wbc_mpc import Target

def main():
    # Lower task asks to undo higher optimum; the lock must prevent it.
    solver=HierarchyQP();G=np.eye(3);lo=np.full(3,-1.);hi=np.ones(3)
    levels=[(np.array([[1.,1.,0.]]),np.array([-3.])),
            (np.array([[1.,0.,1.]]),np.array([2.])),(np.eye(3),np.zeros(3))]
    x,info=solver.solve(levels,G,lo,hi,np.zeros(3))
    np.testing.assert_allclose(x,[1,1,-1],atol=2e-7)
    assert info['hierarchy_kkt']<2e-7 and info['hierarchy_lock']<2e-7
    # A bound row inherited by an exact task lock has only roundoff left.
    # Its 1e-13 violation must not become a huge normalized inconsistency.
    x,info=HierarchyQP().solve([(np.array([[1.,0.,0.]]),np.array([-1.])),
        (np.array([[0.,1.,0.]]),np.array([-3.])),(np.eye(3),np.zeros(3))],
        np.eye(3),np.full(3,-1.),np.array([1.-1e-13,1.,1.]),np.zeros(3))
    np.testing.assert_allclose(x,[1,1,0],atol=2e-7)
    root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/shared_geometric_wbc'
    p=native.Plant(str(root));p.fast_reference=True;s=p.truth();mux=ReferenceMux(p,s)
    target=Target(p,'normal',2.,out);r=mux.reference(0.);a=CommonWBC(p,'se3');b=CommonWBC(p,'se3')
    captured={};original=a.solve_level
    def capture(i,C,d,G,lo,hi,warm):
        if i==0:captured.update(G=G.copy(),lo=lo.copy(),hi=hi.copy())
        return original(i,C,d,G,lo,hi,warm)
    a.solve_level=capture
    ta,aa=a.control(s,r);tb,ab=b.control(s,r)
    np.testing.assert_array_equal(ta,tb);np.testing.assert_array_equal(aa,ab)
    assert not hasattr(a,'arm_tracking_accel_bounds')
    controllers={m:SharedUpper(p,root,target,m) for m in ['allocation','mpc']}
    anchor=mux.state(.02)
    for m,c in controllers.items():
        calls=[];source=c.target.target
        class Spy:
            def at(self,t):calls.append(t);return source.at(t)
        c.target.target=Spy()
        plan=c.solve(s,.02,a.prev_tau,anchor[2],anchor=anchor,first_slew_time=.02)
        if m=='allocation':assert set(calls)=={.02},calls
        else:assert max(calls)>=.42-1e-10
        assert c.last_info['qp_residual']<2e-7
        assert c.last_info['hierarchy_lock']<2e-7
        assert mux.install(plan,.02)<1e-8
    ca,cm=controllers['allocation'].debug,controllers['mpc'].debug
    np.testing.assert_allclose(ca['G'],cm['G'][:134,:18],atol=1e-12)
    np.testing.assert_allclose(cm['G'][:134,18:],0,atol=1e-12)
    np.testing.assert_allclose(ca['lo'],cm['lo'][:134],atol=1e-12)
    np.testing.assert_allclose(ca['hi'],cm['hi'][:134],atol=1e-12)
    np.testing.assert_allclose(captured['G'][:38],ca['G'][:38],atol=1e-10)
    np.testing.assert_allclose(captured['G'][92:110],ca['G'][56:74],atol=1e-9)
    np.testing.assert_allclose(captured['lo'][92:110],ca['lo'][56:74],atol=1e-9)
    np.testing.assert_allclose(captured['hi'][92:110],ca['hi'][56:74],atol=1e-9)
    np.testing.assert_allclose((captured['hi'][38:56]-captured['lo'][38:56])/.005,
        (ca['hi'][38:56]-ca['lo'][38:56])/.02,atol=1e-9)
    print('PASS: strict priorities and degenerate rows; identical WBC outputs; shared upper/WBC physical rows and slew limits; preview isolation; C2 splice; original KKT gates')

if __name__=='__main__':main()
