"""Independent numerical checks for the fixed-QP ADMM prototype.

Run after sourcing relu_hgmpc/env.sh. No robot simulations are started.
The reference solver and every acceptance check use the ORIGINAL QP data.
"""
import json
import pathlib
import sys

import numpy as np
import scipy.sparse as sp
import osqp

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "structured_admm_native" / "build"))
import structured_admm_native as admm
from central_relu import residuals

TOL = 2e-7
METRICS = ("qp_primal_relative", "qp_dual_relative", "qp_normal_relative")


def reference(H, g, A, l, u):
    """Scale rows only; return non-scaled constraint multipliers."""
    rows = 1.0 / np.maximum(np.linalg.norm(A, axis=1), 1e-12)
    qp = osqp.OSQP(algebra="builtin")
    qp.setup(P=sp.csc_matrix(np.triu(H)), q=g, A=sp.csc_matrix(rows[:, None] * A),
             l=rows * l, u=rows * u, verbose=False, eps_abs=1e-11,
             eps_rel=1e-11, max_iter=100000, polishing=True,
             check_termination=10, adaptive_rho=True)
    sol = qp.solve(raise_error=False)
    assert sol.info.status_val in (1, 2), sol.info.status
    return sol.x, rows * sol.y


def check_case(name, H, g, A, l, u, expected_x=None, **options):
    original = [a.copy() for a in (H, g, A, l, u)]
    options = dict(tol=TOL, primal_abs_tol=TOL, max_iter=20000,
                   **options)
    result = admm.solve(H, g, A, l, u, **options)
    independent = residuals(H, g, A, l, u, result["x"], result["y"])
    assert result["converged"], (name, result["status"], independent)
    assert max(independent[k] for k in METRICS) <= TOL, (name, independent)
    assert independent["qp_primal_raw"] <= TOL, (name, independent)
    for key in (*METRICS, "qp_primal_raw", "qp_dual_raw", "qp_objective"):
        assert np.isclose(result[key], independent[key], rtol=1e-7, atol=1e-12), (
            name, key, result[key], independent[key])
    for before, after in zip(original, (H, g, A, l, u)):
        assert np.array_equal(before, after), (name, "input changed")
    ref_x, ref_y = reference(H, g, A, l, u)
    ref_metrics = residuals(H, g, A, l, u, ref_x, ref_y)
    assert max(ref_metrics[k] for k in METRICS) <= TOL, (name, "reference", ref_metrics)
    obj_error = abs(independent["qp_objective"] - ref_metrics["qp_objective"])
    assert obj_error <= 3e-6 * max(1., abs(ref_metrics["qp_objective"])), (name, obj_error)
    if expected_x is not None:
        assert np.max(np.abs(result["x"] - expected_x)) < 3e-6, (name, result["x"])
    print(name, result["iterations"], result["factorization_count"],
          max(independent[k] for k in METRICS), flush=True)
    return result, dict(name=name, iterations=result["iterations"],
                        factorizations=result["factorization_count"],
                        objective_error=obj_error, **independent)


def main():
    reports = []
    H = np.diag([2., 4., 1.]); g = np.array([-6., 8., -.5])
    A = np.vstack((np.eye(3), np.zeros((1, 3)), np.ones((1, 3))))
    l = np.array([0., -1., -2., -np.inf, -np.inf])
    u = np.array([1., 2., 2., np.inf, np.inf])
    box, report = check_case("analytic_pd_box", H, g, A, l, u,
                            expected_x=np.array([1., -1., .5]), adaptive_interval=0)
    reports.append(report)
    assert box["factorization_count"] == 1
    assert box["y"][0] > 0 and box["y"][1] < 0
    assert np.max(np.abs(box["y"][2:])) < 1e-6

    warm, report = check_case("fixed_rho_warm", H, g, A, l, u,
                             expected_x=np.array([1., -1., .5]),
                             adaptive_interval=0, warm_x=box["x"], warm_y=box["y"])
    reports.append(report)
    assert warm["factorization_count"] == 1
    assert warm["iterations"] <= box["iterations"]

    whitened, report = check_case("spd_exact_whitening", H, g, A, l, u,
                                 expected_x=np.array([1., -1., .5]),
                                 preconditioner_shift=0.)
    reports.append(report)
    assert abs(whitened["qp_objective"] + 11.125) < 3e-6

    Hpsd = np.diag([2., 0., 0.]); gpsd = np.array([-2., 0., 0.])
    Ap = np.eye(3); lp = np.array([-1., -np.inf, -1.]); up = np.array([.5, np.inf, 1.])
    # Distinct null-space coordinates are valid; compare only objective/KKT.
    psd, report = check_case("psd_nonunique", Hpsd, gpsd, Ap, lp, up,
                            warm_x=np.array([0., 3., .2]))
    reports.append(report)
    assert abs(psd["qp_objective"] + .75) < 2e-6

    # A shift inside the coordinate preconditioner must not add a penalty to
    # the original objective. A linear null-space objective makes that mistake
    # observable (an artificial quadratic would change its bound optimum).
    Hpre = np.diag([2., 0., 0.]); gpre = np.array([-2., -.01, 0.])
    lpre = np.array([-1., -2., -1.]); upre = np.array([.5, 2., 1.])
    preconditioned, report = check_case("psd_coordinate_preconditioning", Hpre,
                                      gpre, np.eye(3), lpre, upre,
                                      preconditioner_shift=1e-6)
    reports.append(report)
    assert abs(preconditioned["qp_objective"] + .77) < 3e-6
    assert abs(preconditioned["y"][1] - .01) < 2e-7

    He = np.array([[2., .2], [.2, 3.]]); ge = np.array([-1., 2.])
    Ae = np.array([[1., 1.], [2., 2.], [1., 0.], [0., 1.]])
    le = np.array([.4, .8, -1., -1.]); ue = np.array([.4, .8, 1., 1.])
    _, report = check_case("redundant_equalities", He, ge, Ae, le, ue)
    reports.append(report)

    Hs = np.diag([3e4, 2e4, 1e4]); gs = np.array([-9e4, 4e4, -4e3])
    scales = np.array([1e-5, 1e3, 1e-2]); As = np.diag(scales)
    ls = scales * np.array([-2., -1., -2.]); us = scales * np.array([1., 2., 2.])
    for scaling in (True, False):
        scaled, report = check_case("physical_duals_scale_" + str(scaling),
                                    Hs, gs, As, ls, us,
                                    expected_x=np.array([1., -1., .4]),
                                    objective_scaling=scaling)
        reports.append(report)
        # A^T y must balance the original objective, not the internally scaled one.
        assert np.max(np.abs(Hs @ scaled["x"] + gs + As.T @ scaled["y"])) < .03

    limited = admm.solve(H, g, A, l, u, max_iter=1, tol=1e-12,
                         primal_abs_tol=1e-12, adaptive_interval=0)
    assert not limited["converged"] and limited["status"] == "maximum iterations"
    assert limited["iterations"] == 1
    rr = residuals(H, g, A, l, u, limited["x"], limited["y"])
    assert max(rr[k] for k in METRICS) > 1e-12
    reports.append(dict(name="iteration_limit_not_success", iterations=1, **rr))
    print(json.dumps(dict(passed=len(reports), tests=reports), indent=2))


if __name__ == "__main__":
    main()
