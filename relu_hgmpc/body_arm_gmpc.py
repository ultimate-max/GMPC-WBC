"""Offline physical body/arm consensus, preserving SharedUpper task hierarchy.

The physical split and local condensation live in body_arm_problem.  Each
hierarchy level is one two-block graph-consensus ADMM: the two local primal
updates are separable, while the second block projects local bounds and the
shared installation acceleration/wrench.  No centralized solver is used as
a fallback, and no reference optimum or reference task lock is an input.

Constraints may be violated at intermediate ADMM iterates.  A failed original
QP certificate stops the hierarchy, and never produces an executable Plan.
"""
import time
import pathlib
import sys
import numpy as np
import scipy.linalg as la
from central_relu import residuals

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent /
                       'body_arm_admm_native' / 'build'))
import body_arm_admm_native as kernel


def inf(a):
    return float(np.max(np.abs(a))) if np.size(a) else 0.


def null_update(Z, task, bounds, active):
    """Same normalized pivoted-QR rank convention as Shared HierarchyQP."""
    face = np.vstack((task, bounds[active])) if len(active) else task
    if not len(face):
        return Z.copy()
    norm = la.norm(face, axis=1)
    face = face[norm > 1e-10] / norm[norm > 1e-10, None]
    if not len(face):
        return Z.copy()
    Q, R, _ = la.qr(face.T, mode='full', pivoting=True)
    diag = np.abs(np.diag(R))
    rank = int(np.sum(diag > 1e-10 * diag.max())) if len(diag) else 0
    return Z @ Q[:, rank:]


def original_problem(problem):
    """Pull the decomposed rows back to ORIGINAL 180 joint-torque variables.

    These lifts eliminate the physical interface analytically, independently
    of ADMM.  They are used for certification, never to compute the answer.
    Every row is assigned once; missing/duplicated rows are fatal.
    """
    n = 18 * problem.N
    G = np.empty((134 * problem.N, n))
    lo, hi = np.empty(len(G)), np.empty(len(G))
    seen = np.zeros(len(G), int)
    sizes = [12 * problem.N, 12 * problem.N, 90 * problem.N]
    levels = [(np.empty((m, n)), np.empty(m)) for m in sizes]
    seen_tasks = [np.zeros(m, int) for m in sizes]
    for name in ('body', 'arm'):
        local = getattr(problem, name)
        lift = getattr(problem, 'lift_' + name)
        offset = getattr(problem, 'offset_' + name)
        ids = local.row_ids
        G[ids] = local.G @ lift
        lo[ids] = local.lo - local.G @ offset
        hi[ids] = local.hi - local.G @ offset
        seen[ids] += 1
        for j, (C, d) in enumerate(local.levels):
            ids = local.task_ids[j]
            levels[j][0][ids] = C @ lift
            levels[j][1][ids] = C @ offset + d
            seen_tasks[j][ids] += 1
    if not np.all(seen == 1) or any(not np.all(a == 1) for a in seen_tasks):
        raise ValueError('Local rows do not partition the original full QP')
    return levels, G, lo, hi


class BodyArmHierarchy:
    DEFAULTS = dict(max_iter=20000, tol=2e-7, interface_tol=2e-7,
                    bound_tol=2e-7, rho=.1, sigma=1e-6, alpha=1.6,
                    check_interval=10, adaptive_interval=50,
                    adaptive_threshold=5., parallel=True)

    def __init__(self, options=None, priority='cooperative'):
        if priority not in ('cooperative', 'joint', 'strict'):
            raise ValueError(priority)
        self.priority = priority
        self.options = dict(self.DEFAULTS)
        self.options.update(options or {})
        self.kernel_tol = float(self.options.pop('kernel_tol', self.options['tol']))
        self.kernel_interface_tol = float(self.options.pop('kernel_interface_tol', self.options['interface_tol']))
        self.objective_multiplier = float(self.options.pop('objective_multiplier', 1.))
        self.preconditioner_shift = self.options.pop('preconditioner_shift', None)
        self.preconditioner_interface_weight = float(self.options.pop('preconditioner_interface_weight', 0.))
        self.screen_bounds = bool(self.options.pop('screen_bounds', False))
        self.max_screen_passes = int(self.options.pop('max_screen_passes', 8))
        if self.kernel_tol <= 0 or self.objective_multiplier <= 0:
            raise ValueError('Solver-only scales must be positive')
        self.records = []
        self.last_local = None

    def solve(self, problem, initial_global):
        start = time.perf_counter()
        cs = time.perf_counter()
        levels, G, lo, hi = original_problem(problem)
        self.original = (levels, G, lo, hi)
        local_levels = [problem.body.levels, problem.arm.levels]
        if hasattr(problem, 'task_layout'):
            layout = problem.task_layout
            if layout.metadata['priority'] != self.priority:
                raise ValueError('Task layout and solver priority disagree')
            levels = layout.global_levels
            local_levels = [layout.body_levels, layout.arm_levels]
        elif self.priority == 'cooperative':
            raise ValueError('Cooperative mode requires the explicitly constructed common TaskLayout')
        elif self.priority == 'joint':
            def merge(ls):
                return [(np.vstack((ls[0][0], ls[1][0])), np.r_[ls[0][1], ls[1][1]]), ls[2]]
            levels = merge(levels)
            local_levels = [merge(ls) for ls in local_levels]
        self.optimization_levels = levels
        original_build_ms = 1000 * (time.perf_counter() - cs)
        xs = list(problem.to_local(initial_global))
        Zs = [np.eye(len(x)) for x in xs]
        Zg = np.eye(len(initial_global))
        xg = np.array(initial_global, copy=True)
        local_locks = [[], []]
        global_locks = []
        self.records = []
        complete = True
        status = 'solved'
        total_iterations = 0
        task_lock_ms = 0.
        certificate_ms = 0.
        setup_ms = 0.
        factor_ms = 0.
        iteration_ms = 0.
        native_ms = 0.
        factor_count = 0
        # Acceleration m/s^2 and rad/s^2, then force N and moment Nm.
        interface_scale = np.tile(np.r_[np.full(6, .1),
                                        np.full(3, .01), np.full(3, .1)], problem.N)
        for level in range(len(levels)):
            assembly_start = time.perf_counter()
            Cg, dg = levels[level]
            Cgr = Cg @ Zg
            objective_scale = max(1., 2 * np.max(np.sum(Cgr * Cgr, axis=0))) if Zg.shape[1] else 1.
            args = []
            local_reduced = []
            objective_factors = []
            for li, (local, x, Z) in enumerate(zip((problem.body, problem.arm), xs, Zs)):
                C, d = local_levels[li][level]
                Cr, dr = C @ Z, C @ x + d
                H = self.objective_multiplier * 2 * Cr.T @ Cr / objective_scale
                g = self.objective_multiplier * 2 * Cr.T @ dr / objective_scale
                factor_scale = np.sqrt(2 * self.objective_multiplier / objective_scale)
                objective_factors.append((factor_scale * Cr, factor_scale * dr))
                Gr, lr, ur = local.G @ Z, local.lo - local.G @ x, local.hi - local.G @ x
                Er, er = local.E @ Z, local.E @ x + local.e
                # Exact task locks can leave numerical zero physical rows.
                # Match existing handling, but always certify ORIGINAL rows.
                Gs, ls, us = Gr.copy(), lr.copy(), ur.copy()
                zero = la.norm(Gr, axis=1) < 1e-10
                inherited_error = max(0., np.max(lr[zero], initial=-np.inf),
                                      np.max(-ur[zero], initial=-np.inf))
                if inherited_error > self.options['bound_tol']:
                    complete = False
                    status = 'inherited_zero_row_violation'
                    break
                Gs[zero] = 0.
                ls[zero], us[zero] = -np.inf, np.inf
                if level:
                    ls[(ls > 0) & (ls < self.options['bound_tol'])] = 0.
                    us[(us < 0) & (us > -self.options['bound_tol'])] = 0.
                args.extend([H, g, Gs, ls, us, Er, er])
                local_reduced.append((Cr, Gr, lr, ur))
            if not complete:
                break
            reduction_ms = 1000 * (time.perf_counter() - assembly_start)
            ks = time.perf_counter()
            kernel_options = dict(self.options, tol=self.kernel_tol,
                                  interface_tol=self.kernel_interface_tol)
            if self.preconditioner_shift is not None or self.screen_bounds:
                from body_arm_admm_solver import solve as graph_solve
                kernel_options['certificate_tol'] = self.options['tol']
                result = graph_solve(args, interface_scale, kernel_options,
                                    preconditioner_shift=self.preconditioner_shift,
                                    screen_bounds=self.screen_bounds,
                                    max_screen_passes=self.max_screen_passes,
                                    objective_factors=objective_factors,
                                    preconditioner_interface_weight=self.preconditioner_interface_weight)
            else:
                result = kernel.solve(*args, interface_scale, **kernel_options)
            kernel_wall_ms = 1000 * (time.perf_counter() - ks)
            zs = [np.asarray(result['x_b']), np.asarray(result['x_a'])]
            previous_xs = [a.copy() for a in xs]
            xs = [x + Z @ z for x, Z, z in zip(xs, Zs, zs)]
            # One common positive objective factor is numerical scaling, not a
            # weight change between tasks or subsystems. Restore ALL duals for
            # certificates of the unchanged objective.
            ygs = [np.asarray(result['y_g_b']) / self.objective_multiplier,
                   np.asarray(result['y_g_a']) / self.objective_multiplier]
            yg = np.zeros(len(lo))
            for local, y in zip((problem.body, problem.arm), ygs):
                yg[local.row_ids] = y
            cgstart = time.perf_counter()
            candidate = problem.to_global(*xs)
            zglobal = Zg.T @ (candidate - xg)
            projection_error = inf(candidate - xg - Zg @ zglobal)
            Gr = G @ Zg
            lr, ur = lo - G @ xg, hi - G @ xg
            dr = Cg @ xg + dg
            H, g = 2 * Cgr.T @ Cgr / objective_scale, 2 * Cgr.T @ dr / objective_scale
            if len(zglobal):
                cert = residuals(H, g, Gr, lr, ur, zglobal, yg)
            else:
                violation = max(0., np.max(lr, initial=-np.inf), np.max(-ur, initial=-np.inf))
                cert = dict(qp_primal_raw=violation, qp_primal_relative=violation,
                            qp_dual_raw=0., qp_dual_relative=0., qp_normal_relative=violation)
            # The QP residual above belongs to the orthogonal projection of the
            # candidate into the inherited task-lock coordinates.  Also certify
            # the ACTUAL candidate's projected gradient and actual bounds; a
            # projected-point certificate alone cannot excuse a lock violation.
            grad = 2 * Cg.T @ (Cg @ candidate + dg) / objective_scale
            pg = Zg.T @ grad
            pa = Gr.T @ yg
            actual_stationarity = inf(pg + pa) / max(1., inf(pg), inf(pa))
            bound = max(0., np.max(lo - G @ candidate), np.max(G @ candidate - hi))
            lock_error = max([inf(C @ candidate - value) for C, value in global_locks] + [0.])
            interface = problem.body.E @ xs[0] + problem.body.e - problem.arm.E @ xs[1] - problem.arm.e
            local_prediction_error = 0.
            for name, x in zip(('body', 'arm'), xs):
                local_prediction_error = max(local_prediction_error,
                    inf(x - getattr(problem, 'lift_' + name) @ candidate - getattr(problem, 'offset_' + name)))
            cert_time = 1000 * (time.perf_counter() - cgstart)
            certificate_ms += cert_time
            native_converged = bool(result['converged'])
            kkt = max(actual_stationarity, *(cert[k] for k in ('qp_primal_relative', 'qp_dual_relative', 'qp_normal_relative')))
            accepted = (native_converged and kkt <= self.options['tol']
                        and bound <= self.options['bound_tol']
                        and inf(interface) <= self.options['interface_tol']
                        and lock_error <= self.options['tol']
                        and projection_error <= self.options['tol'])
            # Save each *own* stage result before making the next task lock.
            scalar_native = {k: v for k, v in result.items()
                             if np.isscalar(v) and not isinstance(v, bytes)}
            self.records.append(dict(level=level, x=candidate.copy(), local_x=[a.copy() for a in xs],
                task_output=Cg @ candidate, objective=float(np.linalg.norm(Cg @ candidate + dg)**2),
                projected_coordinate_residuals=cert, original_residuals=cert,
                actual_projected_stationarity=float(actual_stationarity),
                original_kkt=float(kkt), physical_bound=float(bound),
                lock_error=float(lock_error), inherited_projection_error=float(projection_error),
                interface_acceleration_raw=inf(interface.reshape(-1, 12)[:, :6]),
                interface_wrench_raw=inf(interface.reshape(-1, 12)[:, 6:]),
                local_prediction_error=float(local_prediction_error), accepted=bool(accepted),
                objective_scale=float(objective_scale), reduction_ms=reduction_ms,
                solver_objective_multiplier=self.objective_multiplier,
                kernel_wall_ms=kernel_wall_ms, certificate_ms=cert_time, native=scalar_native,
                active_rows=np.flatnonzero(np.abs(yg) > 1e-8)))
            total_iterations += int(result['iterations'])
            setup_ms += float(result.get('setup_ms', 0.))
            factor_ms += float(result.get('factor_ms', 0.))
            iteration_ms += float(result.get('iteration_ms', 0.))
            native_ms += kernel_wall_ms
            factor_count += int(result.get('factorization_count', 0))
            xg = candidate
            if not accepted:
                complete = False
                status = 'native_or_original_certificate_failed_at_level_' + str(level)
                break
            global_locks.append((Cg.copy(), Cg @ candidate))
            if level < len(levels)-1:
                ls = time.perf_counter()
                for i, (local, Z, yr, reduced) in enumerate(zip(
                        (problem.body, problem.arm), Zs, ygs, local_reduced)):
                    C, _ = local_levels[i][level]
                    local_locks[i].append((C.copy(), C @ xs[i]))
                    active = np.flatnonzero(np.abs(yr) > 1e-8)
                    Zs[i] = null_update(Z, reduced[0], reduced[1], active)
                Zg = null_update(Zg, Cgr, Gr, np.flatnonzero(np.abs(yg) > 1e-8))
                task_lock_ms += 1000 * (time.perf_counter() - ls)
        self.last_local = tuple(xs)
        wall_ms = 1000 * (time.perf_counter() - start)
        info = dict(converged=bool(complete and len(self.records) == len(levels)), status=status,
                    priority=self.priority, expected_levels=len(levels),
                    wall_ms=wall_ms, iterations=total_iterations, completed_levels=len(self.records),
                    original_build_ms=original_build_ms, task_lock_ms=task_lock_ms,
                    certificate_ms=certificate_ms, setup_ms=setup_ms, factor_ms=factor_ms,
                    iteration_ms=iteration_ms, kernel_wall_ms=native_ms,
                    factorization_count=factor_count, options=self.options.copy(),
                    kernel_tol=self.kernel_tol, solver_objective_multiplier=self.objective_multiplier,
                    kernel_interface_tol=self.kernel_interface_tol,
                    preconditioner_shift=self.preconditioner_shift, screen_bounds=self.screen_bounds,
                    preconditioner_interface_weight=self.preconditioner_interface_weight,
                    levels=[{k: v for k, v in r.items()
                             if k not in ('x', 'local_x', 'task_output', 'active_rows')}
                            for r in self.records])
        return xg, info
