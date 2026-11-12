"""Exact local coordinates and certified constraint generation for graph ADMM.

H/g enter in the caller's ONE common objective scale. Cholesky(H + shift I)
defines only an invertible coordinate transform: the transformed objective is
T.T @ H @ T, never T.T @ (H + shift I) @ T. The original arrays are not changed.

Constraint screening solves a relaxation, checks every original row, and adds
all violated omitted rows. A relaxed optimum satisfying all original rows and
the original KKT conditions is also an optimum of the full convex QP. A failed
native solve is returned as failure; neither a centralized fallback nor a
reference solution is used. Active-row knowledge is not shared across calls.

Timing: total_ms is the complete wrapper wall, including pybind conversion and
certificates. setup_ms excludes coordinate factorization; factor_ms includes
both coordinate and native factors. precondition_ms is a diagnostic subtotal
overlapping setup_ms/factor_ms and must NOT be added to those again. screen_ms
is row-selection bookkeeping only; original full-row certificates are check_ms.
"""
import pathlib
import sys
import time

import numpy as np
import scipy.linalg as la

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent /
                       'body_arm_admm_native' / 'build'))
import body_arm_admm_native as kernel


def _inf(a):
    return float(np.max(np.abs(a))) if np.size(a) else 0.


def _metrics(blocks, xs, ygs, yes, interface_scale, objective_factors=None):
    """Independent caller-coordinate KKT certificate, including omitted rows."""
    objective = bound = primal = normal = stationarity = stationarity_raw = 0.
    matrix_stationarity = matrix_stationarity_raw = matrix_objective = 0.
    pscale = dscale = 1.
    interfaces, violations = [], []
    finite = True
    for bi, (block, x, yg, ye) in enumerate(zip(blocks, xs, ygs, yes)):
        H, g, G, lower, upper, E, e = block
        ax, hx = G @ x, H @ x
        aty = G.T @ yg + E.T @ ye
        ir = E @ x + e
        finite = finite and all(np.all(np.isfinite(a)) for a in (x, yg, ye, ax, hx, aty, ir))
        interfaces.append(ir)
        projected = np.clip(ax, lower, upper)
        violation = np.abs(ax - projected)
        violations.append(violation)
        bound = max(bound, _inf(violation))
        metric = 1. / np.maximum(1., la.norm(G, axis=1))
        ma, mp = metric * ax, metric * projected
        pscale = max(pscale, _inf(ma), _inf(mp))
        primal = max(primal, _inf(metric * violation))
        nproj = np.clip(ma + yg / metric, metric * lower, metric * upper)
        normal = max(normal, _inf(ma - nproj))
        matrix_raw = _inf(hx + g + aty)
        matrix_stationarity_raw = max(matrix_stationarity_raw, matrix_raw)
        matrix_stationarity = max(matrix_stationarity,
            matrix_raw/max(1., _inf(hx), _inf(g), _inf(aty)))
        matrix_objective += float(.5*x@hx+g@x)
        if objective_factors is None:
            grad = hx+g
            objective += float(.5*x@hx+g@x)
        else:
            F, bias = objective_factors[bi]
            fx = F@x
            grad = F.T@(fx+bias)
            objective += float(.5*fx@fx+bias@fx)
        raw = _inf(grad + aty)
        stationarity_raw = max(stationarity_raw, raw)
        stationarity = max(stationarity, raw / max(1., _inf(hx), _inf(g), _inf(aty)))
        dscale = max(dscale, _inf(ye))
    interface = interfaces[0] - interfaces[1]
    dual_balance = _inf(yes[0] + yes[1])
    result = dict(objective=objective, bound_raw=bound,
                  primal_raw=max(bound, _inf(interface)), primal_relative=primal/pscale,
                  interface_raw=_inf(interface),
                  scaled_interface=_inf(interface_scale * interface),
                  local_stationarity=stationarity,
                  local_stationarity_raw=stationarity_raw,
                  original_local_stationarity=stationarity,
                  original_local_stationarity_raw=stationarity_raw,
                  matrix_local_stationarity=matrix_stationarity,
                  matrix_local_stationarity_raw=matrix_stationarity_raw,
                  matrix_objective=matrix_objective,
                  normal=max(normal/pscale, dual_balance/dscale),
                  interface_dual_balance=dual_balance, finite=finite)
    return result, violations, interfaces


def solve(args14list, interface_scale, kernel_options, *,
          preconditioner_shift=1e-6, screen_bounds=True, max_screen_passes=8,
          objective_factors=None, preconditioner_interface_weight=0.):
    """Return the native result schema in original local coordinates.

    args14list is [Hb,gb,Gb,lb,ub,Eb,eb,Ha,ga,Ga,la,ua,Ea,ea].
    Optional warm_x_b/a in kernel_options are ORIGINAL local coordinates.
    Optional certificate_tol controls the independent original-coordinate KKT
    gate (defaults to native tol); it is not forwarded to the kernel. Bounds and
    interfaces always use the original bound_tol/interface_tol, independently
    of variable or row scaling. None disables coordinate preconditioning.
    objective_factors may contain [(F_body,b_body),(F_arm,b_arm)] representing
    H=F.T@F and g=F.T@b in the SAME input objective scale. Its roundoff agreement
    with H/g is verified. QR of [F;sqrt(shift)I] then avoids a cancellation-prone
    normal matrix; H_hat=(F@T).T@(F@T), g_hat=(F@T).T@b, and the independent
    original-coordinate gradient is evaluated as F.T@(F@x+b). No eigenvalue
    clipping or objective regularization is performed. Raw H/g matrix-gradient
    diagnostics remain separately available for checking roundoff amplification.
    A positive preconditioner_interface_weight optionally appends
    sqrt(weight)*diag(interface_scale)*E to the QR stack; only the coordinate
    metric changes, while H_hat and g_hat still use F alone.
    """
    start = time.perf_counter()
    if len(args14list) != 14:
        raise ValueError('Expected exactly 14 local QP arrays')
    if max_screen_passes < 1:
        raise ValueError('max_screen_passes must be positive')
    if preconditioner_shift is not None and (
            not np.isfinite(preconditioner_shift) or preconditioner_shift < 0.):
        raise ValueError('preconditioner_shift must be finite and nonnegative, or None')
    if not np.isfinite(preconditioner_interface_weight) or preconditioner_interface_weight < 0.:
        raise ValueError('preconditioner_interface_weight must be finite and nonnegative')
    options = dict(kernel_options)
    tol = float(options.get('tol', 2e-7))
    certificate_tol = float(options.pop('certificate_tol', tol))
    bound_tol = float(options.get('bound_tol', 2e-7))
    interface_tol = float(options.get('interface_tol', 2e-7))
    if any(not np.isfinite(x) or x <= 0. for x in (certificate_tol, bound_tol, interface_tol)):
        raise ValueError('Certificate tolerances must be positive and finite')
    w = np.array(interface_scale, dtype=float, copy=True).reshape(-1)
    if not np.all(np.isfinite(w)) or np.any(w <= 0.):
        raise ValueError('Interface scales must be positive and finite')

    prep_start = time.perf_counter()
    blocks, transformed, transforms, inv_transforms, selected = [], [], [], [], []
    factors = [] if objective_factors is not None else None
    factor_errors = []
    if objective_factors is not None and len(objective_factors) != 2:
        raise ValueError('Expected one objective factor/bias pair per local block')
    xs, coordinate_factor_count = [], [0, 0]
    coordinate_factor_ms = 0.
    for bi, offset in enumerate((0, 7)):
        H, g, G, lower, upper, E, e = (
            np.array(a, dtype=float, copy=True) for a in args14list[offset:offset+7])
        if any(a.ndim != 1 for a in (g, lower, upper, e)):
            raise ValueError('g/l/u/e must be one-dimensional')
        n, m = len(g), len(lower)
        if (H.shape != (n, n) or G.shape != (m, n) or len(upper) != m
                or E.shape != (len(w), n) or len(e) != len(w)):
            raise ValueError('Local QP dimensions do not match')
        if any(not np.all(np.isfinite(a)) for a in (H, g, G, E, e)):
            raise ValueError('Local matrices and affine terms must be finite')
        if (_inf(H-H.T) > 1e-9*max(1., _inf(H)) or np.any(np.isnan(lower))
                or np.any(np.isnan(upper)) or np.any(lower > upper)
                or np.any(np.isposinf(lower)) or np.any(np.isneginf(upper))):
            raise ValueError('Invalid symmetry or local intervals')
        x = options.pop('warm_x_b' if bi == 0 else 'warm_x_a', None)
        x = np.zeros(n) if x is None else np.array(x, dtype=float, copy=True).reshape(-1)
        if x.shape != (n,) or not np.all(np.isfinite(x)):
            raise ValueError('Invalid original-coordinate warm start')
        xs.append(x)
        blocks.append((H, g, G, lower, upper, E, e))
        F = bias = None
        if objective_factors is not None:
            F, bias = (np.array(a, dtype=float, copy=True) for a in objective_factors[bi])
            if F.ndim != 2 or F.shape[1] != n or bias.shape != (F.shape[0],):
                raise ValueError('Objective factor dimensions do not match the local QP')
            if not np.all(np.isfinite(F)) or not np.all(np.isfinite(bias)):
                raise ValueError('Objective factors must be finite')
            ferr = _inf(F.T@F-H)/max(1., _inf(H))
            gerr = _inf(F.T@bias-g)/max(1., _inf(g))
            if max(ferr,gerr) > 5e-10:
                raise ValueError('Objective factors do not represent the supplied H/g')
            factors.append((F,bias));factor_errors.append((ferr,gerr))
        T, invT = np.eye(n), np.eye(n)
        if n and preconditioner_shift is not None:
            fs = time.perf_counter()
            if F is None:
                P = .5*(H+H.T) + preconditioner_shift*np.eye(n)
                if preconditioner_interface_weight:
                    WE = w[:,None]*E
                    P += preconditioner_interface_weight*(WE.T@WE)
                U = la.cholesky(P, lower=False)
            else:
                # Forming F.T@F before whitening would magnify its nullspace
                # roundoff by as much as 1/shift. QR retains the PSD structure.
                stack = [F, np.sqrt(preconditioner_shift)*np.eye(n)]
                if preconditioner_interface_weight:
                    stack.append(np.sqrt(preconditioner_interface_weight)*w[:,None]*E)
                stack = np.vstack(stack)
                _, U = la.qr(stack, mode='economic')
            coordinate_factor_ms += 1000*(time.perf_counter()-fs)
            coordinate_factor_count[bi] = 1
            T = la.solve_triangular(U, np.eye(n), lower=False)
            invT = U
        if F is None:
            Ht, gt = T.T@H@T, T.T@g
        else:
            Ft = F@T
            Ht, gt = Ft.T@Ft, Ft.T@bias
        transformed.append((.5*(Ht+Ht.T), gt, G@T, lower, upper, E@T, e))
        transforms.append(T); inv_transforms.append(invT)
        selected.append(np.zeros(m, dtype=bool) if screen_bounds else np.ones(m, dtype=bool))
    precondition_ms = 1000*(time.perf_counter()-prep_start)

    totals = dict(setup_ms=precondition_ms-coordinate_factor_ms,
                  factor_ms=coordinate_factor_ms, iteration_ms=0., coord_ms=0.,
                  check_ms=0., screen_ms=0., kernel_wall_ms=0., native_total_ms=0.)
    counts = dict(iterations=0, checks=0, rho_updates=0,
                  factorization_count=sum(coordinate_factor_count),
                  factorization_count_b=coordinate_factor_count[0],
                  factorization_count_a=coordinate_factor_count[1])
    per_factor_ms = [0., 0.]
    history, passes, status, accepted = [], 0, 'not run', False
    last_native = None
    last_metrics = None
    last_interfaces = None
    last_ygs, last_yes = [np.zeros(len(b[3])) for b in blocks], [np.zeros(len(w)), np.zeros(len(w))]

    for screen_pass in range(max_screen_passes):
        ps = time.perf_counter()
        args, warm = [], []
        for data, active, invT, x in zip(transformed, selected, inv_transforms, xs):
            Ht, gt, Gt, lower, upper, Et, e = data
            args.extend((Ht, gt, Gt[active], lower[active], upper[active], Et, e))
            warm.append(invT @ x)
        totals['setup_ms'] += 1000*(time.perf_counter()-ps)
        ks = time.perf_counter()
        raw = kernel.solve(*args, w, warm_x_b=warm[0], warm_x_a=warm[1], **options)
        totals['kernel_wall_ms'] += 1000*(time.perf_counter()-ks)
        last_native = raw;passes += 1
        for key in ('setup_ms', 'factor_ms', 'iteration_ms', 'coord_ms', 'check_ms'):
            totals[key] += float(raw[key])
        totals['native_total_ms'] += float(raw['total_ms'])
        for key in counts:
            counts[key] += int(raw[key])
        per_factor_ms[0] += float(raw['factor_ms_b'])
        per_factor_ms[1] += float(raw['factor_ms_a'])
        xs = [T @ raw['x_'+suffix] for T, suffix in zip(transforms, ('b', 'a'))]
        ygs, yes = [], []
        for block, active, suffix in zip(blocks, selected, ('b', 'a')):
            y = np.zeros(len(block[3]));y[active] = raw['y_g_'+suffix]
            ygs.append(y);yes.append(np.array(raw['y_e_'+suffix], copy=True))
        ck = time.perf_counter()
        metrics, violations, interfaces = _metrics(blocks, xs, ygs, yes, w, factors)
        totals['check_ms'] += 1000*(time.perf_counter()-ck)
        counts['checks'] += 1
        last_metrics, last_interfaces = metrics, interfaces
        last_ygs, last_yes = ygs, yes
        history.append(dict(pass_index=screen_pass, native_converged=bool(raw['converged']),
                            native_status=str(raw['status']), native_iterations=int(raw['iterations']),
                            rows_b=int(selected[0].sum()), rows_a=int(selected[1].sum()),
                            original_bound_raw=metrics['bound_raw'],
                            original_interface_raw=metrics['interface_raw'],
                            original_stationarity=metrics['local_stationarity']))
        if not bool(raw['converged']):
            status = 'native failure: '+str(raw['status'])
            break
        if not metrics['finite']:
            status = 'nonfinite original-coordinate result'
            break
        ss = time.perf_counter()
        add = [(v > bound_tol) & ~mask for v, mask in zip(violations, selected)]
        history[-1]['added_rows_b'] = int(add[0].sum())
        history[-1]['added_rows_a'] = int(add[1].sum())
        if any(np.any(a) for a in add):
            if screen_pass+1 == max_screen_passes:
                status = 'constraint-generation pass limit'
                totals['screen_ms'] += 1000*(time.perf_counter()-ss)
                break
            for mask, newly in zip(selected, add):
                mask |= newly
            totals['screen_ms'] += 1000*(time.perf_counter()-ss)
            continue
        totals['screen_ms'] += 1000*(time.perf_counter()-ss)
        accepted = (metrics['bound_raw'] <= bound_tol
                    and metrics['interface_raw'] <= interface_tol
                    and metrics['primal_relative'] <= certificate_tol
                    and metrics['local_stationarity'] <= certificate_tol
                    and metrics['normal'] <= certificate_tol)
        status = 'solved' if accepted else 'original-coordinate certificate failed'
        break

    # Native scalar fields are useful diagnostics, but all physical certificates
    # and coordinates below overwrite their transformed counterparts.
    result = dict(last_native)
    result.update(last_metrics)
    result.update(totals); result.update(counts)
    result.update(x_b=xs[0], x_a=xs[1], y_g_b=last_ygs[0], y_g_a=last_ygs[1],
                  y_e_b=last_yes[0], y_e_a=last_yes[1],
                  interface_b=last_interfaces[0], interface_a=last_interfaces[1],
                  converged=bool(accepted), status=status,
                  native_converged=bool(last_native['converged']),
                  native_status=str(last_native['status']),
                  native_local_stationarity=float(last_native['local_stationarity']),
                  screen_bounds=bool(screen_bounds), screen_passes=passes,
                  screen_history=history, rows_used_b=int(selected[0].sum()),
                  rows_used_a=int(selected[1].sum()),
                  rows_total_b=len(blocks[0][3]), rows_total_a=len(blocks[1][3]),
                  preconditioner_shift=preconditioner_shift, precondition_ms=precondition_ms,
                  preconditioner_interface_weight=preconditioner_interface_weight,
                  coordinate_factor_ms=coordinate_factor_ms,
                  coordinate_factorization_count=sum(coordinate_factor_count),
                  coordinate_transform_ms=precondition_ms-coordinate_factor_ms,
                  native_factorization_count=counts['factorization_count']-sum(coordinate_factor_count),
                  # These native per-block times exclude the separately reported
                  # coordinate factor time, whose wall is included in factor_ms.
                  factor_ms_b=per_factor_ms[0], factor_ms_a=per_factor_ms[1],
                  certificate_tol=certificate_tol)
    result['objective_factors_used'] = objective_factors is not None
    result['objective_factor_H_relative_error'] = max((v[0] for v in factor_errors),default=0.)
    result['objective_factor_g_relative_error'] = max((v[1] for v in factor_errors),default=0.)
    result['total_ms'] = 1000*(time.perf_counter()-start)
    return result
