"""Explicit task-order variants shared by centralized and distributed GMPC.

cooperative: end-effector pose/twist > base pose/twist plus a soft arm
singularity preference > the existing posture/acceleration/torque smoothness.
Hard physics and the original three task groups are never modified in place.

The singularity preference is the first-order residual of
  exp(-.5 * (m(q0+p_arm) - m(q0)))
at p_arm=0, where m=.5 log det(J_scaled J_scaled.T + eps**2 I).
It is a local, regularized soft preference, not a singularity-avoidance guarantee.
Its gradient can vanish at an exact stationary/singular configuration.
"""
from dataclasses import dataclass
import time
import numpy as np
import pinocchio as pin


@dataclass
class TaskLayout:
    global_levels: list
    body_levels: list
    arm_levels: list
    metadata: dict
    construction_ms: float
    common_grad_ms: float = 0.
    central_rows_ms: float = 0.
    local_rows_ms: float = 0.
    original_rows_ms: float = 0.


def regularized_manipulability(model, q, *, eps=.05, length_scale=.5,
                              frame_name='tool_tip', data=None):
    """Dimensionless regularized log-volume; LOCAL arm Jacobian only."""
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError('singularity_eps must be finite and positive')
    if not np.isfinite(length_scale) or length_scale <= 0:
        raise ValueError('length_scale must be finite and positive')
    if model.nv != 24:
        raise ValueError('The matched B2+Z1 model must have 24 velocity coordinates')
    q = np.asarray(q)
    if q.shape != (model.nq,) or not np.all(np.isfinite(q)):
        raise ValueError('q must be the finite current measured model configuration')
    if not model.existFrame(frame_name):
        raise ValueError(f'Unknown end-effector frame {frame_name}')
    data = model.createData() if data is None else data
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    J = np.array(pin.getFrameJacobian(model, data, model.getFrameId(frame_name), pin.LOCAL))[:, 18:24]
    J[:3] /= length_scale
    gram = J@J.T+eps**2*np.eye(6)
    sign, logdet = np.linalg.slogdet(gram)
    if sign != 1 or not np.isfinite(logdet):
        raise FloatingPointError('Regularized manipulability is not positive definite')
    return .5*float(logdet)


def manipulability_linearization(model, q, *, eps=.05, length_scale=.5, step=1e-5):
    """One six-joint central difference per control cycle; no shared Data state."""
    if not np.isfinite(step) or step <= 0:
        raise ValueError('finite-difference step must be finite and positive')
    data = model.createData()
    metric = lambda x: regularized_manipulability(model, x, eps=eps,
                                                 length_scale=length_scale, data=data)
    m0 = metric(q)
    gradient = np.zeros(6)
    for j in range(6):
        delta = np.zeros(model.nv); delta[18+j] = step
        gradient[j] = (metric(pin.integrate(model, q, delta))-
                       metric(pin.integrate(model, q, -delta)))/(2*step)
    return m0, gradient


def _global_original(problem):
    """Reconstruct only task rows; no unnecessary bound-matrix multiplication."""
    result = []
    for level, count in enumerate((12*problem.N, 12*problem.N, 90*problem.N)):
        C, d = np.zeros((count, 18*problem.N)), np.zeros(count)
        for name in ('body', 'arm'):
            local = getattr(problem, name)
            L, offset = getattr(problem, 'lift_'+name), getattr(problem, 'offset_'+name)
            ids = local.task_ids[level]
            lc, ld = local.levels[level]
            C[ids], d[ids] = lc@L, lc@offset+ld
        result.append((C, d))
    return result


def _stack(*levels):
    return np.vstack([C for C, _ in levels]), np.concatenate([d for _, d in levels])


def make_task_layout(problem, priority='cooperative', *, q=None, model=None,
                     original_levels=None, original_G=None, original_lo=None,
                     singularity_weight=1.,
                     singularity_eps=.05, length_scale=.5):
    """Return one common hierarchy for both optimizers, with measured setup time.

    Supply original_levels from the existing native centralized assembly to
    avoid constructing those rows again.  With original_G/original_lo, global
    singularity rows are independently constructed from the original arm trust
    bounds, without borrowing local condensed trajectories or their lift.
    q/model are mandatory only for a
    nonzero cooperative singularity preference.  This function never solves a
    QP or reads a centralized optimum.  SV output is reused by both solvers.
    """
    start = time.perf_counter()
    if priority not in ('cooperative', 'strict', 'joint'):
        raise ValueError(priority)
    if not np.isfinite(singularity_weight) or singularity_weight < 0:
        raise ValueError('singularity_weight must be finite and nonnegative')
    if (original_G is None) != (original_lo is None):
        raise ValueError('original_G and original_lo must be supplied together')
    if original_G is not None:
        if np.shape(original_G) != (134*problem.N, 18*problem.N) or np.shape(original_lo) != (134*problem.N,):
            raise ValueError('Expected the unchanged 134-rows-per-node original constraints')
    common_grad_ms = central_rows_ms = local_rows_ms = 0.
    tick = time.perf_counter()
    original = _global_original(problem) if original_levels is None else original_levels
    original_rows_ms = 1000*(time.perf_counter()-tick)
    if len(original) != 3:
        raise ValueError('Expected the three unchanged original task groups')
    body, arm = problem.body.levels, problem.arm.levels
    metadata = dict(priority=priority, singularity_enabled=False,
                    singularity_weight=float(singularity_weight),
                    singularity_eps=float(singularity_eps), length_scale_m=float(length_scale),
                    geometric_frame='tool_tip LOCAL', arm_velocity_columns=[18, 24],
                    finite_difference_step=1e-5,
                    physical_constraints_unchanged=True, original_task_groups_unchanged=True)
    if priority == 'strict':
        tick = time.perf_counter()
        global_levels, body_levels, arm_levels = list(original), list(body), list(arm)
        central_rows_ms += 1000*(time.perf_counter()-tick)
        metadata['order'] = ['base', 'end_effector', 'smoothness']
    elif priority == 'joint':
        tick = time.perf_counter()
        global_levels = [_stack(original[0], original[1]), original[2]]
        central_rows_ms += 1000*(time.perf_counter()-tick)
        tick = time.perf_counter()
        body_levels = [_stack(body[0], body[1]), body[2]]
        arm_levels = [_stack(arm[0], arm[1]), arm[2]]
        local_rows_ms += 1000*(time.perf_counter()-tick)
        metadata['order'] = ['base_and_end_effector', 'smoothness']
    else:
        sv_arm = (np.zeros((0, problem.arm.n)), np.zeros(0))
        sv_global = (np.zeros((0, 18*problem.N)), np.zeros(0))
        if singularity_weight:
            if q is None or model is None:
                raise ValueError('cooperative singularity task requires the current q and model')
            tick = time.perf_counter()
            m0, gradient = manipulability_linearization(model, q, eps=singularity_eps,
                                                        length_scale=length_scale)
            common_grad_ms = 1000*(time.perf_counter()-tick)
            tick = time.perf_counter()
            w = np.sqrt(singularity_weight)
            rows, bias = [], []
            for P, p, *_ in problem.arm.nodes:
                rows.append(-.5*w*gradient@P[6:])
                bias.append(w*(1.-.5*gradient@p[6:]))
            C, d = np.vstack(rows), np.asarray(bias)
            sv_arm = (C, d)
            local_rows_ms += 1000*(time.perf_counter()-tick)
            tick = time.perf_counter()
            if original_G is None:
                sv_global = (C@problem.lift_arm, C@problem.offset_arm+d)
                global_source = 'local affine lift (audit/prototype path)'
            else:
                grows, gbias = [], []
                for k in range(problem.N):
                    ids = slice(134*k+104, 134*k+110)
                    P_arm = original_G[ids]
                    p_arm = -.15-original_lo[ids]
                    grows.append(-.5*w*gradient@P_arm)
                    gbias.append(w*(1.-.5*gradient@p_arm))
                sv_global = (np.vstack(grows), np.asarray(gbias))
                global_source = 'independent original joint-trust rows 134*k+104:110'
            central_rows_ms += 1000*(time.perf_counter()-tick)
            metadata.update(singularity_enabled=True, regularized_log_volume=m0,
                            manipulability_gradient=gradient.tolist(),
                            singularity_rows=problem.N,
                            singularity_global_source=global_source,
                            residual='sqrt(weight) * (1 - 0.5 * grad(m).T * p_arm)',
                            approximation='first-order normalized exponential residual at measured q',
                            guarantee='secondary soft preference only; stationary gradients may vanish')
        else:
            metadata['singularity_rows'] = 0
        tick = time.perf_counter()
        global_levels = [original[1], _stack(original[0], sv_global), original[2]]
        central_rows_ms += 1000*(time.perf_counter()-tick)
        tick = time.perf_counter()
        body_levels = [body[1], body[0], body[2]]
        arm_levels = [arm[1], sv_arm, arm[2]]
        local_rows_ms += 1000*(time.perf_counter()-tick)
        metadata['order'] = ['end_effector', 'base_and_singularity_preference', 'smoothness']
    body_ids, arm_ids = problem.body.task_ids, problem.arm.task_ids
    if priority == 'strict':
        ids_b, ids_a = body_ids, arm_ids
    elif priority == 'joint':
        ids_b = [np.r_[body_ids[0], 12*problem.N+body_ids[1]], body_ids[2]]
        ids_a = [np.r_[arm_ids[0], 12*problem.N+arm_ids[1]], arm_ids[2]]
    else:
        ids_b = [body_ids[1], body_ids[0], body_ids[2]]
        ids_a = [arm_ids[1], 12*problem.N+np.arange(metadata['singularity_rows']), arm_ids[2]]
    metadata['row_ids'] = dict(body=[ids.tolist() for ids in ids_b],
                               arm=[ids.tolist() for ids in ids_a])
    total_ms = 1000*(time.perf_counter()-start)
    metadata['timing'] = dict(common_grad_ms=common_grad_ms, central_rows_ms=central_rows_ms,
                              local_rows_ms=local_rows_ms, original_rows_ms=original_rows_ms,
                              construction_ms=total_ms)
    return TaskLayout(global_levels, body_levels, arm_levels, metadata, total_ms,
                      common_grad_ms, central_rows_ms, local_rows_ms, original_rows_ms)
