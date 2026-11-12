"""Task ordering, local/global identity, and manipulability-gradient direction."""
from pathlib import Path
import numpy as np
import pinocchio as pin
import native
from body_arm_problem import BodyArmBuilder
from body_arm_tasks import (make_task_layout, regularized_manipulability,
                            manipulability_linearization)
from run_online_wbc_mpc import Target
from model import equilibrium
from shared_upper import SharedUpper


class Captured(Exception):
    pass


class Capture:
    def solve(self, levels, G, lo, hi, initial):
        self.levels, self.G, self.lo, self.hi = levels, G, lo, hi
        raise Captured()


def smooth_join(problem, rb, ra):
    out = np.zeros(90*problem.N)
    out[problem.body.task_ids[2]] = rb
    out[problem.arm.task_ids[2]] = ra
    return out


def main():
    root = Path(__file__).resolve().parents[1]
    plant = native.Plant(str(root)); state = plant.truth()
    target = Target(plant, 'normal', 2., root/'results/shared_geometric_wbc')
    builder = BodyArmBuilder(plant, root, target)
    anchor = (state['q'].copy(), state['v'].copy(), np.zeros(24))
    problem = builder.build(state, .62, equilibrium(state), anchor)
    original = problem.central_from_local()[0]
    upper = SharedUpper(plant, root, target, 'mpc')
    capture = Capture(); upper.hierarchy = capture
    try:
        upper.solve(state, .62, equilibrium(state), anchor[2], anchor=anchor, first_slew_time=.02)
    except Captured:
        pass
    else:
        raise AssertionError('Unmodified native assembly capture did not run')
    old_body = [(c.copy(), d.copy()) for c, d in problem.body.levels]
    old_arm = [(c.copy(), d.copy()) for c, d in problem.arm.levels]
    constraints = [x.copy() for x in (problem.body.G, problem.body.lo, problem.body.hi,
                                     problem.arm.G, problem.arm.lo, problem.arm.hi)]
    rng = np.random.default_rng(39301)
    worst = 0.
    for priority in ('strict', 'joint', 'cooperative'):
        layout = make_task_layout(problem, priority, q=state['q'], model=builder.interface.full,
                                  original_levels=original)
        independent = make_task_layout(problem, priority, q=state['q'], model=builder.interface.full,
            original_levels=capture.levels, original_G=capture.G, original_lo=capture.lo)
        for lifted, direct in zip(layout.global_levels, independent.global_levels):
            np.testing.assert_allclose(lifted[0], direct[0], atol=2e-8, rtol=1e-12)
            np.testing.assert_allclose(lifted[1], direct[1], atol=2e-8, rtol=1e-12)
        for j, (C, d) in enumerate(layout.global_levels):
            ids = layout.metadata['row_ids']
            np.testing.assert_array_equal(np.sort(np.r_[ids['body'][j], ids['arm'][j]]), np.arange(len(d)))
            assert len(ids['body'][j]) == len(layout.body_levels[j][1])
            assert len(ids['arm'][j]) == len(layout.arm_levels[j][1])
        for trial in range(3):
            u = rng.normal(size=180)*.1
            xb, xa = problem.to_local(u)
            local_r = [[C@x+d for C, d in levels] for x, levels in
                       ((xb, layout.body_levels), (xa, layout.arm_levels))]
            if priority == 'strict':
                ordered = [local_r[0][0], local_r[1][1],
                           smooth_join(problem, local_r[0][2], local_r[1][2])]
            elif priority == 'joint':
                ordered = [np.r_[local_r[0][0], local_r[1][0]],
                           smooth_join(problem, local_r[0][1], local_r[1][1])]
            else:
                ordered = [local_r[1][0], np.r_[local_r[0][1], local_r[1][1]],
                           smooth_join(problem, local_r[0][2], local_r[1][2])]
            for (C, d), rr in zip(layout.global_levels, ordered):
                diff = float(np.max(np.abs(C@u+d-rr)))
                worst = max(worst, diff)
                np.testing.assert_allclose(C@u+d, rr, atol=2e-8, rtol=1e-12)
        if priority == 'cooperative':
            assert 'independent original' in independent.metadata['singularity_global_source']
            assert independent.common_grad_ms > 0 and independent.central_rows_ms > 0
            assert independent.local_rows_ms > 0
            assert layout.metadata['singularity_rows'] == 10
            assert layout.global_levels[0][0].shape[0] == 120
            assert layout.global_levels[1][0].shape[0] == 130
            assert layout.body_levels[0][0].shape[0] == 0
            assert layout.arm_levels[1][0].shape[0] == 10
        else:
            assert not layout.metadata['singularity_enabled']
    disabled = make_task_layout(problem, 'cooperative', singularity_weight=0., original_levels=original)
    for new, old in zip(disabled.global_levels, (original[1], original[0], original[2])):
        np.testing.assert_array_equal(new[0], old[0]); np.testing.assert_array_equal(new[1], old[1])
    # The Jacobian metric is invariant to a common change of floating-base pose.
    delta = np.zeros(24); delta[:6] = np.array([.1, -.2, .05, .3, -.1, .2])
    m0, grad = manipulability_linearization(builder.interface.full, state['q'])
    q_base = pin.integrate(builder.interface.full, state['q'], delta)
    np.testing.assert_allclose(regularized_manipulability(builder.interface.full, q_base), m0, atol=2e-12)
    assert np.linalg.norm(grad) > 1e-4, grad
    direction = grad/np.linalg.norm(grad)
    delta[:] = 0.; delta[18:] = 1e-5*direction
    qp = pin.integrate(builder.interface.full, state['q'], delta)
    qm = pin.integrate(builder.interface.full, state['q'], -delta)
    mp = regularized_manipulability(builder.interface.full, qp)
    mm = regularized_manipulability(builder.interface.full, qm)
    assert mp > m0 > mm, (mm, m0, mp)
    np.testing.assert_allclose((mp-mm)/2e-5, np.linalg.norm(grad), atol=2e-5, rtol=2e-5)
    # The chosen linearized residual reduces in the ascent direction.
    assert (1.-.5*grad@delta[18:])**2 < 1.
    for old, now in zip(old_body+old_arm, problem.body.levels+problem.arm.levels):
        np.testing.assert_array_equal(old[0], now[0]); np.testing.assert_array_equal(old[1], now[1])
    for old, now in zip(constraints, (problem.body.G, problem.body.lo, problem.body.hi,
                                    problem.arm.G, problem.arm.lo, problem.arm.hi)):
        np.testing.assert_array_equal(old, now)
    print('PASS: strict/joint preserved; cooperative EE > base+SV > smooth; '
          'local/global residual identity and independent native-constraint SV assembly; '
          'all task-row mappings; zero-weight reduction; gradient ascent; '
          'base-frame invariance; original task and hard-bound arrays unchanged.',
          dict(max_pullback_error=worst, log_volume=m0, gradient=grad.tolist(),
               layout_ms=layout.construction_ms))


if __name__ == '__main__':
    main()
