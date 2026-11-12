"""Compare independently assembled body/arm QPs to unmodified native GMPC."""
from pathlib import Path
import numpy as np
import native
from body_arm_problem import BodyArmBuilder
from shared_upper import SharedUpper
from run_online_wbc_mpc import Target
from model import equilibrium


class Captured(Exception):
    pass


class Capture:
    def solve(self, levels, G, lo, hi, initial):
        self.levels = [(c.copy(), d.copy()) for c, d in levels]
        self.G, self.lo, self.hi = G.copy(), lo.copy(), hi.copy()
        raise Captured()


def main():
    root = Path(__file__).resolve().parents[1]
    plant = native.Plant(str(root)); s0 = plant.truth()
    target = Target(plant, 'normal', 2., root/'results/shared_geometric_wbc')
    upper = SharedUpper(plant, root, target, 'mpc')
    builder = BodyArmBuilder(plant, root, target)
    capture = Capture(); upper.hierarchy = capture
    rng = np.random.default_rng(82647)
    worst, elapsed = 0., []
    for i, t in enumerate((.02, .62, 1.02)):
        q = plant.integrate(s0['q'], rng.normal(size=24)*(.008 if i else 0.))
        v = rng.normal(size=24)*(.08 if i else 0.)
        s = plant.evaluate(q, v)
        anchor = (plant.integrate(q, .002*rng.normal(size=24)),
                  v+.01*rng.normal(size=24), .1*rng.normal(size=24))
        previous = equilibrium(s)
        try:
            upper.solve(s, t, previous, anchor[2], anchor=anchor, first_slew_time=.02)
        except Captured:
            pass
        else:
            raise AssertionError('Native controller capture was not reached')
        problem = builder.build(s, t, previous, anchor)
        assert problem.body.n == 180 and problem.arm.n == 120
        assert problem.body.G.shape == (980, 180)
        assert problem.arm.G.shape == (360, 120)
        np.testing.assert_array_equal(np.sort(np.r_[problem.body.row_ids, problem.arm.row_ids]),
                                      np.arange(1340))
        for level, size in enumerate((120, 120, 900)):
            np.testing.assert_array_equal(np.sort(np.r_[problem.body.task_ids[level],
                                                       problem.arm.task_ids[level]]), np.arange(size))
        for trial in range(4):
            u = rng.normal(size=180)*(.2 if trial else 0.)
            audit = problem.audit_equivalence(capture.levels, capture.G, capture.lo, capture.hi, u)
            errs = np.concatenate([np.atleast_1d(value) for value in audit.values()])
            worst = max(worst, float(np.max(errs)))
            assert np.max(errs) < 2e-7, audit
            xb, xa = problem.to_local(u)
            recovered = problem.recover(xb, xa)
            assert max(recovered['residual_max'].values()) < 2e-7, recovered['residual_max']
            # A one-sided interface perturbation must be observable even if
            # the two torque sequences have not changed.
            changed = xa.copy(); changed[6] += .01
            broken = problem.recover(xb, changed)
            assert broken['residual_max']['interface_wrench'] >= .999
        elapsed.append(problem.timing)
    print('PASS: native centralized vs independent local condensation; all 1340 '
          'constraints, 3 priorities/1140 objective rows, affine lifts, interface '
          'and torque roundtrip; max error', worst)
    print('Construction times (ms, including explicit audit lifts):', elapsed)


if __name__ == '__main__':
    main()
