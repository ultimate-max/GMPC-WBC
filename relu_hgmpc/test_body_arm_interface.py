"""Physical partition and frozen-prediction equivalence (not speed assertions)."""
from pathlib import Path
import numpy as np
import pinocchio as pin
import native
from body_arm_interface import BodyArmInterface, full_frozen_maps


def main():
    root = Path(__file__).resolve().parents[1]
    plant = native.Plant(str(root))
    model = BodyArmInterface(root)
    s0 = plant.truth()
    rng = np.random.default_rng(537192)
    worst = dict(partition=0., maps=0., wrench=0., dynamics=0., motion=0.)
    for sample in range(5):
        q = plant.integrate(s0['q'], rng.normal(size=24)*(.012 if sample else 0.))
        v = rng.normal(size=24)*(.12 if sample else 0.)
        state = plant.evaluate(q, v)
        split = model.linearize(state)
        C = np.array(pin.computeCoriolisMatrix(model.full, model.data['full'], q, v))
        full = full_frozen_maps(state, C)
        merged = split.recombine()
        worst['partition'] = max(worst['partition'], max(split.exactness.values()))
        for name in full:
            err = float(np.max(np.abs(merged[name]-full[name])))
            worst['maps'] = max(worst['maps'], err)
            np.testing.assert_allclose(merged[name], full[name], atol=3e-9, rtol=2e-10,
                                       err_msg=f'{sample}: {name}')
        recovered = split.wrench_map(full)
        for name in recovered:
            err = float(np.max(np.abs(recovered[name]-merged[name])))
            worst['wrench'] = max(worst['wrench'], err)
            np.testing.assert_allclose(recovered[name], merged[name], atol=3e-9, rtol=2e-10)
        # Ten-node rollout exercises velocity and contact-restoration terms.
        p = rng.normal(size=24)*.01
        vel = v+rng.normal(size=24)*.02
        pb, pa = p[split.body.idx].copy(), p[split.arm.idx].copy()
        vb, va = vel[split.body.idx].copy(), vel[split.arm.idx].copy()
        for k in range(10):
            tau = rng.normal(size=18)*np.r_[np.full(12, 8.), np.ones(6)]
            acc = full['T']@tau+full['a0']+full['Aq']@p+full['Av']@(vel-v)
            force = full['F']@tau+full['f0']+full['Fq']@p+full['Fv']@(vel-v)
            wrench = recovered['Wtau']@tau+recovered['w0']+recovered['Wq']@p+recovered['Wv']@(vel-v)
            b, a = split.body, split.arm
            ab = b.T@tau[:12]+b.W@wrench+b.a0+b.Aq@pb+b.Av@(vb-v[b.idx])
            aa = a.T@tau[12:]+a.W@wrench+a.a0+a.Aq@pa+a.Av@(va-v[a.idx])
            np.testing.assert_allclose(ab, acc[b.idx], atol=2e-8, rtol=2e-9)
            np.testing.assert_allclose(aa, acc[a.idx], atol=2e-8, rtol=2e-9)
            residual = split.residuals(p, vel, tau, acc, force, wrench)
            worst['dynamics'] = max(worst['dynamics'], max(residual.values()))
            motion = np.max(np.abs(b.Jinterface@ab-a.Jinterface@aa))
            worst['motion'] = max(worst['motion'], float(motion))
            assert max(residual.values()) < 2e-8, residual
            assert motion < 2e-8, motion
            # Negative control at the SAME state: wrong sign must be detected.
            wrong = split.residuals(p, vel, tau, acc, force, -wrench)
            assert max(wrong.values()) > 1., wrong
            dt = .04
            p += dt*vel+.5*dt*dt*acc
            vel += dt*acc
            pb += dt*vb+.5*dt*dt*ab
            vb += dt*ab
            pa += dt*va+.5*dt*dt*aa
            va += dt*aa
            np.testing.assert_allclose(pb[:6], pa[:6], atol=1e-9)
            np.testing.assert_allclose(vb[:6], va[:6], atol=1e-8)
        tau_seq = rng.normal(size=(10, 18))
        recovered_rollout = split.rollout_global(tau_seq, np.zeros(24), v, maps=full)
        assert recovered_rollout['p'].shape == (11, 24)
        for k in range(10):
            err = split.residuals(recovered_rollout['p'][k], recovered_rollout['v'][k],
                tau_seq[k], recovered_rollout['a'][k], recovered_rollout['ground_force'][k],
                recovered_rollout['wrench'][k])
            assert max(err.values()) < 2e-8, err
    print('PASS: disjoint inertia partition, M/h/C identities, frozen maps, '
          '10-node trajectories, physical interface wrench signs', worst)


if __name__ == '__main__':
    main()
