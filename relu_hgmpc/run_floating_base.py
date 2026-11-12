"""Matched standing loco-manipulation runs; no controller formulation changes.

Base displacement is measured from a common initial pose, not called tracking
error: the allocator deliberately moves its internal base reference, whereas
GMPC retains the home pose as a secondary objective. True base states and
actuator/contact channels are sampled before each 1 ms MuJoCo step.
"""
import argparse
import csv
import hashlib
import json
import pathlib
import time

import numpy as np
import pinocchio as pin
import native
from model import Reference
from paper_gmpc import ReferenceCache
from hierarchical_gmpc import HierarchicalGMPC
from prioritized import PrioritizedController


class CanonicalReference(Reference):
    def pose_at(self, t):
        return super().pose_at(round(t, 9))

    def twist(self, t):
        return super().twist(round(t, 9))


def rotation(q):
    return pin.Quaternion(q[6], *q[3:6]).matrix()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--controller', choices=['wbc_threshold', 'wbc_hysteresis', 'hgmpc'], required=True)
    ap.add_argument('--duration', type=float, default=12.)
    ap.add_argument('--scale', type=float, default=1.)
    ap.add_argument('--name', required=True)
    ap.add_argument('--results-dir', default='floating_base')
    ap.add_argument('--progress-every', type=int, default=100)
    args = ap.parse_args()
    if any(pathlib.Path(x).name != x for x in [args.name, args.results_dir]):
        raise ValueError('Output must be a simple directory/run name')
    if args.duration <= 0 or not np.isclose(args.duration / .02, round(args.duration / .02)):
        raise ValueError('Duration must contain whole feedback intervals')
    root = pathlib.Path(__file__).resolve().parents[1]
    out = root / 'results' / args.results_dir
    out.mkdir(parents=True, exist_ok=True)
    plant = native.Plant(str(root))
    s = plant.truth()
    home, R0 = s['q'].copy(), rotation(s['q'])
    ncycles = round(args.duration / .02)
    init = time.perf_counter()
    rawref = CanonicalReference(plant, home, s['pose'], args.scale)
    is_mpc = args.controller == 'hgmpc'
    if is_mpc:
        ref = ReferenceCache(rawref, args.duration, .02, 10, .03)
        ctrl = HierarchicalGMPC(plant=plant, ref=ref, root=root, mode='central',
            N=10, dt=.03, backend='osqp', qp_tol=2e-7, reuse_static=True,
            reuse=True, lock_mode='nullspace', algebra='builtin')
    else:
        # Exact cache of current-reference values only, with the same canonical
        # clock and charged initialization. WBC receives no future reference.
        ref = ReferenceCache(rawref, args.duration, .02, 0, .03, geometric=False)
        ctrl = PrioritizedController(plant, ref, task_dim=6, sigma_min=.1,
            activation_mode=args.controller.removeprefix('wbc_'), task_error='cartesian')
    initialization_ms = 1000 * (time.perf_counter() - init)
    poses = np.array([ref.pose_at(k * .02) for k in range(ncycles)])
    twists = np.array([ref.twist(k * .02) for k in range(ncycles)])
    reference_hash = hashlib.sha256(poses.tobytes() + twists.tobytes()).hexdigest()
    plant.instrument = True
    plant.instrument_base = True
    rows, details = [], []
    failed, advanced = None, 0
    for k in range(ncycles):
        t = k * .02
        start = time.perf_counter()
        s = plant.truth()
        try:
            tau, acc = ctrl.control(s, t)
        except Exception as exc:
            failed = dict(time=t, reason=str(exc))
            break
        feedback_ms = 1000 * (time.perf_counter() - start)
        R = rotation(s['q'])
        dp = s['q'][:3] - home[:3]
        dr = pin.log3(R0.T @ R)
        row = dict(time=t, feedback_ms=feedback_ms,
            position_error_mm=1000 * float(np.linalg.norm(s['pose'][:3, 3] - poses[k, :3, 3])),
            rotation_error_deg=float(np.linalg.norm(pin.log3(poses[k, :3, :3].T @ s['pose'][:3, :3])) * 180 / np.pi),
            base_offset_mm=1000 * float(np.linalg.norm(dp)),
            base_rotation_offset_deg=float(np.linalg.norm(dr) * 180 / np.pi),
            base_height_delta_mm=1000 * float(dp[2]),
            base_x= float(s['q'][0]), base_y=float(s['q'][1]), base_z=float(s['q'][2]),
            base_roll_deg=float(pin.rpy.matrixToRpy(R)[0] * 180 / np.pi),
            base_pitch_deg=float(pin.rpy.matrixToRpy(R)[1] * 180 / np.pi),
            base_yaw_deg=float(pin.rpy.matrixToRpy(R)[2] * 180 / np.pi),
            contacts=s['contacts'],
            joint_position_violation=float(max(0, np.max(s['q_min'] - s['q'][7:]), np.max(s['q'][7:] - s['q_max']))),
            joint_velocity_violation=float(max(0, np.max(np.abs(s['v'][6:]) - s['v_max']))),
            **ctrl.info, clips=0)
        detail = dict(q=s['q'].copy(), v=s['v'].copy(), tau=tau.copy(), acc=acc.copy())
        if not is_mpc:
            internal_R = pin.rpy.rpyToMatrix(ctrl.desired[3:6])
            detail.update(internal_base_goal=np.r_[ctrl.desired[:3], pin.Quaternion(internal_R).coeffs()],
                allocation_command=ctrl.debug['command'].copy(),
                allocation_acceleration=ctrl.debug['desired_acc'].copy())
            # This target belongs to this controller only: never rank it against
            # the GMPC home objective as an identical base tracking task.
            row['internal_base_position_error_mm'] = 1000 * float(np.linalg.norm(s['q'][:3] - ctrl.desired[:3]))
            row['internal_base_rotation_error_deg'] = float(np.linalg.norm(pin.log3(internal_R.T @ R)) * 180 / np.pi)
        details.append(detail)
        if (not np.all(np.isfinite(tau)) or s['q'][2] < .35 or np.linalg.norm(dr) > .5
            or row['joint_position_violation'] > 1e-5 or row['joint_velocity_violation'] > 1e-4
            or (not is_mpc and row['primal'] > 1e-3)):
            failed = dict(time=t, reason='Physical/constraint check', row=row)
            rows.append(row)
            break
        row['clips'] = (plant.advance(tau, acc, 20) if is_mpc else
                        plant.advance_reference(tau, ctrl.q_reference, ctrl.v_reference, 20))
        advanced += 1
        rows.append(row)
        if k % max(1, args.progress_every) == 0:
            print(args.name, {key: round(row[key], 7) for key in
                ['time', 'base_offset_mm', 'base_rotation_offset_deg', 'position_error_mm', 'feedback_ms']}, flush=True)
    if is_mpc:
        ctrl.close()
    trace, base = plant.take_trace(), plant.take_base_trace()
    final = plant.truth()
    assert trace.shape == (advanced * 20, 129)
    assert base.shape == (advanced * 20, 20)
    if advanced:
        # Validate every recorded pre-step base state against the independent
        # 50 Hz Pinocchio/MuJoCo truth channel, and each channel's clock.
        assert np.max(np.abs(base[:, 0] - trace[:, 0])) == 0
        assert np.max(np.abs(np.diff(base[:, 0]) - .001)) < 1e-10
        assert np.max(np.abs(base[::20, 1:8] - np.array([d['q'][:7] for d in details[:advanced]]))) < 1e-12
        assert np.max(np.abs(base[::20, 8:14] - np.array([d['v'][:6] for d in details[:advanced]]))) < 1e-12
    np.savez_compressed(out / f'{args.name}_trace.npz', trace=trace, base=base, home=home,
        desired_pose=poses, desired_twist=twists, final_q=final['q'], final_v=final['v'],
        **({key: np.array([d[key] for d in details]) for key in details[0]} if details else {}))
    if rows:
        with (out / f'{args.name}.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = dict(args=vars(args), failed=failed, cycles=len(rows), simulated_cycles=advanced,
        initialization_ms=initialization_ms, reference_cache_misses=ref.misses,
        reference_sha256=reference_hash, instrumentation_verified=True,
        base_trace_columns=['time', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw',
            'vx_body', 'vy_body', 'vz_body', 'wx_body', 'wy_body', 'wz_body',
            'vx_world', 'vy_world', 'vz_world', 'wx_world', 'wy_world', 'wz_world'],
        source_sha256={name: hashlib.sha256((pathlib.Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ['run_floating_base.py', 'prioritized.py', 'hierarchical_gmpc.py',
                         'paper_gmpc.py', 'osqp_acceleration.py', 'split_gmpc.py', 'model.py', 'native.cpp']},
        timing_scope='state read + complete controller, including model/matrices/solver/checks; reference and controller initialization amortized; logging and MuJoCo excluded; no compute-delay injection',
        scope='6D engineering extension of original allocation+WBC versus existing two-level GMPC; unchanged controller objectives/gains/constraints; threshold and hysteresis reported separately; not a same-QP or geometry-only comparison')
    if rows:
        summary.update(position_rmse_mm=float(np.sqrt(np.mean([r['position_error_mm']**2 for r in rows]))),
            rotation_rmse_deg=float(np.sqrt(np.mean([r['rotation_error_deg']**2 for r in rows]))),
            torque_clips=sum(r['clips'] for r in rows),
            amortized_feedback_ms=(sum(r['feedback_ms'] for r in rows) + initialization_ms) / len(rows))
        for key in rows[0]:
            values = np.array([r[key] for r in rows], dtype=float)
            summary[key] = dict(mean=float(values.mean()), max=float(values.max()), min=float(values.min()),
                p95=float(np.quantile(values, .95)), p99=float(np.quantile(values, .99)))
    (out / f'{args.name}.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: summary[key] for key in ['args', 'failed', 'simulated_cycles',
        'position_rmse_mm', 'rotation_rmse_deg', 'amortized_feedback_ms'] if key in summary}), flush=True)
    if failed:
        raise RuntimeError(failed)


if __name__ == '__main__':
    main()
