"""Replay saved states and capture immutable, identical GMPC QPs.

This does not run MuJoCo forward or replace a controller.  Hierarchy capture
observes the unmodified reference implementation with a temporary Python line
trace; numerical reference decisions are therefore not copied/reimplemented.
Reference solves, serialization and observation overhead are never counted as
matrix construction.  All timings here are corpus construction measurements,
not timings of the candidate ADMM implementation.
"""
import argparse
import hashlib
import inspect
import json
import pathlib
import sys
import time

import numpy as np
import scipy.linalg as la

import native
from central_relu import residuals
from online_wbc_mpc import OnlineMPC
from run_online_wbc_mpc import Target
from shared_hierarchy import HierarchyQP
from shared_upper import SharedUpper


RESIDUAL_KEYS = ('qp_primal_relative', 'qp_dual_relative', 'qp_normal_relative')
TIMES = [0., .5, 1., 2., 4., 5.8, 6., 6.1, 6.2, 8., 10., 11.]
CALIBRATION_TIMES = {0., .5}


def digest_arrays(arrays):
    digest = hashlib.sha256()
    for key in sorted(arrays):
        a = np.ascontiguousarray(arrays[key])
        digest.update(key.encode())
        digest.update(str(a.shape).encode())
        digest.update(a.dtype.str.encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


class RecordingHierarchy(HierarchyQP):
    """Observe each original level immediately before its iterate is lifted."""
    def __init__(self):
        super().__init__()
        self.records = []
        self.level_inputs = None
        lines, begin = inspect.getsourcelines(HierarchyQP.solve)
        matching = [begin + i for i, line in enumerate(lines)
                    if 'x+=Z@z;iterations+=count' in line]
        if len(matching) != 1:
            raise RuntimeError('Hierarchy capture hook no longer matches source')
        self.capture_line = matching[0]

    def solve(self, levels, G, lo, hi, initial):
        self.records = []
        self.level_inputs = [(C.copy(), d.copy()) for C, d in levels]
        reference_code = HierarchyQP.solve.__code__
        old_trace = sys.gettrace()
        if old_trace is not None:
            raise RuntimeError('Refusing to replace an existing Python trace')

        def observe(frame, event, arg):
            if frame.f_code is not reference_code:
                return None
            if event == 'line' and frame.f_lineno == self.capture_line:
                loc = frame.f_locals
                unconstrained = loc['err'] < 1e-10 and loc['bound'] < 1e-9
                self.records.append(dict(
                    level=int(loc['i']), C=loc['C'].copy(), d=loc['d'].copy(),
                    lift_x=loc['x'].copy(), lift_Z=loc['Z'].copy(),
                    reference_x=loc['z'].copy(),
                    reference_dual=np.zeros(len(lo)) if unconstrained else loc['y'].copy(),
                    reference_fastpath=bool(unconstrained),
                    active_rows=loc['active'].copy(),
                    physical_G=G.copy(), physical_lo=lo.copy(), physical_hi=hi.copy()))
            return observe

        sys.settrace(observe)
        try:
            return super().solve(levels, G, lo, hi, initial)
        finally:
            sys.settrace(old_trace)


def build_reduced(record):
    """Exact original inherited rows, with no zero-row repair or regularizer."""
    C, d = record['C'], record['d']
    x, Z = record['lift_x'], record['lift_Z']
    G, lo, hi = record['physical_G'], record['physical_lo'], record['physical_hi']
    Cr = C if record['level'] == 0 else C @ Z
    dr = C @ x + d
    Gr = G if record['level'] == 0 else G @ Z
    lr, ur = lo - G @ x, hi - G @ x
    scale = max(1., 2 * np.max(np.sum(Cr * Cr, axis=0)))
    H, g = 2 * Cr.T @ Cr / scale, 2 * Cr.T @ dr / scale
    return dict(H=H, g=g, G=Gr, lo=lr, hi=ur, task_C=Cr,
                task_d=dr, objective_scale=np.array(scale))


def measure_partition(levels, repeats):
    """Measure the actual base/EE/smoothing slicing omitted by shared timer."""
    n = levels[0][0].shape[1]
    N = levels[0][0].shape[0] // 12
    D, d = np.empty((114 * N, n)), np.empty(114 * N)
    for k in range(N):
        for j in range(2):
            D[114*k+12*j:114*k+12*(j+1)] = levels[j][0][12*k:12*(k+1)]
            d[114*k+12*j:114*k+12*(j+1)] = levels[j][1][12*k:12*(k+1)]
        D[114*k+24:114*(k+1)] = levels[2][0][90*k:90*(k+1)]
        d[114*k+24:114*(k+1)] = levels[2][1][90*k:90*(k+1)]
    measurements = []
    for _ in range(repeats):
        start = time.perf_counter()
        split = []
        for frame in range(2):
            Cs, ds = [], []
            for k in range(N):
                offset = 114*k + 12*frame
                Cs.append(D[offset:offset+12])
                ds.append(d[offset:offset+12])
            split.append((np.vstack(Cs), np.concatenate(ds)))
        ii = np.concatenate([np.arange(114*k+24, 114*(k+1)) for k in range(N)])
        split.append((D[ii], d[ii]))
        measurements.append(1000 * (time.perf_counter() - start))
        for actual, expected in zip(split, levels):
            assert np.array_equal(actual[0], expected[0])
            assert np.array_equal(actual[1], expected[1])
    return float(np.median(measurements))


def measure_task_lock(record, repeats):
    """Time the original QR lock after a solved level, excluding its solve."""
    if record['level'] >= 2:
        return 0.
    # Inputs to the lock were already built by build_reduced in the controller.
    qp = build_reduced(record)
    Cr, Gr = qp['task_C'], qp['G']
    active = record['active_rows']
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        face = np.vstack((Cr, Gr[active])) if len(active) else Cr
        norm = la.norm(face, axis=1)
        face = face[norm > 1e-10] / norm[norm > 1e-10, None]
        Q, R, _ = la.qr(face.T, mode='full', pivoting=True)
        diag = np.abs(np.diag(R))
        rank = int(np.sum(diag > 1e-10 * diag.max())) if len(diag) else 0
        next_Z = record['lift_Z'] @ Q[:, rank:]
        times.append(1000 * (time.perf_counter() - start))
        assert next_Z.shape[0] == 180
    return float(np.median(times))


def capture_weighted(plant, root, target, s, t, prev, anchor, repeats):
    measurements = []
    record = None
    for _ in range(repeats):
        mpc = OnlineMPC(plant, root, target, backend='osqp_screened', assembly='native')
        mpc.solve(s, t, prev, anchor[2], anchor=anchor, first_slew_time=.02)
        measurements.append(float(mpc.last_info['construction_ms']))
        qp = {key: mpc.debug[key].copy() for key in ('H', 'g', 'G', 'lo', 'hi')}
        if record is not None:
            assert digest_arrays(qp) == digest_arrays(record['qp'])
        record = dict(qp=qp, reference_x=mpc.debug['z'].copy(),
                      reference_dual=mpc.engine.last_dual.copy(),
                      reference_fastpath=bool(mpc.last_info.get('active_face_fastpath', 0)))
    return record, float(np.median(measurements))


def capture_hierarchy(plant, root, target, s, t, prev, anchor, repeats):
    measurements = []
    selected = None
    for _ in range(repeats):
        mpc = SharedUpper(plant, root, target, 'mpc')
        recorder = RecordingHierarchy()
        mpc.hierarchy = recorder
        mpc.solve(s, t, prev, anchor[2], anchor=anchor, first_slew_time=.02)
        measurements.append(float(mpc.last_info['construction_ms']))
        selected = recorder
    return selected, float(np.median(measurements))


def save_qp(out, entry, arrays):
    qp = {key: arrays[key] for key in ('H', 'g', 'G', 'lo', 'hi')}
    r = residuals(**dict(H=qp['H'], g=qp['g'], A=qp['G'], l=qp['lo'], u=qp['hi'],
                         x=arrays['reference_x'], dual=arrays['reference_dual']))
    error = max(r[key] for key in RESIDUAL_KEYS)
    if error > 2e-7:
        raise RuntimeError(f'Reference does not certify original saved QP: {r}')
    eig = la.eigvalsh(qp['H'])
    threshold = max(1., float(np.max(np.abs(eig)))) * 1e-10
    entry.update(n=len(qp['g']), m=len(qp['lo']), rank_H=int(np.sum(eig > threshold)),
                 rank_H_threshold=threshold, positive_definite_float=bool(eig[0] > 0),
                 min_eigenvalue=float(eig[0]), max_eigenvalue=float(eig[-1]),
                 qp_sha256=digest_arrays(qp), reference_kkt=float(error),
                 reference_residuals=r, reference_dual_available=True)
    arrays.update(A=qp['G'], l=qp['lo'], u=qp['hi'], meta_json=np.array(json.dumps(entry)))
    np.savez_compressed(out / entry['path'], **arrays)
    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None,
                        help='Maximum snapshots per case; includes calibration then heldout')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--cases', nargs='+', choices=['normal', 'double', 'push'],
                        default=['normal', 'double', 'push'])
    parser.add_argument('--output', default='results/structured_admm/qp_corpus')
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError('repeats must be positive')
    root = pathlib.Path(__file__).resolve().parents[1]
    history = root / 'results/online_wbc_mpc_50hz'
    out = root / args.output
    out.mkdir(parents=True, exist_ok=True)
    sources = ['capture_structured_qps.py', 'online_wbc_mpc.py', 'online_assembly.hpp',
               'shared_upper.py', 'shared_hierarchy.py', 'online_osqp.py',
               'central_relu.py', 'run_online_wbc_mpc.py', 'geometric_wbc.py', 'native.cpp']
    manifest = dict(schema_version=1, N=10, dt=.04, control_request_dt=.02,
                    reference_delay=.02, original_qp_tolerance=2e-7,
                    original_full_n=180, original_full_m=1340,
                    arguments=vars(args), source_sha256={name: hashlib.sha256(
                        (root / 'relu_hgmpc' / name).read_bytes()).hexdigest() for name in sources},
                    traces={}, target_sha256={}, entries=[], failures=[], snapshots=[],
                    timing_definition='construction_ms is shared model/QP assembly plus per-level reduction; '
                    'state_evaluation_ms is separate and must be included once per snapshot. '
                    'No reference solve, tracing, snapshot serialization or corpus hashing is included. '
                    'Hierarchy snapshot_common_construction_ms is counted once across all its levels; '
                    'level_construction_ms is counted for every level. '
                    'hierarchy_lock_construction_ms separately measures the exact QR lock after each reference level; '
                    'include it once per level when aggregating a complete hierarchy cycle.',
                    interpretation='Offline fixed-state replay only. No controller replaced and no new closed loop run. '
                    'Hierarchy QPs use one fixed original reference path and original unsanitized inherited rows. '
                    'No new Hessian regularization or changed bounds are introduced. '
                    'Original weighted formulation retains its existing 1e-9 Hessian regularizer.',
                    calibration_rule='Only state times 0 and 0.5 seconds are calibration. Remaining states are heldout.')
    times = TIMES if args.limit is None else TIMES[:args.limit]
    plant = native.Plant(str(root))
    plant.fast_reference = True
    # Model/dynamic-library loading is explicitly offline, not a control period.
    plant.evaluate(plant.truth()['q'], plant.truth()['v'])
    for case in args.cases:
        trace_path = history / f'{case}_fixed50_r1_trace.npz'
        target_case = 'normal' if case == 'push' else case
        target_path = history / f'target_{target_case}_12.npz'
        if not target_path.exists():
            raise FileNotFoundError(f'Original target cache required: {target_path}')
        target = Target(plant, target_case, 12., history)
        manifest['traces'][case] = dict(path=str(trace_path), sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest())
        manifest['target_sha256'][case] = target.hash
        with np.load(trace_path) as data:
            trace = {key: data[key] for key in ['q', 'v', 'tau', 'reference_q', 'reference_v', 'reference_a']}
        for state_time in times:
            k = round(state_time / .005)
            snapshot_id = f'{case}_{k:04d}'
            base = dict(snapshot_id=snapshot_id, case=case, state_time=state_time,
                        time=state_time + .02, index=k,
                        split='calibration' if state_time in CALIBRATION_TIMES else 'heldout')
            if k + 4 >= len(trace['q']):
                manifest['failures'].append(dict(**base, family='snapshot', error='History shorter than requested index'))
                continue
            state_ms = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                state = plant.evaluate(trace['q'][k], trace['v'][k])
                state_ms.append(1000 * (time.perf_counter() - start))
            base['state_evaluation_ms'] = float(np.median(state_ms))
            anchor = tuple(trace[key][k+4].copy() for key in ['reference_q', 'reference_v', 'reference_a'])
            previous_tau = trace['tau'][max(0, k-1)].copy()
            manifest['snapshots'].append(base)
            for family in ['weighted', 'hierarchy']:
                try:
                    if family == 'weighted':
                        record, common = capture_weighted(plant, root, target, state, base['time'], previous_tau, anchor, args.repeats)
                        entry = dict(**base, family=family, level=-1, level_label='weighted',
                                     path=f'{snapshot_id}_weighted.npz', construction_ms=common,
                                     snapshot_common_construction_ms=common, level_construction_ms=0.,
                                     reference_fastpath=record['reference_fastpath'])
                        arrays = dict(**record['qp'], reference_x=record['reference_x'], reference_dual=record['reference_dual'],
                                      task_C=np.zeros((0, 180)), task_d=np.zeros(0), lift_x=np.zeros(180), lift_Z=np.eye(180))
                        manifest['entries'].append(save_qp(out, entry, arrays))
                    else:
                        recorder, common = capture_hierarchy(plant, root, target, state, base['time'], previous_tau, anchor, args.repeats)
                        partition_ms = measure_partition(recorder.level_inputs, args.repeats)
                        common += partition_ms
                        for record in recorder.records:
                            level_ms = []
                            for _ in range(args.repeats):
                                start = time.perf_counter()
                                qp = build_reduced(record)
                                level_ms.append(1000 * (time.perf_counter() - start))
                            level_time = float(np.median(level_ms))
                            lock_time = measure_task_lock(record, args.repeats)
                            entry = dict(**base, family=family, level=record['level'],
                                         level_label=['base', 'end_effector', 'smoothing'][record['level']],
                                         path=f"{snapshot_id}_hierarchy_{record['level']}.npz",
                                         construction_ms=common + level_time,
                                         snapshot_common_construction_ms=common,
                                         hierarchy_partition_ms=partition_ms, level_construction_ms=level_time,
                                         hierarchy_lock_construction_ms=lock_time,
                                         reference_fastpath=record['reference_fastpath'])
                            arrays = dict(qp, **{key: record[key] for key in ['reference_x', 'reference_dual',
                                'lift_x', 'lift_Z', 'physical_G', 'physical_lo', 'physical_hi', 'active_rows']})
                            manifest['entries'].append(save_qp(out, entry, arrays))
                    print(snapshot_id, family, 'captured', flush=True)
                except Exception as exc:
                    manifest['failures'].append(dict(**base, family=family,
                                                     error=f'{type(exc).__name__}: {exc}'))
                    print(snapshot_id, family, 'FAILED', type(exc).__name__, str(exc), flush=True)
                (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(dict(entries=len(manifest['entries']), failures=len(manifest['failures']), output=str(out))))


if __name__ == '__main__':
    main()
