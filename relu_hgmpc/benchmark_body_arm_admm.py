"""Paired physical body/arm decomposition benchmark; never drives MuJoCo.

The two methods receive the same saved truth, reference anchor, torque slew
initial condition, task source and horizon.  The candidate builds and solves
its own hierarchy; centralized task locks/solutions are only used afterwards
for validation.  Benchmark I/O and full algebraic equivalence audits are not
hidden inside a claimed online speedup.
"""
import argparse
import hashlib
import inspect
import json
import math
import os
import pathlib
import sys
import time

import numpy as np

import native
from body_arm_problem import BodyArmBuilder
from body_arm_gmpc import BodyArmHierarchy
from body_arm_tasks import make_task_layout
from capture_structured_qps import TIMES, CALIBRATION_TIMES
from run_online_wbc_mpc import Target
from shared_hierarchy import HierarchyQP
from shared_upper import SharedUpper
from report_body_arm_admm import make_summary, markdown


ROOT = pathlib.Path(__file__).resolve().parents[1]
CASES = ('normal', 'double', 'push')


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def scalar_details(value, prefix=''):
    """Retain scalar diagnostics without serializing all QP matrices in JSON."""
    out = {}
    if isinstance(value, dict):
        for key, val in value.items():
            name = prefix + str(key)
            if isinstance(val, (float, int, str, bool, np.generic)) or val is None:
                out[name] = plain(val)
            elif isinstance(val, dict):
                out.update(scalar_details(val, name + '.'))
    return out


def norminf(x):
    return float(np.max(np.abs(x))) if np.size(x) else 0.


class AssemblyCaptured(Exception):
    pass


class CaptureAssembly:
    """Stop unmodified native assembly before any centralized solve occurs."""
    def solve(self, levels, G, lo, hi, initial):
        self.levels, self.G, self.lo, self.hi, self.initial = levels, G, lo, hi, initial
        raise AssemblyCaptured()

    @property
    def debug(self):
        return dict(G=self.G, lo=self.lo, hi=self.hi)


def source_hashes():
    names = ('shared_upper.py', 'shared_hierarchy.py', 'online_assembly.hpp',
             'online_osqp.py', 'geometric_wbc.py', 'native.cpp',
             'body_arm_interface.py', 'body_arm_problem.py', 'body_arm_gmpc.py', 'body_arm_tasks.py',
             'body_arm_admm_solver.py',
             'body_arm_admm_native/solver.cpp', 'body_arm_admm_native/CMakeLists.txt',
             'benchmark_body_arm_admm.py', 'report_body_arm_admm.py')
    return {name: hashlib.sha256((ROOT/'relu_hgmpc'/name).read_bytes()).hexdigest()
            for name in names if (ROOT/'relu_hgmpc'/name).exists()}


def physical_cores():
    cores = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        base = pathlib.Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key = tuple((base/f).read_text().strip() for f in ('physical_package_id', 'core_id'))
        cores.setdefault(key, cpu)
    return list(cores.values())


def history(case):
    folder = ROOT/'results/online_wbc_mpc_50hz'
    path = folder/f'{case}_fixed50_r1_trace.npz'
    with np.load(path) as d:
        trace = {key: d[key].copy() for key in
                 ('q', 'v', 'tau', 'reference_q', 'reference_v', 'reference_a')}
    return trace, path


def original_constraints(debug, u):
    gu = debug['G'] @ u
    return float(max(0., np.max(debug['lo'] - gu), np.max(gu - debug['hi'])))


def effective_task_audit(problem, layout, u):
    """Independent local row pullback, including the new singularity rows."""
    result = dict(matrix_error=[], bias_error=[], residual_error=[], row_coverage_errors=0)
    for level, (C, d) in enumerate(layout.global_levels):
        pulled_C, pulled_d = np.zeros_like(C), np.zeros_like(d)
        seen = np.zeros(len(d), int)
        for side in ('body', 'arm'):
            local_C, local_d = getattr(layout, side+'_levels')[level]
            ids = np.asarray(layout.metadata['row_ids'][side][level], dtype=int)
            lift, offset = getattr(problem, 'lift_'+side), getattr(problem, 'offset_'+side)
            pulled_C[ids] = local_C@lift
            pulled_d[ids] = local_C@offset+local_d
            seen[ids] += 1
        result['row_coverage_errors'] += int(np.count_nonzero(seen != 1))
        result['matrix_error'].append(norminf(pulled_C-C))
        result['bias_error'].append(norminf(pulled_d-d))
        result['residual_error'].append(norminf(pulled_C@u+pulled_d-C@u-d))
    return result


def normalize_candidate_info(info, wall):
    values = scalar_details(info)
    result = dict(total_ms=wall, solve_ms=wall, info=plain(info))
    result.update({k: plain(v) for k, v in info.items()
                   if isinstance(v, (int, float, str, bool, np.generic))})
    timing = info.get('timing', {})
    result.update({k: plain(v) for k, v in timing.items()
                   if isinstance(v, (int, float, np.generic))})
    result['diagnostics'] = values
    result['coordinate_ms'] = sum(float(stage.get('native', {}).get('coord_ms', 0.))
                                  for stage in info.get('levels', []))
    result['check_ms'] = sum(float(stage.get('native', {}).get('check_ms', 0.))
                             for stage in info.get('levels', []))
    result['lock_ms'] = float(info.get('task_lock_ms', 0.))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    p.add_argument('--times', nargs='+', type=float)
    p.add_argument('--limit', type=int)
    p.add_argument('--split', choices=('all', 'calibration', 'heldout'), default='all')
    p.add_argument('--priority', choices=('cooperative', 'joint', 'strict'), default='cooperative')
    p.add_argument('--repeat', type=int, default=1)
    p.add_argument('--max-iter', type=int, default=10000)
    p.add_argument('--rho', type=float, default=.1)
    p.add_argument('--sigma', type=float, default=1e-6)
    p.add_argument('--alpha', type=float, default=1.6)
    p.add_argument('--adaptive-interval', type=int, default=50)
    p.add_argument('--serial', action='store_true')
    p.add_argument('--tol', type=float, default=2e-7)
    p.add_argument('--kernel-tol', type=float,
                   help='Optional tighter inner stopping tolerance; external common gates do not change.')
    p.add_argument('--kernel-interface-tol', type=float)
    p.add_argument('--preconditioner-shift', type=float)
    p.add_argument('--preconditioner-interface-weight', type=float)
    p.add_argument('--screen-bounds', action='store_true')
    p.add_argument('--objective-multiplier', type=float, default=1.,
                   help='Common positive scaling of both local objectives; original weights unchanged.')
    p.add_argument('--singularity-weight', type=float, default=1.)
    p.add_argument('--singularity-eps', type=float, default=.05)
    p.add_argument('--length-scale', type=float, default=.5)
    p.add_argument('--task-tol', type=float, default=2e-7)
    p.add_argument('--torque-tol', type=float, default=1e-4)
    p.add_argument('--output', default='results/body_arm_admm/calibration')
    args = p.parse_args()
    if args.repeat < 1 or args.max_iter < 1 or args.objective_multiplier <= 0:
        raise ValueError('positive repetition/iteration count required')
    times = list(TIMES if args.times is None else args.times)
    times = [v for v in times if args.split == 'all' or
             (v in CALIBRATION_TIMES) == (args.split == 'calibration')]
    if args.limit is not None:
        times = times[:args.limit]
    if not times:
        raise ValueError('No sample times selected')
    if len(set(args.cases)) != len(args.cases) or len(set(times)) != len(times):
        raise ValueError('Duplicate cases or sample times would bias paired statistics')
    if min(times) < 0:
        raise ValueError('Historical state times must be nonnegative')
    cores = physical_cores()
    selected = cores[:1 if args.serial else 2]
    if len(selected) < (1 if args.serial else 2):
        raise RuntimeError('Requested number of separate physical cores unavailable')
    os.sched_setaffinity(0, set(selected))
    # Native OpenMP workers inherit this affinity.  BLAS remains single-threaded.
    out = ROOT/args.output
    out.mkdir(parents=True, exist_ok=True)
    (out/'samples').mkdir(exist_ok=True)
    options = dict(tol=args.tol, interface_tol=args.tol, bound_tol=args.tol,
                   max_iter=args.max_iter, rho=args.rho, sigma=args.sigma,
                   alpha=args.alpha, adaptive_interval=args.adaptive_interval,
                   parallel=not args.serial, objective_multiplier=args.objective_multiplier)
    if args.kernel_tol is not None:
        options['kernel_tol'] = args.kernel_tol
    if args.kernel_interface_tol is not None:
        options['kernel_interface_tol'] = args.kernel_interface_tol
    if args.preconditioner_shift is not None:
        options['preconditioner_shift'] = args.preconditioner_shift
    if args.preconditioner_interface_weight is not None:
        options['preconditioner_interface_weight'] = args.preconditioner_interface_weight
    if args.screen_bounds:
        options['screen_bounds'] = True
    nheld = sum(t not in CALIBRATION_TIMES for t in times)
    formal = (set(args.cases) == set(CASES)
              and set(t for t in times if t not in CALIBRATION_TIMES)
              == set(TIMES)-CALIBRATION_TIMES)
    result = dict(schema_version=1, arguments=vars(args), options=options,
                  N=10, dt=.04, control_request_dt=.02, first_slew_time=.02,
                  task_mode=args.priority, expected_solver_levels=2 if args.priority == 'joint' else 3,
                  expected_pairs=len(args.cases)*len(times)*args.repeat,
                  expected_heldout_pairs=len(args.cases)*nheld*args.repeat,
                  formal_validation=formal,
                  tolerances=dict(original_kkt=args.tol, original_bound_abs=args.tol,
                                  interface_abs=args.tol, hierarchy_task_output_abs=args.task_tol,
                                  torque_abs_Nm=args.torque_tol),
                  timing=dict(qp_endpoint='Same measured state evaluation plus physical linearization, '
                      'condensing/assembly, shared singularity gradient, method-specific task rows, '
                      'hierarchy locks, factorization, iterations and routine certificates. '
                      'Central assembly is the unmodified native SharedUpper call stopped at its solver '
                      'entry; the original HierarchyQP then solves the complete effective hierarchy. '
                      'No reference solve, task lock or optimum is used to initialize ADMM. '
                      'Distributed build/solve wall includes Python/native conversion. '
                      'MuJoCo, Plan creation and nonlinear trajectory diagnostics are excluded from BOTH endpoints.',
                      complete_prediction='QP endpoint plus affine predicted trajectory recovery is reported '
                      'separately for each method. Routine candidate certificates are already in solve wall. '
                      'Reference-solution comparison and full algebraic equivalence audit are offline-only; '
                      'neither is charged to online total. Plan construction and nonlinear defects remain excluded.',
                      comparison='Published acceleration gate uses QP endpoint only; complete prediction '
                      'and certificate costs are disclosed and must be considered before closed-loop scheduling.'),
                  warm_start='Both receive the recorded previous torque repeated over the horizon. '
                      'No centralized solution or active face is passed to ADMM. Reference solver workspace '
                      'may persist within each case/repetition, as in the existing online implementation.',
                  provenance=dict(source_sha256=source_hashes(), cpu_affinity=selected,
                                  numpy=np.__version__, histories={}, target_hashes={}),
                  records=[])
    plant = native.Plant(str(ROOT))
    plant.fast_reference = True
    plant.evaluate(plant.truth()['q'], plant.truth()['v'])
    scale = np.r_[np.full(12, 100.), np.full(6, 10.)]
    for repeat in range(args.repeat):
        for case in args.cases:
            trace, path = history(case)
            result['provenance']['histories'][case] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            target = Target(plant, 'normal' if case == 'push' else case, 12., path.parent)
            result['provenance']['target_hashes'][case] = target.hash
            central = SharedUpper(plant, ROOT, target, 'mpc')
            capture = CaptureAssembly()
            central.hierarchy = capture
            reference_solver = HierarchyQP()
            builder = BodyArmBuilder(plant, ROOT, target, N=10, dt=.04)
            distributed = BodyArmHierarchy(options, priority=args.priority)
            for sample_index, state_time in enumerate(times):
                k = round(state_time/.005)
                row = dict(snapshot_id=f'{case}_{k:04d}', case=case, repeat=repeat,
                           state_time=state_time, time=state_time+.02,
                           split='calibration' if state_time in CALIBRATION_TIMES else 'heldout',
                           accepted=False, failed_gates=[], central={}, distributed={}, metrics={})
                result['records'].append(row)
                try:
                    if k+4 >= len(trace['q']):
                        raise IndexError('Historical trace shorter than requested state')
                    anchor = tuple(trace[key][k+4].copy() for key in
                                   ('reference_q', 'reference_v', 'reference_a'))
                    prev = trace['tau'][max(0, k-1)].copy()
                    begin = time.perf_counter()
                    state = plant.evaluate(trace['q'][k], trace['v'][k])
                    state_ms = 1000*(time.perf_counter()-begin)
                    initial = np.tile(prev/scale, 10)
                    objects = {}
                    begin = time.perf_counter()
                    try:
                        central.solve(state, state_time+.02, prev, anchor[2],
                                      anchor=anchor, first_slew_time=.02)
                    except AssemblyCaptured:
                        pass
                    else:
                        raise RuntimeError('Original assembly capture was not reached')
                    central_build_ms = 1000*(time.perf_counter()-begin)
                    begin = time.perf_counter()
                    problem = builder.build(state, state_time+.02, prev, anchor,
                                            first_slew_time=.02)
                    local_build_ms = 1000*(time.perf_counter()-begin)
                    layout = make_task_layout(problem, priority=args.priority, q=state['q'],
                        model=builder.interface.full, original_levels=capture.levels,
                        original_G=capture.G, original_lo=capture.lo,
                        singularity_weight=args.singularity_weight,
                        singularity_eps=args.singularity_eps, length_scale=args.length_scale)
                    problem.task_layout = layout
                    # Count the shared geometric gradient once in each hypothetical
                    # standalone controller.  Each side pays its own task-row work.
                    if all(hasattr(layout, key) for key in ('common_grad_ms', 'central_rows_ms', 'local_rows_ms')):
                        common_grad_ms = float(layout.common_grad_ms)
                        original_rows_ms = float(getattr(layout, 'original_rows_ms', 0.))
                        overhead_ms = max(0., float(layout.construction_ms)-common_grad_ms-
                                          float(layout.central_rows_ms)-float(layout.local_rows_ms)-original_rows_ms)
                        common_task_ms = common_grad_ms+original_rows_ms+overhead_ms
                        central_task_ms = common_task_ms+float(layout.central_rows_ms)
                        local_task_ms = common_task_ms+float(layout.local_rows_ms)
                        task_clock = 'shared_gradient_plus_own_rows'
                    else:
                        common_grad_ms = 0.
                        central_task_ms = local_task_ms = float(layout.construction_ms)
                        task_clock = 'full_layout_construction_charged_to_both'
                    row['task_layout'] = plain(layout.metadata)
                    row['task_layout']['timing_policy'] = task_clock
                    row['actual_solver_levels'] = len(layout.global_levels)
                    row['assembly'] = dict(central_native_ms=central_build_ms,
                                           distributed_local_ms=local_build_ms,
                                           common_gradient_ms=common_grad_ms,
                                           central_task_ms=central_task_ms,
                                           distributed_task_ms=local_task_ms,
                                           complete_layout_ms=float(layout.construction_ms))
                    objects['problem'] = problem

                    def run_central():
                        begin = time.perf_counter()
                        try:
                            u, info = reference_solver.solve(layout.global_levels, capture.G,
                                                             capture.lo, capture.hi, initial)
                            solve_ms = 1000*(time.perf_counter()-begin)
                            data = dict(info, state_evaluation_ms=state_ms,
                                        construction_ms=central_build_ms+central_task_ms,
                                        common_gradient_ms=common_grad_ms,
                                        task_construction_ms=central_task_ms,
                                        solve_ms=solve_ms,
                                        total_ms=state_ms+central_build_ms+central_task_ms+solve_ms,
                                        priority=args.priority, timing_complete=True)
                            row['central'] = plain(data)
                            objects['central_u'] = np.asarray(u).copy()
                        except Exception as error:
                            elapsed = 1000*(time.perf_counter()-begin)
                            row['central'] = dict(error=f'{type(error).__name__}: {error}',
                                failed_attempt_elapsed_ms=elapsed, timing_complete=False,
                                total_ms=state_ms+central_build_ms+central_task_ms+elapsed)

                    def run_distributed():
                        begin = time.perf_counter()
                        try:
                            u, info = distributed.solve(problem, initial)
                            solve_ms = 1000*(time.perf_counter()-begin)
                            data = normalize_candidate_info(info, solve_ms)
                            data.update(state_evaluation_ms=state_ms,
                                        construction_ms=local_build_ms+local_task_ms,
                                        common_gradient_ms=common_grad_ms,
                                        task_construction_ms=local_task_ms,
                                        solve_ms=solve_ms,
                                        total_ms=state_ms+local_build_ms+local_task_ms+solve_ms,
                                        timing_complete=True)
                            row['distributed'] = data
                            objects.update(problem=problem, distributed_u=np.asarray(u),
                                           distributed_info=info, local=distributed.last_local)
                        except Exception as error:
                            elapsed = 1000*(time.perf_counter()-begin)
                            row['distributed'] = dict(error=f'{type(error).__name__}: {error}',
                                failed_attempt_elapsed_ms=elapsed, timing_complete=False,
                                total_ms=state_ms+local_build_ms+local_task_ms+elapsed)

                    # Counterbalance order without changing either warm-start policy.
                    for run in ((run_central, run_distributed) if (sample_index+repeat)%2 == 0
                                else (run_distributed, run_central)):
                        run()
                    if 'central_u' not in objects or 'distributed_u' not in objects:
                        row['failed_gates'].append('solver_error')
                    else:
                        cref, cand = objects['central_u'], objects['distributed_u']
                        if not np.all(np.isfinite(cref)) or not np.all(np.isfinite(cand)):
                            row['failed_gates'].append('nonfinite_control')
                        levels = capture.levels
                        problem = objects['problem']
                        metrics = row['metrics']
                        begin = time.perf_counter()
                        metrics['original_bound_abs'] = original_constraints(capture.debug, cand)
                        for i, (C, d) in enumerate(levels):
                            metrics[f'level_{i}_task_output_abs'] = norminf(C@(cand-cref))
                            cr, ca = C@cref+d, C@cand+d
                            metrics[f'level_{i}_objective_gap_abs'] = abs(float(ca@ca-cr@cr))
                        for i, (C, d) in enumerate(layout.global_levels):
                            metrics[f'effective_level_{i}_task_output_abs'] = norminf(C@(cand-cref))
                            cr, ca = C@cref+d, C@cand+d
                            metrics[f'effective_level_{i}_objective_gap_abs'] = abs(float(ca@ca-cr@cr))
                        difference = (cand-cref).reshape(10, 18)*scale
                        metrics['torque_horizon_abs_Nm'] = norminf(difference)
                        metrics['torque_first_abs_Nm'] = norminf(difference[0])
                        metrics['scaled_control_abs'] = norminf(cand-cref)
                        metrics['central_kkt'] = float(row['central']['hierarchy_kkt'])
                        metrics['central_lock_abs'] = float(row['central']['hierarchy_lock'])
                        metrics['central_bound_abs'] = original_constraints(capture.debug, cref)
                        row['offline_solution_comparison_ms'] = 1000*(time.perf_counter()-begin)
                        begin = time.perf_counter()
                        problem.split.rollout_global(cref.reshape(10, 18)*scale,
                            problem.initial_p, problem.initial_v, problem.dt, problem.full_maps)
                        row['central']['recovery_ms'] = 1000*(time.perf_counter()-begin)
                        row['central']['complete_prediction_ms'] = (
                            row['central']['total_ms']+row['central']['recovery_ms'])
                        begin = time.perf_counter()
                        audit = problem.audit_equivalence(levels, capture.G, capture.lo, capture.hi, cref)
                        row['offline_equivalence_audit_ms'] = 1000*(time.perf_counter()-begin)
                        audit_max = max((norminf(value) for value in audit.values()), default=0.)
                        audit['passed'] = bool(audit_max <= 1e-8)
                        metrics['algebraic_equivalence_max'] = audit_max
                        row['equivalence'] = plain(audit)
                        effective_audit = effective_task_audit(problem, layout, cref)
                        effective_audit_max = max(norminf(value) for value in effective_audit.values())
                        effective_audit['passed'] = bool(effective_audit_max <= 1e-8)
                        row['effective_task_equivalence'] = plain(effective_audit)
                        metrics['effective_task_equivalence_max'] = effective_audit_max
                        begin = time.perf_counter()
                        recovery = problem.recover(*objects['local'])
                        row['distributed']['recovery_ms'] = 1000*(time.perf_counter()-begin)
                        row['recovery_diagnostics'] = scalar_details(recovery)
                        metrics.update({'recovery_'+key: float(value)
                                        for key, value in recovery.get('residual_max', {}).items()})
                        row['distributed']['complete_prediction_ms'] = (
                            row['distributed']['total_ms']+row['distributed']['recovery_ms'])
                        info = objects['distributed_info']
                        if info.get('completed_levels') != result['expected_solver_levels']:
                            row['failed_gates'].append('incomplete_or_mismatched_hierarchy')
                        if 'priority' in info and info['priority'] != args.priority:
                            row['failed_gates'].append('priority_mode_mismatch')
                        if not all(np.all(np.isfinite(value)) for value in objects['local']):
                            row['failed_gates'].append('nonfinite_local_variables')
                        for key in ('original_kkt', 'actual_projected_stationarity', 'physical_bound', 'lock_error',
                                    'inherited_projection_error', 'interface_acceleration_raw',
                                    'interface_wrench_raw', 'local_prediction_error'):
                            vals = [stage[key] for stage in info.get('levels', []) if key in stage]
                            if vals:
                                metrics['distributed_'+key] = max(float(v) for v in vals)
                        for key in ('qp_primal_raw', 'qp_primal_relative', 'qp_dual_raw',
                                    'qp_dual_relative', 'qp_normal_relative'):
                            vals = [stage.get('original_residuals', {}).get(key)
                                    for stage in info.get('levels', [])]
                            vals = [float(v) for v in vals if v is not None]
                            if vals:
                                metrics['distributed_'+key] = max(vals)
                        for key, value in info.get('validation', {}).items():
                            if isinstance(value, (int, float, np.generic)):
                                metrics[key] = float(value)
                        # Candidate's own hierarchy certificate includes interface consistency.
                        if not info.get('converged', False):
                            row['failed_gates'].append('distributed_not_converged')
                        if info.get('validated') is False:
                            row['failed_gates'].append('distributed_original_certificate')
                        if not audit.get('passed', False):
                            row['failed_gates'].append('algebraic_equivalence')
                        if not effective_audit['passed']:
                            row['failed_gates'].append('effective_task_equivalence')
                        if max(recovery.get('residual_max', {}).get(key, math.inf) for key in
                               ('interface_acceleration', 'interface_velocity',
                                'interface_position_tangent', 'interface_wrench')) > args.tol:
                            row['failed_gates'].append('recovered_interface_agreement')
                        if max(metrics['central_kkt'], metrics['central_lock_abs'], metrics['central_bound_abs']) > args.tol:
                            row['failed_gates'].append('central_accuracy')
                        if metrics['original_bound_abs'] > args.tol:
                            row['failed_gates'].append('original_physical_bound')
                        if max(metrics[f'level_{i}_task_output_abs'] for i in range(3)) > args.task_tol:
                            row['failed_gates'].append('original_task_outputs')
                        if max(metrics[f'effective_level_{i}_task_output_abs']
                               for i in range(len(layout.global_levels))) > args.task_tol:
                            row['failed_gates'].append('effective_task_outputs')
                        if metrics['torque_horizon_abs_Nm'] > args.torque_tol:
                            row['failed_gates'].append('physical_torque_agreement')
                        row['accepted'] = not row['failed_gates']
                        np.savez_compressed(out/'samples'/f"{row['snapshot_id']}_r{repeat}.npz",
                            q=state['q'], v=state['v'], prev_tau=prev, anchor_q=anchor[0],
                            anchor_v=anchor[1], anchor_a=anchor[2], central_u=cref, distributed_u=cand,
                            body_x=objects['local'][0], arm_x=objects['local'][1])
                except Exception as error:
                    row['error'] = f'{type(error).__name__}: {error}'
                    row['failed_gates'].append('benchmark_error')
                print(row['snapshot_id'], repeat, 'PASS' if row['accepted'] else 'FAIL',
                      row['failed_gates'], flush=True)
                (out/'results.json').write_text(json.dumps(plain(result), indent=2, ensure_ascii=False))
    result['provenance']['source_sha256_after'] = source_hashes()
    result['provenance']['sources_unchanged_during_run'] = (
        result['provenance']['source_sha256'] == result['provenance']['source_sha256_after'])
    (out/'results.json').write_text(json.dumps(plain(result), indent=2, ensure_ascii=False))
    summary = make_summary(result)
    (out/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (out/'RESULTS.md').write_text(markdown(summary))
    print(json.dumps(dict(output=str(out), accepted=summary['accepted_pairs'],
                         attempts=summary['attempted_pairs'], gate=summary['closed_loop_gate'])))


if __name__ == '__main__':
    main()
