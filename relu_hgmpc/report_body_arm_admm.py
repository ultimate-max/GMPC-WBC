"""Honest offline report for physical body/arm GMPC consensus experiments.

The benchmark stores one paired record per state/repetition.  Accuracy gates
are computed by the benchmark from original physical quantities; this report
never converts a failed attempt into a successful speed comparison.
"""
import argparse
import hashlib
import json
import math
import pathlib
import statistics
from collections import Counter


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def distribution(values):
    a = sorted(float(v) for v in values if finite(v))
    if not a:
        return dict(count=0)
    p = .95 * (len(a) - 1)
    i = int(p)
    q95 = a[i] + (p - i) * (a[min(i + 1, len(a) - 1)] - a[i])
    return dict(count=len(a), mean=statistics.fmean(a), median=statistics.median(a),
                p95=q95, maximum=max(a), minimum=min(a))


def maximum(rows, key):
    vals = [r.get('metrics', {}).get(key) for r in rows]
    vals = [float(v) for v in vals if finite(v)]
    return max(vals) if vals else None


def last_level(row):
    levels = row.get('distributed', {}).get('info', {}).get('levels', [])
    return levels[-1] if levels else {}


def make_summary(result):
    rows = result['records']
    expected = result.get('expected_pairs', len(rows))
    good = [r for r in rows if r.get('accepted', False)]
    held = [r for r in rows if r.get('split') == 'heldout']
    held_ok = [r for r in held if r.get('accepted', False)]
    methods = {}
    for method in ('central', 'distributed'):
        attempted = [r.get(method, {}) for r in rows]
        accepted = [r.get(method, {}) for r in good]
        if method == 'central':
            certified_rows = [r for r in rows if not r.get('central', {}).get('error')
                              and all(finite(r.get('metrics', {}).get(k)) and
                                      r['metrics'][k] <= result.get('tolerances', {}).get('original_kkt', 2e-7)
                                      for k in ('central_kkt', 'central_lock_abs', 'central_bound_abs'))]
            converged_count = len(certified_rows)
        else:
            certified_rows = good
            converged_count = sum(bool(x.get('converged', False)) for x in attempted)
        certified_ids = {id(r) for r in certified_rows}
        methods[method] = dict(
            attempted_total_ms=distribution([x.get('total_ms') for x in attempted]),
            attempted_complete_prediction_ms=distribution([x.get('complete_prediction_ms') for x in attempted]),
            matched_success_total_ms=distribution([x.get('total_ms') for x in accepted]),
            errors=sum(bool(x.get('error')) for x in attempted),
            successful_hierarchy_attempts=converged_count,
            certified_comparison_attempts=len(certified_rows),
            successful_total_ms=distribution([r[method].get('total_ms') for r in certified_rows]),
            failed_attempt_total_ms=distribution([r.get(method, {}).get('total_ms') for r in rows
                                                  if id(r) not in certified_ids]),
            not_reached_last_level_attempts=(sum(x.get('completed_levels', 0) < result.get('expected_solver_levels', 3)
                                                 for x in attempted) if method == 'distributed' else 0),
            attempted_levels=distribution([x.get('completed_levels') for x in attempted]),
            timing_components={name: distribution([x.get(name) for x in attempted])
                               for name in ('state_evaluation_ms', 'construction_ms',
                                            'linearization_ms', 'condensing_ms',
                                            'solve_ms', 'factor_ms', 'iteration_ms',
                                            'coordinate_ms', 'check_ms', 'lock_ms',
                                            'recovery_ms', 'certificate_ms',
                                            'factorization_count', 'iterations')})
    matched = [r for r in good if finite(r.get('central', {}).get('total_ms'))
               and finite(r.get('distributed', {}).get('total_ms'))
               and r['distributed']['total_ms'] > 0]
    ct = distribution([r['central']['total_ms'] for r in matched])
    dt = distribution([r['distributed']['total_ms'] for r in matched])
    ratios = [r['central']['total_ms'] / r['distributed']['total_ms'] for r in matched]
    comparison = dict(matched_success_pairs=len(matched),
                      excluded_failure_pairs=len(rows) - len(matched),
                      speedup_definition='central_total_ms / distributed_total_ms',
                      ratio_of_means=ct['mean']/dt['mean'] if matched else None,
                      ratio_of_p95=ct['p95']/dt['p95'] if matched else None,
                      paired_speedups=distribution(ratios),
                      distributed_faster_pairs=sum(r > 1. for r in ratios))
    expected_heldout = result.get('expected_heldout_pairs', len(held))
    held_ct = distribution([r['central']['total_ms'] for r in held_ok
                            if finite(r.get('central', {}).get('total_ms'))])
    held_dt = distribution([r['distributed']['total_ms'] for r in held_ok
                            if finite(r.get('distributed', {}).get('total_ms'))])
    coverage = bool(result.get('formal_validation', False) and len(rows) == expected
                    and len(held) == expected_heldout and expected_heldout > 0)
    sources_stable = bool(result.get('provenance', {}).get('sources_unchanged_during_run', False))
    accuracy = coverage and sources_stable and len(held_ok) == expected_heldout
    faster_mean = bool(held_ok and held_ct.get('mean', math.inf) > held_dt.get('mean', math.inf))
    faster_p95 = bool(held_ok and held_ct.get('p95', math.inf) > held_dt.get('p95', math.inf))
    gate = dict(passed=bool(accuracy and faster_mean and faster_p95),
                complete_coverage=coverage, formal_validation=bool(result.get('formal_validation', False)),
                all_heldout_accurate=accuracy, sources_unchanged_during_run=sources_stable,
                mean_total_faster=faster_mean, p95_total_faster=faster_p95,
                expected_heldout_pairs=expected_heldout, observed_heldout_pairs=len(held),
                accepted_heldout_pairs=len(held_ok),
                rule='Full heldout coverage, every original-model/priority/interface accuracy gate, '
                     'and both complete-cycle mean and P95 improvement are required. '
                     'No closed-loop replacement is performed by this report.')
    metric_names = sorted({key for row in rows for key in row.get('metrics', {})
                           if finite(row.get('metrics', {}).get(key))})
    cases = {}
    for case in sorted({r['case'] for r in rows}):
        cr = [r for r in rows if r['case'] == case]
        cases[case] = dict(attempted=len(cr), accepted=sum(bool(r.get('accepted')) for r in cr),
                          maxima={k: maximum(cr, k) for k in metric_names})
    return dict(schema_version=1, arguments=result.get('arguments', {}),
                tolerances=result.get('tolerances', {}), attempted_pairs=len(rows),
                expected_pairs=expected, accepted_pairs=len(good), failed_pairs=len(rows)-len(good),
                unique_states=len({r['snapshot_id'] for r in rows}),
                methods=methods, comparison=comparison, by_case=cases,
                metric_maxima_all={k: maximum(rows, k) for k in metric_names},
                metric_maxima_accepted={k: maximum(good, k) for k in metric_names},
                closed_loop_gate=gate, timing=result.get('timing', {}),
                warm_start=result.get('warm_start', {}),
                provenance=result.get('provenance', {}), task_mode=result.get('task_mode', 'strict'),
                failed_at_level=dict(Counter(str(last_level(r).get('level', 'before_first'))
                                             for r in rows if not r.get('distributed', {}).get('converged', False))),
                last_native_status=dict(Counter(str(last_level(r).get('native', {}).get('status', 'unavailable'))
                                                for r in rows if not r.get('distributed', {}).get('converged', False))),
                failures=[dict(snapshot_id=r['snapshot_id'], case=r['case'],
                               repeat=r.get('repeat'), failed_gates=r.get('failed_gates', []),
                               central_error=r.get('central', {}).get('error'),
                               distributed_error=r.get('distributed', {}).get('error'))
                          for r in rows if not r.get('accepted')])


def num(v):
    return '未取得' if v is None else f'{v:.3f}'


def markdown(summary):
    gate = summary['closed_loop_gate']
    lines = ['# 四足本体—机械臂分布式 GMPC 离线验证', '',
             ('**离线接入门槛通过。** 可开展单独的 MPC＋WBC 闭环验证；本报告不证明闭环稳定性。'
              if gate['passed'] else
              '**离线接入门槛未通过。** 保持现有统一几何 WBC 和在线集中式路径，不将本轮候选接入闭环。'), '',
             f"共 {summary['unique_states']} 个历史状态、{summary['attempted_pairs']} 对测试；"
             f"通过 {summary['accepted_pairs']} 对，失败 {summary['failed_pairs']} 对。", '',
             '## 相同问题与测试边界', '',
             '- 模型保持当前冻结动力学、一阶局部几何任务和固定接触模式。',
             ('- 预测为 10 个节点、每节点 0.04 s；末端主任务、底盘名义位姿与奇异性偏好、原平滑任务依次为三级。'
              if summary.get('task_mode') == 'cooperative' else
              '- 预测为 10 个节点、每节点 0.04 s；底盘与末端共同一级，姿态及控制平滑为第二级。'
              if summary.get('task_mode') == 'joint' else
              '- 预测为 10 个节点、每节点 0.04 s；层级为底盘、末端、姿态及控制平滑。'),
             '- 分布式方法按物理安装接口拆分质量与惯性，协调接口运动及六维作用力。',
             '- 各层使用分布式方法自己的已求解任务输出构造下一层锁；集中式解不作为候选初值。',
             '- 数据来自保存的状态，未推进 MuJoCo；结果是离线求解验证，不是新闭环跟踪结果。', '',
             '## 尝试耗时与成功求解耗时', '']
    for key, label in [('central', '当前集中式层级求解'), ('distributed', '本体—机械臂共识 ADMM')]:
        stat = summary['methods'][key]['attempted_total_ms']
        lines.append(f"- {label}：全部尝试均值 {num(stat.get('mean'))} ms，"
                     f"P95 {num(stat.get('p95'))} ms。")
    comparison = summary['comparison']
    lines.extend(['', f"仅共同通过的 {comparison['matched_success_pairs']} 对："
                  f"均值加速比 {num(comparison['ratio_of_means'])}，"
                  f"P95 加速比 {num(comparison['ratio_of_p95'])}。比值大于 1 才表示分布式更快。",
                  f"失败或缺失的 {comparison['excluded_failure_pairs']} 对没有计入成功加速比，"
                  '但已保留在全部尝试耗时和失败清单中。',
                  f"集中式成功完成并通过自身精度检查 {summary['methods']['central']['successful_hierarchy_attempts']} 次；"
                  f"分布式成功完成层级求解 {summary['methods']['distributed']['successful_hierarchy_attempts']} 次。",
                  f"分布式有 {summary['methods']['distributed']['not_reached_last_level_attempts']} 次尚未到达末层；"
                  '到达末层仍可能在该层失败。记录中的 completed_levels 实际计数为尝试过的层数，不能当作成功层数。',
                  f"失败层级（从 0 编号）：{summary['failed_at_level']}。",
                  f"最后一个原生求解状态：{summary['last_native_status']}。",
                  '失败尝试耗时不能解释为合格完成整个优化问题所需的时间。', '', '## 精度与接口一致性', ''])
    for key, value in summary['metric_maxima_all'].items():
        lines.append(f'- `{key}`：{value:.6e}。')
    lines.extend(['', '验收阈值：', ''])
    for key, value in summary['tolerances'].items():
        lines.append(f'- `{key}`：{value}。')
    lines.extend(['', '## 接入条件', '',
                  f"- 验证集覆盖完整：{gate['complete_coverage']}。",
                  f"- 验证集全部精度通过：{gate['all_heldout_accurate']}。",
                  f"- 验证集完整耗时均值改善：{gate['mean_total_faster']}。",
                  f"- 验证集完整耗时 P95 改善：{gate['p95_total_faster']}。", '',
                  '模型等价、有限迭代求解精度、总耗时和闭环稳定性是不同问题。'
                  '本报告仅按前述条件决定是否具备开展下一步闭环实验的依据。', '',
                  '## 计时与复现', '', str(summary['timing']), '', str(summary['warm_start']), ''])
    lines.extend(['运行期间源码未变的标记仅描述采集原始 results.json 的实验时间段。'
                  '此报告可以在实验结束后修订统计名称和说明，当前报告源文件哈希另行记录；'
                  '原始数据、冻结求解器和运行期间的源文件哈希不因此改写。', ''])
    if summary['failures']:
        lines.extend(['## 失败记录', ''])
        for failure in summary['failures']:
            lines.append(f"- {failure['snapshot_id']} / repeat {failure['repeat']}: "
                         f"{', '.join(failure['failed_gates']) or failure.get('distributed_error') or failure.get('central_error') or 'accuracy gate'}")
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=pathlib.Path)
    parser.add_argument('--output', type=pathlib.Path)
    args = parser.parse_args()
    result = json.loads(args.results.read_text())
    summary = make_summary(result)
    summary['results_sha256'] = hashlib.sha256(args.results.read_bytes()).hexdigest()
    summary['report_source_sha256'] = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
    out = args.output or args.results.parent
    out.mkdir(parents=True, exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (out/'RESULTS.md').write_text(markdown(summary))
    print(json.dumps(dict(output=str(out), gate=summary['closed_loop_gate']), ensure_ascii=False))


if __name__ == '__main__':
    main()
