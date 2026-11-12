"""Report paired, fixed-QP replay without concealing unsuccessful solves.

The benchmark's cycle records already add common assembly exactly once and
are the authoritative source for full-hierarchy time.  Per-QP total_ms must
never be summed to form a three-level cycle.
"""
import argparse
import hashlib
import json
import math
import pathlib
import statistics


ROOT = pathlib.Path(__file__).resolve().parents[1]
LABELS = {
    'osqp_full': 'OSQP full (raw)',
    'osqp_preconditioned': 'OSQP same preconditioner',
    'structured_admm': 'Structured ADMM',
    'osqp_screened': 'Existing screened OSQP',
    'existing_hierarchy': 'Existing hybrid hierarchy',
}
ORDER = list(LABELS)
KEYS = ('qp_primal_relative', 'qp_dual_relative', 'qp_normal_relative')


def finite(values):
    return [float(x) for x in values if isinstance(x, (int, float)) and math.isfinite(x)]


def distribution(values):
    a = sorted(finite(values))
    if not a:
        return {}
    pos = .95 * (len(a) - 1)
    lower = int(pos)
    upper = min(lower + 1, len(a) - 1)
    p95 = a[lower] + (pos - lower) * (a[upper] - a[lower])
    return dict(count=len(a), mean=statistics.fmean(a), median=statistics.median(a),
                p95=p95, maximum=max(a), minimum=min(a))


def maximum(rows, key):
    a = finite([r.get(key) for r in rows])
    return max(a) if a else None


def fmt(value, decimals=3):
    if value is None:
        return '—'
    return f'{value:.{decimals}f}'


def expfmt(value):
    return '—' if value is None else f'{value:.3e}'


def paired(cycles, family, baseline):
    def keyed(name):
        return {(c['repeat'], c['snapshot_id']): c for c in cycles
                if c['family'] == family and c['solver'] == name}
    new = keyed('structured_admm')
    old = keyed(baseline)
    shared = sorted(new.keys() & old.keys())
    ok = [key for key in shared if new[key]['accepted'] and old[key]['accepted']]
    new_times = [new[key]['total_ms'] for key in ok]
    old_times = [old[key]['total_ms'] for key in ok]
    ratios = [old[key]['total_ms'] / new[key]['total_ms'] for key in ok]
    ns, os = distribution(new_times), distribution(old_times)
    return dict(family=family, baseline=baseline, speedup_definition='baseline_ms / structured_admm_ms',
                attempted_shared_pairs=len(shared), common_success_pairs=len(ok),
                excluded_failure_pairs=len(shared)-len(ok),
                missing_admm_pairs=len(old.keys()-new.keys()),
                missing_baseline_pairs=len(new.keys()-old.keys()),
                speedup_ratio_of_means=os['mean']/ns['mean'] if ok else None,
                speedup_ratio_of_p95=os['p95']/ns['p95'] if ok else None,
                paired_speedups=distribution(ratios),
                admm_faster_pairs=sum(v > 1 for v in ratios),
                admm_total_ms_common_success=ns, baseline_total_ms_common_success=os)


def make_summary(result, manifest):
    records = result['records']
    cycles = result['cycles']
    selected = result['args']['solvers'].split(',')
    tol = result['tolerances']['original_primal_dual_normal']
    physical_tol = result['tolerances']['physical_primal_abs']
    qp_stats = {}
    for family in ('weighted', 'hierarchy'):
        for method in selected:
            rows = [r for r in records if r['family'] == family and r['solver'] == method and r.get('applicable')]
            good = [r for r in rows if r.get('accepted')]
            bad = [r for r in rows if not r.get('accepted')]
            key = family + '/' + method
            qp_stats[key] = dict(attempted=len(rows), accepted=len(good), failed=len(bad),
                missing_certificate=sum(not all(k in r for k in KEYS) for r in rows),
                exceeded_original_kkt_count=sum(all(k in r for k in KEYS) and max(r[k] for k in KEYS)>tol for r in rows),
                exceeded_physical_primal_count=sum(r.get('qp_primal_raw', 0)>physical_tol for r in rows),
                original_kkt_max_all=maximum([dict(kkt=max(r[k] for k in KEYS)) for r in rows if all(k in r for k in KEYS)], 'kkt'),
                original_kkt_max_accepted=maximum([dict(kkt=max(r[k] for k in KEYS)) for r in good], 'kkt'),
                physical_primal_max_all=maximum(rows, 'qp_primal_raw'),
                objective_gap_relative_max_all=maximum(rows, 'objective_gap_relative'),
                objective_gap_relative_max_accepted=maximum(good, 'objective_gap_relative'),
                objective_gap_absolute_max_all=maximum(rows, 'objective_gap_absolute'),
                task_output_reference_difference_max_all=maximum(rows, 'task_output_reference_difference'),
                task_output_reference_difference_max_accepted=maximum(good, 'task_output_reference_difference'),
                solver_wall_ms_all_attempts=distribution([r.get('solver_wall_ms') for r in rows]),
                iterations_all_attempts=distribution([r.get('iterations') for r in rows]),
                certified_fastpath_calls=sum(bool(r.get('active_face_fastpath')) for r in good),
                total_factorization_count=distribution([r['factorization_count']+r.get('coordinate_factorization_count', 0)
                    for r in rows if 'factorization_count' in r]),
                diagnostics={name: distribution([r.get(name) for r in rows]) for name in (
                    'preprocess_ms', 'factor_ms', 'coordinate_factor_ms', 'coordinate_transform_ms',
                    'iteration_ms', 'check_ms', 'certificate_ms', 'setup_update_ms',
                    'factorization_count', 'rho_updates', 'osqp_rho_updates')})
    cycle_stats = {}
    for key, published in result['cycle_summary'].items():
        family, method = key.split('/')
        rows = [c for c in cycles if c['family'] == family and c['solver'] == method]
        passed = [c for c in rows if c['accepted']]
        assert published['count'] == len(rows)
        assert published['accepted'] == len(passed)
        cycle_stats[key] = dict(published, failed=len(rows)-len(passed),
            acceptance_rate=len(passed)/len(rows) if rows else None,
            attempted_total_ms=distribution([c['total_ms'] for c in rows]),
            accepted_total_ms=distribution([c['total_ms'] for c in passed]),
            common_construction_ms=distribution([c['common_construction_ms'] for c in rows]))
    comparisons = [paired(cycles, family, method) for family in ('weighted', 'hierarchy')
                   for method in selected if method != 'structured_admm'
                   and any(c['family'] == family and c['solver'] == method for c in cycles)]
    requested_split = result['args']['split']
    selected_entries = [e for e in manifest['entries'] if requested_split == 'all' or e['split'] == requested_split]
    expected_cycles = len({e['snapshot_id'] for e in selected_entries}) * result['args']['repeat']
    gate = {}
    for family, baseline in [('weighted', 'osqp_screened'), ('hierarchy', 'existing_hierarchy')]:
        candidate = cycle_stats.get(family + '/structured_admm', {})
        reference = cycle_stats.get(family + '/' + baseline, {})
        comparison = next((c for c in comparisons if c['family'] == family and c['baseline'] == baseline), None)
        coverage = candidate.get('count') == expected_cycles and reference.get('count') == expected_cycles
        accurate = coverage and candidate.get('failed') == 0 and reference.get('failed') == 0
        faster_mean = bool(comparison and comparison['speedup_ratio_of_means'] is not None and comparison['speedup_ratio_of_means'] > 1.)
        faster_p95 = bool(comparison and comparison['speedup_ratio_of_p95'] is not None and comparison['speedup_ratio_of_p95'] > 1.)
        gate[family] = dict(baseline=baseline, expected_cycles=expected_cycles,
            complete_coverage=coverage, all_cycles_accurate=accurate,
            mean_total_faster=faster_mean, p95_total_faster=faster_p95,
            passed=bool(requested_split == 'heldout' and accurate and faster_mean and faster_p95))
    failures = [r for r in records if r.get('applicable') and not r.get('accepted')]
    return dict(args=result['args'], tolerances=result['tolerances'],
        n_unique_qps=len({r['path'] for r in records}), repeats=result['args']['repeat'],
        qp_summary=qp_stats, cycle_summary=cycle_stats, paired_comparisons=comparisons,
        closed_loop_gate=dict(passed=all(g['passed'] for g in gate.values()), by_family=gate,
            rule='Every heldout QP/cycle must pass common accuracy, full paired coverage is required, '
                 'and complete-cycle mean plus P95 must improve against the optimized existing baseline. '
                 'Individual paired win counts are reported separately.'),
        failures=failures, failure_count=len(failures),
        provenance=dict(cpu_affinity=result['cpu_affinity'], versions=result['versions'],
                        source_sha256=result['source_sha256'], corpus_manifest_sha256=result['corpus_manifest_sha256']),
        timing=result['timing'], warm_start=result['warm_start'])


def markdown(summary, result_path):
    ok = summary['closed_loop_gate']['passed']
    out = [
        '# 固定 QP 结构化 ADMM 对照结果', '',
        ('**闭环接入门槛通过。** 可以开展独立的 MPC＋WBC 闭环验证；本报告本身不证明闭环性能。'
         if ok else '**闭环接入门槛未通过。** 当前证据不足以替换已优化的在线求解路径；本轮保持 MPC＋WBC 闭环不变。'), '',
        f"输入：`{result_path}`。{summary['n_unique_qps']} 个独立 {summary['args']['split']} QP，每个重复 {summary['repeats']} 次。",
        '语料来自 normal、double、push 的历史固定状态回放，使用同一 H、g、G、l、u；每个原问题均校验 SHA256。'
        'N=10、dt=0.04 s、0.4 s 时域保持不变；完整加权问题180变量/1340约束行，层级为基座→末端→平滑。', '',
        '## 完整周期耗时与失败', '',
        '以下以 benchmark 的 `cycles`/`cycle_summary` 为准。层级一周期包含全部三级，公共构造只加一次。'
        '“全部尝试”包含未达精度的求解耗时；“成功”仅作辅助统计，不代表失败可以忽略。', '',
        '|问题|方法|成功/尝试周期|失败|全部尝试均值/P95 ms|成功均值/P95 ms|',
        '|---|---|---:|---:|---:|---:|',
    ]
    for family in ('weighted', 'hierarchy'):
        for method in ORDER:
            row = summary['cycle_summary'].get(family+'/'+method)
            if not row or not row['count']:
                continue
            all_t, good_t = row['attempted_total_ms'], row['accepted_total_ms']
            out.append(f"|{family}|{LABELS[method]}|{row['accepted']}/{row['count']}|{row['failed']}|"
                       f"{fmt(all_t.get('mean'))}/{fmt(all_t.get('p95'))}|{fmt(good_t.get('mean'))}/{fmt(good_t.get('p95'))}|")
    out += ['', '`osqp_screened`只用于正定完整加权QP和层级末级，不组成完整三级方法；'
            '`existing_hierarchy`才是现有完整层级基线，它混合使用最小二乘、quadprog精修与末级OSQP，并非纯OSQP。', '',
            '## 配对比较', '',
            '速度比定义为“基线总耗时÷ADMM总耗时”，大于1表示ADMM较快。'
            '按同一 repeat＋snapshot_id 配对，共成功子集的速度比不外推到失败样本。', '',
            '|问题|对照基线|共成功/共有配对|排除失败配对|均值速度比|P95速度比|ADMM逐周期较快|',
            '|---|---|---:|---:|---:|---:|---:|']
    for row in summary['paired_comparisons']:
        out.append(f"|{row['family']}|{LABELS[row['baseline']]}|{row['common_success_pairs']}/{row['attempted_shared_pairs']}|"
                   f"{row['excluded_failure_pairs']}|{fmt(row['speedup_ratio_of_means'])}|"
                   f"{fmt(row['speedup_ratio_of_p95'])}|{row['admm_faster_pairs']}/{row['common_success_pairs']}|")
    out += ['', '## 原问题精度', '',
            '接受条件为原坐标归一化 primal/dual/normal 三项残差均≤2e−7，且物理原始约束违反≤2e−7。'
            '内部求解器状态或内部终止容差均不能替代此共同证书。下表最大值包括未通过样本；异常缺少证书时单独计数。', '',
            '|问题|方法|接受/尝试QP|未达精度/异常|原KKT最大值|物理违反最大值|相对目标差最大值|任务输出差最大值|',
            '|---|---|---:|---:|---:|---:|---:|---:|']
    for family in ('weighted', 'hierarchy'):
        for method in ORDER:
            row = summary['qp_summary'].get(family+'/'+method)
            if not row or not row['attempted']:
                continue
            out.append(f"|{family}|{LABELS[method]}|{row['accepted']}/{row['attempted']}|{row['failed']}|"
                       f"{expfmt(row['original_kkt_max_all'])}|{expfmt(row['physical_primal_max_all'])}|"
                       f"{expfmt(row['objective_gap_relative_max_all'])}|{expfmt(row['task_output_reference_difference_max_all'])}|")
    out += ['', '任务输出差为保存的加权任务矩阵 `C @ (x-reference_x)` 的最大绝对值，'
            '不同任务混有不同单位和权重，不是末端米制跟踪误差。半正定QP可以有不同的解向量，'
            '但真正处于任务零空间的解差应满足C差为零，不能用“解不唯一”解释这里的非零任务输出差。'
            '共同KKT容差在当前尺度/病态性下尚不足以保证任务输出等价，后续还需独立任务输出验收。'
            '本结果不声称闭环跟踪等价、关节抖振改善或非线性可行性保证。', '',
            '## 求解时间组成', '',
            '原生ADMM直接记录预处理、坐标变换、数值分解、迭代及原坐标检查。'
            '`factor_ms`已包含`coordinate_factor_ms`，两者不能重复相加。'
            '总分解次数为`factorization_count + coordinate_factorization_count`，统计已另存summary.json。'
            'OSQP的setup/update包含数值分解和其他准备，solve包含可能的rho更新、重分解及polish，'
            '不能将其字段直接解释成纯分解或纯迭代时间。完整solver_wall始终用于主比较。', '',
            '|ADMM问题|预处理均值 ms|坐标变换均值 ms|分解均值 ms|迭代均值 ms|内部检查均值 ms|外部证书均值 ms|',
            '|---|---:|---:|---:|---:|---:|---:|']
    for family in ('weighted', 'hierarchy'):
        row = summary['qp_summary'].get(family+'/structured_admm', {})
        fields = row.get('diagnostics', {})
        out.append('|'+family+'|'+'|'.join(fmt(fields.get(key, {}).get('mean')) for key in
            ('preprocess_ms', 'coordinate_transform_ms', 'factor_ms', 'iteration_ms', 'check_ms', 'certificate_ms'))+'|')
    out += ['', '这些分项为每QP的所有尝试均值，层级三项求和及公共构造仍应使用前面的完整周期结果。', '',
            '## 计时与解释边界', '',
            '- 公共构造包含当前状态动力学评估、接触等式消元、几何任务与预测矩阵构造；层级还包含任务划分、约化H/约束和QR任务锁。',
            '- 所有方法使用同一批预先测量的公共构造时间；它是公共项加求解器墙钟的配对估计，不是重新部署后的实际控制周期测量。',
            '- 求解器墙钟包含输入转换、预条件、必要分解/更新、全部迭代与完整原问题复核；磁盘I/O、参考求解和语料保存排除。',
            '- 原生ADMM未调用OSQP作为fallback，也没有活跃面直接解或改目标正则化。当前模型已接触消元并全时域凝聚；尚未实现支链/时间域并行。',
            '- 所有被测初始原始/对偶向量为零。OSQP工作区及按维数保留的rho可能复用；原生ADMM每QP从配置rho开始。无参考解或参考活跃集泄漏。',
            '- 正式v2仅保留同预条件OSQP、结构化ADMM和两种适用的现有工程基线。原始全约束OSQP的v1数据属于探索记录，不作为正式加速主张的依据。',
            '- ADMM预条件/罚参数只用calibration确定；首次测试后发现OSQP的精确不可行证书受继承近零行舍入影响，v2将eps_prim_inf/eps_dual_inf设为1e−30后复测，原QP矩阵和边界不变。该诊断修复意味着v2不是一次盲测，不能把v1状态概括为OSQP算法失败。',
            '- 这是固定QP求解层对照，不是DWMPC论文完整复现；未修改原建模、预测时域或控制器闭环。', '',
            '独立数值测试9项通过；另已只读审查输入不变性、原目标/乘子恢复、分项计时与公共构造归并。测试不替代下述真实QP验收。', '',
            '## 闭环接入判定', '',
            '|问题|优化工程基线|完整覆盖|全部精度通过|均值下降|P95下降|通过|',
            '|---|---|---|---|---|---|---|']
    for family, gate in summary['closed_loop_gate']['by_family'].items():
        out.append('|'+family+'|'+LABELS[gate['baseline']]+'|'+'|'.join('是' if gate[k] else '否' for k in
            ('complete_coverage', 'all_cycles_accurate', 'mean_total_faster', 'p95_total_faster', 'passed'))+'|')
    out += ['', '若只超过原始全约束OSQP而未超过现有预处理/筛选/层级混合路径，不能据此声称工程整体加速。', '',
            '## 失败记录', '', f"全部失败记录（{summary['failure_count']}条）保存在 `summary.json` 的 `failures`，包含策略、样本、重复、状态、残差和耗时；原始结果保留于输入文件。", '',
            f"CPU affinity: `{summary['provenance']['cpu_affinity']}`；版本：`{json.dumps(summary['provenance']['versions'])}`。", '']
    return '\n'.join(out)


def plot(summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    colors = {'structured_admm': '#0072B2', 'osqp_full': '#999999', 'osqp_preconditioned': '#E69F00',
              'osqp_screened': '#009E73', 'existing_hierarchy': '#CC79A7'}
    for column, family in enumerate(('weighted', 'hierarchy')):
        methods = [method for method in ORDER if summary['cycle_summary'].get(family+'/'+method, {}).get('count')]
        rows = [summary['cycle_summary'][family+'/'+method] for method in methods]
        labels = [LABELS[method].replace(' ', '\n', 1) for method in methods]
        means = [r['attempted_total_ms']['mean'] for r in rows]
        p95 = [r['attempted_total_ms']['p95'] for r in rows]
        ax = axes[0, column]
        ax.bar(range(len(rows)), means, color=[colors[m] for m in methods], alpha=.8, label='Mean, all attempts')
        ax.scatter(range(len(rows)), p95, marker='_', s=150, color='black', label='P95, all attempts')
        ax.set_yscale('log')
        ax.set_ylabel('Full-cycle time (ms, log scale)')
        ax.set_title(family + (' (three levels)' if family == 'hierarchy' else ''))
        ax.set_xticks(range(len(rows)), labels, fontsize=8)
        ax.grid(axis='y', alpha=.2)
        ax.legend(fontsize=8)
        ax = axes[1, column]
        failures = [100*r['failed']/r['count'] for r in rows]
        ax.bar(range(len(rows)), failures, color=[colors[m] for m in methods], alpha=.8)
        for i, row in enumerate(rows):
            ax.text(i, failures[i]+1, f"{row['failed']}/{row['count']}", ha='center', fontsize=9)
        ax.set_ylim(0, max(15, max(failures, default=0)+12))
        ax.set_ylabel('Failed full cycles (%)')
        ax.set_xticks(range(len(rows)), labels, fontsize=8)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Held-out fixed-QP replay: assembly + factorization + iterations + certification')
    fig.savefig(out/'comparison.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='results/structured_admm/heldout_v2/results.json')
    parser.add_argument('--manifest', default='results/structured_admm/qp_corpus/manifest.json')
    parser.add_argument('--output', default='results/structured_admm')
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    input_path, manifest_path, out = ROOT/args.input, ROOT/args.manifest, ROOT/args.output
    result = json.loads(input_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == result['corpus_manifest_sha256'], 'Corpus changed after benchmark'
    summary = make_summary(result, manifest)
    summary['report_source_sha256'] = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    (out/'RESULTS.md').write_text(markdown(summary, args.input))
    if not args.no_plot:
        plot(summary, out)
    print(json.dumps(dict(qps=summary['n_unique_qps'], failures=summary['failure_count'],
                         closed_loop_gate=summary['closed_loop_gate'], output=str(out)), ensure_ascii=False))


if __name__ == '__main__':
    main()
