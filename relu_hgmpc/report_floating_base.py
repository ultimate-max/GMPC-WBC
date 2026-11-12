"""Base motion and stability metrics, without conflating motion with tracking."""
import csv
import hashlib
import importlib.metadata
import json
import pathlib
import platform
import shutil
import subprocess
import sys

import numpy as np
from scipy.signal import welch
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/floating_base'
METHODS = ['wbc_threshold', 'wbc_hysteresis', 'hgmpc']
LABELS = {'wbc_threshold': 'Allocation + WBC (threshold)',
          'wbc_hysteresis': 'Allocation + WBC (hysteresis)', 'hgmpc': 'Hierarchical GMPC'}
CN = {'wbc_threshold': '运动分配＋WBC（原阈值）',
      'wbc_hysteresis': '运动分配＋WBC（滞环）', 'hgmpc': '优先级GMPC'}
COLORS = {'wbc_threshold': '#929292', 'wbc_hysteresis': '#ca752b', 'hgmpc': '#216e9b'}


def rms(x):
    x = np.asarray(x)
    return float(np.sqrt(np.mean(np.sum(x*x, axis=-1) if x.ndim == 2 else x*x)))


def band_rms(x):
    # Exclude first 0.5 s settling and report vector (sum-of-axes) RMS.
    y = x[500:]
    if len(y) < 2048:
        return None
    f, psd = welch(y, fs=1000, nperseg=2048, noverlap=1024, axis=0)
    ix = (f >= 10) & (f <= 100)
    return float(np.sqrt(np.trapezoid(psd[ix], f[ix], axis=0).sum()))


def run_metrics(name):
    s = json.loads((OUT / f'{name}.json').read_text())
    z = np.load(OUT / f'{name}_trace.npz')
    b, tr = z['base'], z['trace']
    R = Rotation.from_quat(b[:, 4:8])
    R0 = Rotation.from_quat(z['home'][3:7])
    dp = b[:, 1:4] - z['home'][:3]
    dr = (R0.inv() * R).as_rotvec()
    rpy = R.as_euler('xyz')
    rpy0 = R0.as_euler('xyz')
    rpy_delta = np.unwrap(rpy - rpy0, axis=0)
    lin, ang = b[:, 14:17], b[:, 17:20]
    contacts = tr[:, 1:5] > 0
    slip = tr[:, 21:33].reshape(-1, 4, 3).copy()
    slip[~contacts] = 0
    row = np.genfromtxt(OUT / f'{name}.csv', delimiter=',', names=True)
    d = dict(name=name, completed=s['failed'] is None and s['simulated_cycles'] == 600,
        failed=s['failed'], simulated_seconds=len(b) / 1000,
        base_position_rms_mm=1000*rms(dp), base_position_peak_mm=1000*float(np.linalg.norm(dp, axis=1).max()),
        base_height_rms_mm=1000*rms(dp[:, 2]), base_height_peak_mm=1000*float(np.abs(dp[:, 2]).max()),
        base_orientation_rms_deg=rms(dr)*180/np.pi,
        base_orientation_peak_deg=float(np.linalg.norm(dr, axis=1).max())*180/np.pi,
        base_axis_position_rms_mm=(1000*np.sqrt(np.mean(dp*dp, axis=0))).tolist(),
        base_axis_rpy_rms_deg=(180/np.pi*np.sqrt(np.mean(rpy_delta*rpy_delta, axis=0))).tolist(),
        base_linear_speed_rms_mps=rms(lin), base_angular_speed_rms_degps=rms(ang)*180/np.pi,
        base_linear_increment_rms_mmps=1000*rms(np.diff(lin, axis=0)),
        base_angular_increment_rms_degps=rms(np.diff(ang, axis=0))*180/np.pi,
        base_linear_band_10_100_hz_mmps=None if band_rms(lin) is None else 1000*band_rms(lin),
        base_angular_band_10_100_hz_degps=None if band_rms(ang) is None else band_rms(ang)*180/np.pi,
        ee_position_rmse_mm=s['position_rmse_mm'], ee_orientation_rmse_deg=s['rotation_rmse_deg'],
        feedback_mean_amortized_ms=s['amortized_feedback_ms'],
        feedback_p95_ms=s['feedback_ms']['p95'], feedback_p99_ms=s['feedback_ms']['p99'],
        feedback_max_ms=s['feedback_ms']['max'], torque_clips=s['torque_clips'],
        min_contacts=int(contacts.sum(axis=1).min()),
        missing_contact_ms=int((contacts.sum(axis=1)<4).sum()),
        per_foot_missing_contact_ms=(~contacts).sum(axis=0).tolist(),
        slip_path_max_mm=1000*.001*float(np.linalg.norm(slip[:, :, :2], axis=2).sum(axis=0).max()),
        reference_sha256=s['reference_sha256'], source_sha256=s['source_sha256'],
        joint_position_violation_max=s['joint_position_violation']['max'],
        joint_velocity_violation_max=s['joint_velocity_violation']['max'])
    if s['args']['controller'] == 'hgmpc':
        d.update(original_qp_relative_residual_max=max(s[k]['max'] for k in
            ['qp_primal_relative', 'qp_dual_relative', 'qp_normal_relative']),
            task_lock_relative_max=s['task_lock_relative']['max'],
            contact_acceleration_residual_max=s['contact_acceleration_residual']['max'])
        assert d['original_qp_relative_residual_max'] <= 2e-7
        assert d['task_lock_relative_max'] <= 2e-7
    else:
        d.update(base_activation_switches=int(np.count_nonzero(np.diff(row['base_active']))),
            base_active_fraction=float(row['base_active'].mean()),
            own_internal_base_position_rmse_mm=rms(row['internal_base_position_error_mm']),
            own_internal_base_orientation_rmse_deg=rms(row['internal_base_rotation_error_deg']),
            internal_goal_displacement_rms_mm=1000*rms(z['internal_base_goal'][:, :3]-z['home'][:3]),
            qp_primal_max=s['primal']['max'], hierarchy_residual_max=s['hierarchy']['max'],
            contact_acceleration_residual_max=s['wbc_contact_acc_residual']['max'])
    for group, sl in [('legs', slice(0, 12)), ('arm', slice(12, 18))]:
        d[f'{group}_velocity_increment_rms_radps'] = float(np.sqrt(np.mean(np.diff(tr[:, 111:129][:, sl], axis=0)**2)))
        d[f'{group}_torque_increment_rms_nm'] = float(np.sqrt(np.mean(np.diff(tr[:, 57:75][:, sl], axis=0)**2)))
    return s, z, row, d, dp, rpy_delta


groups, individual, repeat_checks, traces = {}, {}, {}, {}
for scale in [1, 2]:
    hashes = []
    initial_states = []
    for method in METHODS:
        names = [f'{method}_scale{scale}_r{r}' for r in [1, 2]]
        loaded = [run_metrics(n) for n in names]
        for n, data in zip(names, loaded):
            individual[n] = data[3]
            hashes.append(data[3]['reference_sha256'])
            initial_states.append(data[1]['home'])
        first, second = loaded
        checks = {key: float(np.max(np.abs(first[1][key]-second[1][key])))
            if first[1][key].shape == second[1][key].shape else None
            for key in ['q', 'v', 'base', 'tau', 'trace']}
        repeat_checks[f'{method}_scale{scale}'] = checks
        d = first[3].copy()
        d['names'] = names
        d['both_completed'] = all(x[3]['completed'] for x in loaded)
        d['feedback_mean_amortized_ms'] = float(np.mean([x[3]['feedback_mean_amortized_ms'] for x in loaded]))
        d['feedback_repeat_means_ms'] = [x[3]['feedback_mean_amortized_ms'] for x in loaded]
        pooled = np.concatenate([x[2]['feedback_ms'] for x in loaded])
        d['feedback_p95_ms'] = float(np.quantile(pooled, .95))
        d['feedback_p99_ms'] = float(np.quantile(pooled, .99))
        d['feedback_max_ms'] = float(pooled.max())
        groups[f'{method}_scale{scale}'] = d
        traces[f'{method}_scale{scale}'] = first
    assert len(set(hashes)) == 1, 'The methods did not receive identical reference samples'
    assert all(np.array_equal(initial_states[0], q) for q in initial_states[1:])

notes = [
    'Existing B2+Z1 model, MuJoCo truth, fixed four-foot standing, identical 6D end-effector reference and initial state. No new geometric WBC implemented.',
    'WBC is the 6D engineering extension, not the original 3D experiment. Original threshold 0.10 and improved hysteresis 0.10/0.12 are reported separately.',
    'No change to controller task hierarchy, gains, constraints, solver tolerances, or joint servo gains. Hierarchies and low-level reference generation differ between architectures.',
    'GMPC: hard physical constraints; first-level end-effector pose/error-rate; second-level base home, posture and smoothing. CPU OSQP, N=10, dt=30 ms, both-level relative original-coordinate gate 2e-7.',
    'All base offsets are relative to initial physical pose, not common base tracking errors. WBC deliberately changes its internal base target; GMPC has a soft home-pose objective. Smaller displacement alone is not better coordination.',
    'True base state/contact/actuator samples at 1 kHz; end-effector error and feedback time at 50 Hz. Base angular quantities are SO(3) distance and world angular velocity; RPY used only for small-angle component plots.',
    'Band metrics: vector RMS from Welch PSD integrated over 10-100 Hz; first 0.5 s excluded, 2048 sample windows, 1024 overlap. This distinguishes high-frequency motion from intentional slow base movement, but does not identify all causes.',
    'Two serialized deterministic repeats per condition; repeats check numerical reproducibility and timing variability, not statistical robustness. Physical metrics use the first repeat; repeat differences are exposed.',
    'Timing includes state read, controller/model/matrices/solvers/residual checks, initialization amortized; logging and simulator advance excluded. Computation delay not injected: no real-time or hardware-stability claim.',
    'Contact slip uses material velocity at measured contact point, not spherical-foot center drift. Missing contact is a measured outcome, not a forced early termination.',
    'Different controller architectures and optimization problems: runtime ratio is not a same-QP solver acceleration factor; differences cannot be assigned to Lie-group representation alone.',
    'This comparison concerns base holding/coordination during the existing end-effector trajectory. No shared moving-base reference or external push-rejection test is included.'
]
report = dict(groups=groups, runs=individual, repeat_max_differences=repeat_checks,
    verification=json.loads((OUT/'verification.json').read_text()), notes=notes,
    total_feedback_cycles=sum(x['simulated_seconds']*50 for x in individual.values()),
    total_physics_samples=sum(x['simulated_seconds']*1000 for x in individual.values()),
    platform=platform.platform(), cpu=subprocess.run(['lscpu'], capture_output=True, text=True).stdout,
    versions={n:importlib.metadata.version(n) for n in ['numpy', 'scipy', 'osqp', 'mujoco', 'quadprog']},
    plant_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                  for name in ['models/b2_z1.urdf', 'paper_repro/simulation.hpp', 'src/robot.cpp']})
(OUT/'comparison.json').write_text(json.dumps(report, indent=2))

for scale in [1, 2]:
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.7), layout='constrained')
    for method in METHODS:
        _, z, _, _, dp, angles = traces[f'{method}_scale{scale}']
        t = z['base'][:, 0]-z['base'][0, 0]
        for j, label in enumerate(['x', 'y', 'z']):
            axes[0, j].plot(t[::5], 1000*dp[::5, j], label=LABELS[method], color=COLORS[method], lw=1.05)
            axes[0, j].set_title(f'Base {label} displacement'); axes[0, j].set_ylabel('From initial pose (mm)')
        for j, label in enumerate(['roll', 'pitch', 'yaw']):
            axes[1, j].plot(t[::5], angles[::5, j]*180/np.pi, label=LABELS[method], color=COLORS[method], lw=1.05)
            axes[1, j].set_title(f'Base {label} change'); axes[1, j].set_ylabel('From initial orientation (deg)')
    for ax in axes.flat:
        ax.grid(alpha=.2); ax.set_xlabel('Simulation time (s)')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f'B2+Z1 floating base: end-effector trajectory scale {scale}\nFirst repeat; 1 kHz truth plotted every 5 samples, no smoothing; offsets are not a shared tracking error', fontsize=11)
    fig.savefig(OUT/f'base_components_scale{scale}.png', dpi=180)
    fig.savefig(OUT/f'base_components_scale{scale}.pdf')
    plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), layout='constrained')
specs = [('base_position_rms_mm', 'Base displacement RMS', 'mm'),
         ('base_orientation_rms_deg', 'Base orientation displacement RMS', 'deg'),
         ('base_angular_band_10_100_hz_degps', 'Base angular velocity: 10-100 Hz RMS', 'deg/s'),
         ('ee_position_rmse_mm', 'End-effector position RMSE', 'mm'),
         ('missing_contact_ms', 'Time with fewer than four contacts', 'ms'),
         ('feedback_mean_amortized_ms', 'Total feedback time (startup amortized)', 'ms')]
for ax, (key, title, unit) in zip(axes.flat, specs):
    for i, method in enumerate(METHODS):
        values = [groups[f'{method}_scale{scale}'][key] for scale in [1, 2]]
        x = np.array([0, 1]) + (i-1)*.25
        bars = ax.bar(x, values, .24, color=COLORS[method], label=LABELS[method])
        for bar, value in zip(bars, values):
            ax.annotate(f'{value:.3g}', (bar.get_x()+bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
    ax.set_xticks([0, 1], ['Normal', 'Double amplitude'])
    ax.set_xlabel('End-effector reference amplitude'); ax.set_ylabel(unit); ax.set_title(title, fontsize=10)
    ax.grid(axis='y', alpha=.2); ax.margins(y=.25)
axes[0, 0].legend(fontsize=7)
fig.suptitle('Floating-base comparison: existing allocation + WBC and hierarchical GMPC\n12 s standing trials; two serialized repeats; no computation-delay injection', fontsize=11)
fig.savefig(OUT/'floating_base_summary.png', dpi=180)
fig.savefig(OUT/'floating_base_summary.pdf')
plt.close(fig)

# Explicitly expose actual angular-velocity spectra, rather than assigning all
# base pose displacement to vibration. The PSD uses exactly the metric window.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout='constrained')
for ax, scale in zip(axes, [1, 2]):
    for method in METHODS:
        z = traces[f'{method}_scale{scale}'][1]
        f, p = welch(z['base'][500:, 17:20]*180/np.pi, fs=1000,
                     nperseg=2048, noverlap=1024, axis=0)
        ax.semilogy(f, p.sum(axis=1), color=COLORS[method], label=LABELS[method])
    ax.axvspan(10, 100, color='black', alpha=.05)
    ax.set_xlim(0, 100); ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Sum of angular-velocity PSDs ((deg/s)^2/Hz)')
    ax.set_title(f'End-effector reference scale {scale}'); ax.grid(alpha=.2)
axes[0].legend(fontsize=8)
fig.suptitle('Measured floating-base angular velocity spectrum\nMuJoCo truth: 0.5-12 s, 1 kHz; Welch 2048 points / 1024 overlap', fontsize=11)
fig.savefig(OUT/'base_angular_spectrum.png', dpi=180)
fig.savefig(OUT/'base_angular_spectrum.pdf')
plt.close(fig)

fields = ['method', 'scale', 'both_completed']+[x[0] for x in specs]+[
    'base_height_rms_mm', 'ee_orientation_rmse_deg', 'base_linear_band_10_100_hz_mmps',
    'min_contacts', 'torque_clips', 'feedback_p95_ms', 'feedback_p99_ms']
with (OUT/'metrics.csv').open('w') as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for scale in [1, 2]:
        for method in METHODS:
            d = groups[f'{method}_scale{scale}']
            w.writerow(dict(method=method, scale=scale, **{k:d[k] for k in fields[2:]}))

lines = ['# 浮动基表现：优先级运动分配＋WBC 对比优先级GMPC', '',
    '2026-11-11。使用现有 B2＋Z1 与四足站立末端6D轨迹，两种幅度各12秒、各两次串行重复。未新增几何WBC，未修改两套控制器的目标、层级或参数。', '',
    '## 如何解读', '',
    '本轮比较的是末端任务中的浮动基保持与协调表现，不是同一条底盘运动轨迹的跟踪精度，也没有外力推扰试验。WBC会主动改变底盘参考，GMPC将回到初始底盘位姿作为次层目标，因此“位移更小”不能单独作为控制更好的证据。需要联合看末端任务、足端接触和高频运动。', '',
    '## 主要发现', '',
    'GMPC在既有站立任务中更接近初始底盘位姿，实际角速度的10–100Hz分量也更小；但滞环WBC同样全程保持四足接触，末端姿态误差更小，计算时间约2ms。不能据此判定GMPC全面更优。', '']
for scale in [1, 2]:
    w = groups[f'wbc_hysteresis_scale{scale}']; g = groups[f'hgmpc_scale{scale}']
    reduction = 100*(1-g['base_angular_band_10_100_hz_degps']/w['base_angular_band_10_100_hz_degps'])
    lines += [f'- 幅度{scale}：相对滞环WBC，GMPC底盘角速度10–100Hz分量RMS降低 {reduction:.2f}%。WBC对自身底盘目标的位置/姿态RMSE为 {w["own_internal_base_position_rmse_mm"]:.3f} mm / {w["own_internal_base_orientation_rmse_deg"]:.3f}°，说明较大的初始位姿偏移主要属于主动运动分配。']
lines += ['', '原阈值WBC在普通/双幅任务中分别切换41/39次，出现130/530ms不足四足接触；滞环版本均仅切换1次且没有失去四足接触。这项改进来自切换策略，不能归于几何表示。', '',
    '仍不能从本轮得出“李群使浮动基跟踪精度提高”的因果结论：任务目标、优先级、速度层分配与预测协调、约束实施以及低层参考生成同时不同。这里报告的是现有完整控制方案的观测表现。', '',
    '## 实测结果', '']
for scale in [1, 2]:
    lines += [f'### {"普通" if scale == 1 else "双幅"}末端参考', '']
    for method in METHODS:
        d = groups[f'{method}_scale{scale}']
        lines += [f'**{CN[method]}**：完成状态 {d["both_completed"]}。底盘位置偏移RMS {d["base_position_rms_mm"]:.6f} mm，姿态偏移RMS {d["base_orientation_rms_deg"]:.6f}°，高度偏移RMS {d["base_height_rms_mm"]:.6f} mm。',
            f'底盘角速度10–100 Hz分量RMS {d["base_angular_band_10_100_hz_degps"]:.6f}°/s；末端位置RMSE {d["ee_position_rmse_mm"]:.6f} mm，姿态RMSE {d["ee_orientation_rmse_deg"]:.6f}°。',
            f'最少接触足数 {d["min_contacts"]}；不足四足接触累计 {d["missing_contact_ms"]} ms；力矩裁剪 {d["torque_clips"]} 次。完整反馈平均 {d["feedback_mean_amortized_ms"]:.3f} ms，P95 {d["feedback_p95_ms"]:.3f} ms，P99 {d["feedback_p99_ms"]:.3f} ms。']
        if method != 'hgmpc':
            lines += [f'底盘启停切换 {d["base_activation_switches"]} 次；自身底盘目标位移RMS {d["internal_goal_displacement_rms_mm"]:.6f} mm。对自身内部目标的位置误差RMS {d["own_internal_base_position_rmse_mm"]:.6f} mm、姿态误差RMS {d["own_internal_base_orientation_rmse_deg"]:.6f}°，这些内部目标与GMPC不同，不用于跨方法排名。']
        lines += ['']
    lines += [f'![浮动基六分量](base_components_scale{scale}.png)', '']
lines += ['![实际角速度频谱](base_angular_spectrum.png)', '', '## 可复核范围与限制', '',
    '- 控制结构不同：原论文速度层运动分配＋层级WBC；GMPC第一层末端位姿与误差变化率，第二层底盘构型与输入平滑。不能把整体差异归因于几何误差或MPC中的某一个因素。',
    '- WBC采用既有6D工程扩展，并明确保留单阈值与滞环两个版本。原论文的3D实验不直接参与6D姿态排名。WBC扩展本身已使用SO(3)对数姿态误差。',
    '- 每个物理步记录基座位置、四元数、局部/世界坐标速度，时间戳与接触/执行器记录一致。新增记录与关闭记录的物理轨迹逐位相同；短GMPC回放与此前版本的q/v/力矩/加速度逐位相同。',
    '- 角速度频带指标来自1kHz真值，对世界坐标角速度三个轴的功率谱求和后积分开方；去掉最初0.5秒，10–100Hz，Welch窗2048点、重叠1024点。图仅抽取每5点绘制，无滤波平滑。',
    '- 时延没有注入物理仿真。控制指令每20ms仿真时间更新，计算耗时大于20ms时不能据此声称50Hz实时控制；总时间包括状态、模型/矩阵、求解和残差检查及初始化摊销，不含数据写盘和MuJoCo推进。',
    '- 两次重复用于确定性复核及计时波动观察，不是独立随机鲁棒性试验。完整来源哈希、重复差值、求解残差见comparison.json。',
    '- 底盘高度/位置单位mm，姿态角度单位deg，速度频带单位mm/s或deg/s。足端打滑使用接触点材料速度，未用球心位移替代。', '',
    '![汇总](floating_base_summary.png)', '', '## 复现', '',
    '在 `/home/ganhe/project/b2z1_mpc` 下运行：', '', '```bash',
    'source relu_hgmpc/env.sh', 'python relu_hgmpc/test_floating_base.py',
    'bash relu_hgmpc/benchmark_floating_base.sh', 'python relu_hgmpc/report_floating_base.py', '```', '',
    '来源：Prioritized Multi-task Motion Coordination of Physically Constrained Quadruped Manipulators，DOI: 10.34133/cbsystems.0203。移植细节及与原文的差异见 relu_hgmpc/PRIORITIZED.md。']
(OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
if len(sys.argv) > 1:
    destination = pathlib.Path(sys.argv[1]); destination.mkdir(parents=True, exist_ok=True)
    for p in OUT.iterdir():
        if p.is_file():
            shutil.copy2(p, destination/p.name)
    for name in ['run_floating_base.py', 'report_floating_base.py', 'test_floating_base.py',
                 'benchmark_floating_base.sh', 'PRIORITIZED.md', 'HIERARCHICAL_GMPC.md', 'native.cpp']:
        shutil.copy2(ROOT/'relu_hgmpc'/name, destination/name)
print(json.dumps({k:{f:d[f] for f in ['both_completed', 'base_position_rms_mm',
    'base_orientation_rms_deg', 'base_angular_band_10_100_hz_degps', 'ee_position_rmse_mm',
    'ee_orientation_rmse_deg', 'missing_contact_ms', 'feedback_mean_amortized_ms']}
    for k,d in groups.items()}, indent=2))
print('Repeat checks:', json.dumps(repeat_checks))
