"""Aggregate the sequential matched experiment, preserving raw run records."""
import csv,json,pathlib,platform
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'results/paper_gmpc'
VARIANTS=['frozen','frozen_cached','geometry_cached','paper','paper_cached']
LABELS={'frozen':'Frozen original','frozen_cached':'Frozen + cache',
        'geometry_cached':'Geometry only + cache','paper':'Paper first-order',
        'paper_cached':'Paper first-order + cache'}
COLORS={'frozen':'#5b6770','frozen_cached':'#7d8c97','geometry_cached':'#c17d23',
        'paper':'#92bdd6','paper_cached':'#0077b6'}
TIMINGS=['feedback_ms','controller_ms','model_ms','setup_ms','solve_ms','derivatives_ms','state_read_ms']
RESIDUALS=['qp_primal_relative','qp_dual_relative','qp_normal_relative','affine_dynamics_max',
           'first_acceleration_defect','local_primal','interface_scaled']
data={};runs={};groups={};hashes=[]
def name(v,r,scale=1):return f'aligned_{v}'+(f'_scale{scale}' if scale!=1 else '')+f'_r{r}'
def stats(a):return dict(mean=float(np.mean(a)),p95=float(np.quantile(a,.95)),
                        p99=float(np.quantile(a,.99)),max=float(np.max(a)))
for scale,variants in [(1,VARIANTS),(2,['frozen_cached','paper_cached'])]:
    for v in variants:
        names=[name(v,r,scale) for r in [1,2]]
        summaries=[json.loads((OUT/f'{n}.json').read_text()) for n in names]
        series=[]
        for n,s in zip(names,summaries):
            with (OUT/f'{n}.csv').open() as f:rows=list(csv.DictReader(f))
            a={k:np.array([float(row[k]) for row in rows]) for k in rows[0]}
            assert s['failed'] is None and s['cycles']==600,(n,s['failed'])
            assert np.all(a['qp_converged']==1),n
            assert s['min_contacts']==4 and s['torque_clips']==0,n
            assert s['missing_contact_ms']==0,n
            assert s['reference_cache_misses']==0,n
            assert max(np.max(a[k]) for k in RESIDUALS[:3])<=2e-7*(1+1e-8),n
            hashes.append(s['source_sha256']);data[n]=a;runs[n]=s;series.append(a)
        pooled={k:np.concatenate([a[k] for a in series]) for k in series[0]}
        count=len(pooled['time']);startup=sum(s['precompute_ms'] for s in summaries)
        result=dict(names=names,scale=scale,cycles=count,
            position_rmse_mm=float(np.sqrt(np.mean(pooled['position_error_mm']**2))),
            rotation_rmse_deg=float(np.sqrt(np.mean(pooled['rotation_error_deg']**2))),
            amortized_feedback_ms=(float(np.sum(pooled['feedback_ms']))+startup)/count,
            startup_ms_per_run=[s['precompute_ms'] for s in summaries],
            feedback_means_per_run=[s['feedback_ms']['mean'] for s in summaries],
            reference_cache_bytes=max(s['reference_cache_bytes'] for s in summaries),
            max_foot_displacement_mm=float(np.max(pooled['foot_center_mm'])),
            max_slip_path_mm=max(s['max_slip_path_mm'] for s in summaries),
            min_base_z=float(np.min(pooled['base_z'])),
            qp_iterations=stats(pooled['qp_iterations']),
            timing={k:stats(pooled[k]) for k in TIMINGS},
            residual_max={k:float(np.max(pooled[k])) for k in RESIDUALS})
        result['timing']['other_ms']=stats(pooled['feedback_ms']-pooled['model_ms']-pooled['setup_ms']-pooled['solve_ms'])
        groups[f'{v}_scale{scale}']=result
assert all(h==hashes[0] for h in hashes),'Source changed during matched benchmark'

# Compare complete simulator trajectories, not only scalar error summaries.
equivalence={}
def compare(n1,n2):
    with np.load(OUT/f'{n1}_trace.npz') as a,np.load(OUT/f'{n2}_trace.npz') as b:
        return {k:float(np.max(np.abs(a[k]-b[k]))) for k in ['q','v','trace']}
for v in VARIANTS:
    equivalence[f'{v}_repeat']=compare(name(v,1),name(v,2))
for v in ['frozen','paper']:
    equivalence[f'{v}_cache']=compare(name(v,1),name(v+'_cached',1))
for v in ['frozen_cached','paper_cached']:
    equivalence[f'{v}_scale2_repeat']=compare(name(v,1,2),name(v,2,2))
assert all(max(x.values())<1e-8 for x in equivalence.values()),equivalence
verify=json.loads((OUT/'verification.json').read_text())
def delta(new,old,key):return 100*(new[key]/old[key]-1)
base=groups['frozen_scale1'];paper=groups['paper_scale1'];cache=groups['paper_cached_scale1']
stress_old=groups['frozen_cached_scale2'];stress_new=groups['paper_cached_scale2']
changes={key:delta(cache,base,key) for key in ['position_rmse_mm','rotation_rmse_deg','amortized_feedback_ms']}
cache_gain=-delta(cache,paper,'amortized_feedback_ms')
stress_changes={key:delta(stress_new,stress_old,key) for key in changes}
summary=dict(groups=groups,cache_and_repeat_max_differences=equivalence,
    changes_paper_cached_vs_frozen_percent=changes,cache_speed_reduction_percent=cache_gain,
    changes_scale2_percent=stress_changes,verification=verify,source_sha256=hashes[0],
    environment=dict(platform=platform.platform(),python=platform.python_version()),
    timing_scope=runs[name('paper_cached',1)]['timing_scope'],
    all_runs_completed=True,all_cycles_qp_converged=True,all_contacts_retained=True,
    limitation='Two repeated deterministic simulations measure timing variation, not statistical controller robustness. No computation delay injection; no real-time claim.')
(OUT/'comparison.json').write_text(json.dumps(summary,indent=2))

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'figure.facecolor':'white','axes.grid':True,'grid.alpha':.2})
fig,axes=plt.subplots(2,2,figsize=(13,7),sharex=True)
for col,scale in enumerate([1,2]):
    for v in ['frozen_cached','paper_cached']:
        a=data[name(v,1,scale)]
        for row,key in enumerate(['position_error_mm','rotation_error_deg']):
            axes[row,col].plot(a['time'],a[key],label=LABELS[v],color=COLORS[v],lw=1.4)
    axes[0,col].set_title(f'Reference amplitude {scale}x')
    axes[1,col].set_xlabel('Simulation time [s]')
    axes[0,col].legend(fontsize=9)
axes[0,0].set_ylabel('End-effector position error [mm]')
axes[1,0].set_ylabel('End-effector orientation error [deg]')
fig.suptitle('Frozen baseline vs paper first-order GMPC | N=10, dt=30 ms, same OSQP tolerance')
fig.text(.5,.008,'Ground-truth state, fixed four-foot contact; computation delay is NOT injected into the simulator.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.025,1,.96]);fig.savefig(OUT/'tracking.png',dpi=180);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(13,5),gridspec_kw={'width_ratios':[1.7,1]})
x=np.arange(len(VARIANTS));bottom=np.zeros(len(x))
for key,label,color in [('model_ms','Model + matrices','#0077b6'),('setup_ms','QP setup','#55a8a0'),
                        ('solve_ms','QP solve','#e4a64b'),('other_ms','Read/recovery/checks','#8c8795')]:
    values=np.array([groups[v+'_scale1']['timing'][key]['mean'] for v in VARIANTS])
    axes[0].bar(x,values,bottom=bottom,label=label,color=color);bottom+=values
for i,v in enumerate(VARIANTS):
    g=groups[v+'_scale1'];axes[0].text(i,bottom[i]+1.3,f"{g['amortized_feedback_ms']:.2f}",ha='center',fontsize=9)
axes[0].set_xticks(x,['Frozen','Frozen\n+ cache','Geometry\n+ cache','Paper','Paper\n+ cache'])
axes[0].set_ylabel('Mean feedback time [ms]');axes[0].set_ylim(0,max(bottom)*1.24)
axes[0].legend(fontsize=9,loc='upper left',ncol=2)
axes[0].set_title('Timing includes matrix construction and QP setup')
axes[1].bar([0,1],[base['position_rmse_mm'],cache['position_rmse_mm']],color=[COLORS['frozen'],COLORS['paper_cached']])
axes[1].set_xticks([0,1],['Frozen','Paper + cache']);axes[1].set_ylabel('Position RMSE [mm]')
axes[1].set_title('Tracking accuracy, normal amplitude')
for i,g in enumerate([base,cache]):axes[1].text(i,g['position_rmse_mm']*1.02,f"{g['position_rmse_mm']:.4f}",ha='center')
axes[1].set_ylim(0,max(base['position_rmse_mm'],cache['position_rmse_mm'])*1.2)
fig.text(.5,.01,'1200 cycles per variant (two sequential 12 s runs). Labels include amortized one-time cache preparation.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.035,1,1]);fig.savefig(OUT/'timing.png',dpi=180);plt.close(fig)

lines=['# 论文一阶几何误差动力学：B2+Z1 闭环对照','',
    '2026-11-10。正式结果只使用 `aligned_*.json/csv`；无该前缀的结果是离散公式对齐前的排查记录。','',
    '## 实现与实验边界','',
    '补上 −ad(Vd)phi、任务速度 HV+Fu+b、Jdot、Cv 和测量冗余速度；末端任务使用原文 (22)–(24) 的前向 Euler，并独立验证输出坐标变换。H/F/b 每周期更新、预测域内冻结，符合原文 Assumption 1。原冻结控制器未修改。',
    '四足接触、浮动底盘和接口力沿用当前工程；关节/接触辅助预测仍是原有半隐式积分。当前是集中式加权多任务扩展，不是固定基论文的严格优先级算法、完整多体 SQP 或稳定性证明复现。',
    '固定四足站立；MuJoCo 真值；Pinocchio；OSQP；N=10、步长30 ms、控制间隔20 ms；420个控制变量。所有版本保持相同权重、物理界限和共同原始 QP 相对残差门槛 2e-7。模型和离散预测映射是自变量，不声称 QP 数值矩阵相同。',
    '每版本 12 s × 2 次，顺序/逆序串行测时；双幅轨迹按 ABBA 顺序。重复仿真轨迹确定相同，两次重复仅帮助观察计时变化，不能作为统计鲁棒性证据。','',
    '## 主结果：普通幅度','']
for v in VARIANTS:
    g=groups[v+'_scale1'];tm=g['timing'];low,high=min(g['feedback_means_per_run']),max(g['feedback_means_per_run'])
    lines.append(f"- **{v}**：位置 RMSE {g['position_rmse_mm']:.6f} mm，姿态 RMSE {g['rotation_rmse_deg']:.6f}°；总反馈均值 {tm['feedback_ms']['mean']:.3f} ms、含启动摊销 {g['amortized_feedback_ms']:.3f} ms、P95 {tm['feedback_ms']['p95']:.3f} ms、P99 {tm['feedback_ms']['p99']:.3f} ms；两次均值范围 {low:.3f}–{high:.3f} ms。")
lines+=['',f"论文模型＋缓存相对原冻结版本：位置 RMSE 变化 {changes['position_rmse_mm']:+.2f}%，姿态 RMSE 变化 {changes['rotation_rmse_deg']:+.2f}%，摊销总耗时变化 {changes['amortized_feedback_ms']:+.2f}%。负号表示下降。",
    f"同一论文模型中，缓存使摊销总耗时下降 {cache_gain:.2f}%；这只是当前机器上的两次实测差异，不是普遍保证。缓存前后完整 q/v 和 1 kHz 仿真记录的最大差异见 comparison.json，均通过 1e-8 门槛。",'',
    '![跟踪误差](tracking.png)','', '## 双幅参考','']
for v in ['frozen_cached','paper_cached']:
    g=groups[v+'_scale2'];lines.append(f"- **{v}**：位置 RMSE {g['position_rmse_mm']:.6f} mm，姿态 RMSE {g['rotation_rmse_deg']:.6f}°；含启动摊销总耗时 {g['amortized_feedback_ms']:.3f} ms，P95 {g['timing']['feedback_ms']['p95']:.3f} ms。")
lines += [f"变化：位置 {stress_changes['position_rmse_mm']:+.2f}%，姿态 {stress_changes['rotation_rmse_deg']:+.2f}%，总耗时 {stress_changes['amortized_feedback_ms']:+.2f}%。",'',
    '## 计时与可复用矩阵','',
    '总反馈从读取状态开始，包含原生模型评估、参考评估、矩阵构造、QP 准备/求解和恢复/诊断。不包括仿真推进和绘图。启动缓存成本另计并摊销；CPU 上无 GPU 数据传输。实际求解器的数据转换/装配计入 setup。',
    '![计时分解](timing.png)','']
for v in ['frozen','paper','paper_cached']:
    g=groups[v+'_scale1'];t=g['timing']
    lines.append(f"- {v}：模型/构造 {t['model_ms']['mean']:.3f} ms，QP setup {t['setup_ms']['mean']:.3f} ms，QP solve {t['solve_ms']['mean']:.3f} ms，剩余读数/恢复/检查 {t['other_ms']['mean']:.3f} ms；导数计算 {t['derivatives_ms']['mean']:.4f} ms（包含在模型耗时内，不重复相加）；平均 QP 迭代 {g['qp_iterations']['mean']:.2f}。")
lines += ['',
    '- 已预计算：规定参考 Td/Vd、ad(Vd)、I−dt ad(Vd)；输入缩放、选择矩阵、节点输入选择和固定摩擦锥模板。输出矩阵按非零块使用。参考改变需重建参考缓存，接触/摩擦改变需重建结构缓存。',
    '- 同一预测域复用：H/F/b、M/C/J/Jdot、动态一致逆、零空间速度、离散 AV/BV/bV。它们每周期依赖实际状态，未假定可永久冻结。',
    '- 尚未跨周期复用：QP 数值 Hessian、约束矩阵和分解；这些随状态改变。当前仍每周期准备 QP，结构/因子复用和接触等式消元是下一步更可能减少主要开销的方向，尚未在本轮证明收益。',
    f"- 缓存数据数组上限 {cache['reference_cache_bytes']/1024:.1f} KiB（不含 Python 容器开销）；单次准备时间 {cache['startup_ms_per_run']} ms；本实验无缓存 miss。",'',
    '## 验证和解释','',
    f"全部 14 次闭环、{sum(g['cycles'] for g in groups.values())} 个控制周期完成，QP 均满足共同相对残差门槛；1 kHz 记录均保持四足接触，无力矩裁剪。",
    f"所有实验的相对 primal/dual/normal 残差最大值分别为 {max(g['residual_max']['qp_primal_relative'] for g in groups.values()):.3e} / {max(g['residual_max']['qp_dual_relative'] for g in groups.values()):.3e} / {max(g['residual_max']['qp_normal_relative'] for g in groups.values()):.3e}。",
    f"模型恒等式最大误差 {max(r['task_acceleration_identity'] for r in verify['identities']):.3e}；Jdot 独立差分最大误差 {max(r['jacobian_time_derivative_error'] for r in verify['identities']):.3e}；论文离散输出模型最大误差 {max(verify['paper_discrete_prediction_errors']):.3e}；一阶误差量减半时余项缩减比约 4。",
    '新模型的 affine_dynamics_max 检查 M a + C v + g 与优化输入。旧诊断 full_dynamics_max 在所有节点使用固定 h，因此对新模型不再表示其仿射等式残差；不能混淆这两个指标。原始 CSV 两者均保留。',
    '模型局部一阶正确并不保证闭环误差必然更小：该对照同时改变任务速度预测和误差离散传播；H/F/b、Jacobian、冗余速度仍沿预测域冻结；权重、时域和原有低层保持不变。geometry_cached 是几何传播消融，但仍包含其离散方式变化，不把误差差异全部归因于 ad 项。',
    '模型验证支持公式实现与局部近似阶数；不构成闭环稳定性或最优性证明。本轮没有新测 ReLU-QP，因为先用相同 OSQP 隔离模型与缓存的影响。',
    '**计算延迟没有注入 MuJoCo。上述耗时均不能证明满足 20 ms 实时周期，当前实现尚不能作此声明。**','',
    '## 复现与文件','',
    '`source /home/ganhe/project/b2z1_mpc/relu_hgmpc/env.sh` 后，运行 `python relu_hgmpc/test_paper_gmpc.py`、`bash relu_hgmpc/benchmark_paper_gmpc.sh`、`python relu_hgmpc/report_paper_gmpc.py`。',
    '源码与公式：`relu_hgmpc/paper_gmpc.py`、`relu_hgmpc/PAPER_FIRST_ORDER.md`。逐周期 CSV、逐运行 JSON 和 1 kHz NPZ 位于当前结果目录；comparison.json 保存合并数据、缓存等价检查和源码 SHA256。','']
(OUT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(dict(changes=changes,cache_speed_reduction_percent=cache_gain,stress_changes=stress_changes,
                     timings={k:g['amortized_feedback_ms'] for k,g in groups.items()},
                     equivalence=equivalence),indent=2))
