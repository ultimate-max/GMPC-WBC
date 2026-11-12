"""Matched four-way closed-loop aggregation; all preparation costs included."""
import pathlib,json,csv,platform
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'results/osqp_acceleration'
METHODS=['baseline','reuse','eliminate','combined']
LABELS=['Fresh OSQP','OSQP reuse','Elimination','Both']
TIMES=['feedback_ms','controller_ms','model_ms','setup_ms','solve_ms','reduction_ms',
       'solver_prepare_ms','assembly_ms','dual_recovery_ms','common_check_ms','derivatives_ms']
RES=['qp_primal_relative','qp_dual_relative','qp_normal_relative','local_primal',
     'affine_dynamics_max','first_acceleration_defect','elimination_map_max','elimination_offset_max']
def name(m,s,r):return f'{m}_scale{s}_r{r}'
def stats(a):return dict(mean=float(np.mean(a)),p95=float(np.quantile(a,.95)),
                        p99=float(np.quantile(a,.99)),max=float(np.max(a)))
data={};groups={};hashes=[];summaries={}
for scale in [1,2]:
    for method in METHODS:
        names=[name(method,scale,r) for r in [1,2]];series=[];ss=[]
        for n in names:
            s=json.loads((OUT/f'{n}.json').read_text());summaries[n]=s;ss.append(s)
            with (OUT/f'{n}.csv').open() as f:rows=list(csv.DictReader(f))
            a={k:np.array([float(row[k]) for row in rows]) for k in rows[0]}
            assert s['failed'] is None and s['cycles']==600,(n,s['failed'])
            assert s['min_contacts']==4 and s['missing_contact_ms']==0 and s['torque_clips']==0,n
            assert s['reference_cache_misses']==0,n
            assert np.all(a['qp_converged']==1),n
            assert max(np.max(a[k]) for k in RES[:3])<=2e-7*(1+1e-8),n
            hashes.append(s['source_sha256']);data[n]=a;series.append(a)
        a={k:np.concatenate([v[k] for v in series]) for k in series[0]}
        g=dict(names=names,scale=scale,cycles=len(a['time']),
            position_rmse_mm=float(np.sqrt(np.mean(a['position_error_mm']**2))),
            rotation_rmse_deg=float(np.sqrt(np.mean(a['rotation_error_deg']**2))),
            timings={k:stats(a[k]) for k in TIMES},
            amortized_feedback_ms=(float(np.sum(a['feedback_ms']))+sum(s['precompute_ms'] for s in ss))/len(a['time']),
            run_means=[s['amortized_feedback_ms'] for s in ss],startup_ms=[s['precompute_ms'] for s in ss],
            residual_max={k:float(np.max(a[k])) for k in RES},
            setups_per_run=[s['osqp_setups'] for s in ss],growths_per_run=[s['pattern_growths'] for s in ss],
            qp_iterations=stats(a['qp_iterations']),
            optimized_variables=int(a['optimized_variables'][0]),optimized_constraints=int(a['optimized_constraints'][0]),
            max_slip_path_mm=max(s['max_slip_path_mm'] for s in ss),
            max_foot_displacement_mm=float(np.max(a['foot_center_mm'])),
            min_base_z=float(np.min(a['base_z'])))
        g['timings']['other_ms']=stats(a['feedback_ms']-a['model_ms']-a['setup_ms']-a['solve_ms'])
        groups[f'{method}_scale{scale}']=g
assert all(h==hashes[0] for h in hashes),'Source changed during formal runs'

verification=json.loads((OUT/'verification.json').read_text())
audit=json.loads((OUT/'accuracy_audit.json').read_text())
changes={}
for scale in [1,2]:
    b=groups[f'baseline_scale{scale}']
    for method in METHODS[1:]:
        g=groups[f'{method}_scale{scale}'];changes[f'{method}_scale{scale}']=dict(
            speedup=b['amortized_feedback_ms']/g['amortized_feedback_ms'],
            feedback_reduction_percent=100*(1-g['amortized_feedback_ms']/b['amortized_feedback_ms']),
            position_change_percent=100*(g['position_rmse_mm']/b['position_rmse_mm']-1),
            rotation_change_percent=100*(g['rotation_rmse_deg']/b['rotation_rmse_deg']-1))
repeat_diffs={}
for scale in [1,2]:
    for method in METHODS:
        with np.load(OUT/f'{name(method,scale,1)}_trace.npz') as a,np.load(OUT/f'{name(method,scale,2)}_trace.npz') as b:
            repeat_diffs[f'{method}_scale{scale}']={key:float(np.max(np.abs(a[key]-b[key]))) for key in ['q','v','tau']}
with np.load(OUT/'baseline_scale1_r1_trace.npz') as a,np.load(ROOT/'results/paper_gmpc/aligned_paper_cached_r1_trace.npz') as b:
    original_baseline_diff={k:float(np.max(np.abs(a[k]-b[k]))) for k in ['q','v','tau']}
assert max(original_baseline_diff.values())<1e-10,original_baseline_diff
result=dict(groups=groups,changes=changes,source_sha256=hashes[0],verification=verification,
    accuracy_audit=audit,repeat_differences=repeat_diffs,previous_baseline_differences=original_baseline_diff,
    all_9600_cycles_passed=True,timing_scope=summaries['baseline_scale1_r1']['timing_scope'],
    environment=dict(platform=platform.platform(),python=platform.python_version()))
(OUT/'comparison.json').write_text(json.dumps(result,indent=2))

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':True,'grid.alpha':.2,'figure.facecolor':'white'})
fig,axes=plt.subplots(1,2,figsize=(13,5))
for ax,scale in zip(axes,[1,2]):
    x=np.arange(4);bottom=np.zeros(4)
    for key,label,color in [('model_ms','Model + matrices','#0077b6'),('setup_ms','Assembly / elimination / setup','#55a8a0'),
                            ('solve_ms','Solve + original-QP checks','#e4a64b'),('other_ms','Read / recovery / diagnostics','#8c8795')]:
        v=np.array([groups[f'{m}_scale{scale}']['timings'][key]['mean'] for m in METHODS])
        ax.bar(x,v,bottom=bottom,color=color,label=label);bottom+=v
    for i,m in enumerate(METHODS):ax.text(i,bottom[i]+1,f"{groups[f'{m}_scale{scale}']['amortized_feedback_ms']:.2f}",ha='center')
    ax.set_xticks(x,['Fresh','Reuse','Eliminate','Both']);ax.set_title(f'Reference amplitude {scale}x')
    ax.set_ylim(0,max(bottom)*1.35);ax.set_ylabel('Mean total feedback time [ms]')
axes[0].legend(fontsize=8,loc='upper right')
fig.suptitle('Same first-order GMPC, weights, bounds and original-QP tolerance (2e-7)')
fig.text(.5,.01,'1200 cycles per method and amplitude; labels include startup amortization. Simulation advance excluded.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.03,1,.95]);fig.savefig(OUT/'timing.png',dpi=180);plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(13,7),sharex=True)
for col,scale in enumerate([1,2]):
    for method,label,color in [('baseline','Fresh OSQP','#737f86'),('combined','Reuse + exact elimination','#0077b6')]:
        a=data[name(method,scale,1)]
        for row,key in enumerate(['position_error_mm','rotation_error_deg']):axes[row,col].plot(a['time'],a[key],label=label,color=color,lw=1.3)
    axes[0,col].set_title(f'Reference amplitude {scale}x');axes[0,col].legend(fontsize=9)
    axes[1,col].set_xlabel('Simulation time [s]')
axes[0,0].set_ylabel('End-effector position error [mm]');axes[1,0].set_ylabel('End-effector orientation error [deg]')
fig.suptitle('Tracking comparison: curves nearly overlap at the shared residual tolerance')
fig.text(.5,.01,'Ground-truth state; fixed four-foot contact; actual computation delay is NOT injected.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.03,1,.95]);fig.savefig(OUT/'tracking.png',dpi=180);plt.close(fig)

lines=['# OSQP 结构复用与接触等式消元：闭环对照','',
    '2026-11-10。以已补全的一阶 PaperGMPC 为唯一模型；以下四种方法解同一数学 QP。此前冻结模型不参与本轮比较。','',
    '## 配置与验收','',
    'N=10、步长30 ms、控制间隔20 ms；MuJoCo 真值、Pinocchio、CPU OSQP。代价权重、参考、物理约束、低层控制和原始420维 QP 的 primal/dual/normal-cone 相对残差门槛2e-7全部一致。参考及固定模板缓存对所有组启用。',
    '四组为 baseline（原完整QP，每周期setup）、reuse（完整QP持久化）、eliminate（精确消元，每周期setup）、combined（消元＋持久化）。普通/双幅轨迹，每组两次12 s、600周期，顺序/逆序串行测时。',
    '总时间包括状态读取、原生模型、全部矩阵构造、消元、setup/update、迭代、原始乘子恢复及原问题残差/物理诊断；启动缓存成本摊销。仿真推进和绘图不计入。各子项定义以源码计时边界为准，setup 包含消元，solve 包含原问题残差检查，不能重复相加。','']
for scale in [1,2]:
    lines += [f'## {"普通幅度" if scale==1 else "双幅参考"}','']
    for m in METHODS:
        g=groups[f'{m}_scale{scale}'];t=g['timings']['feedback_ms']
        lines.append(f"- **{m}**：位置 RMSE {g['position_rmse_mm']:.6f} mm，姿态 RMSE {g['rotation_rmse_deg']:.6f}°；含启动总耗时 {g['amortized_feedback_ms']:.3f} ms，P95 {t['p95']:.3f} ms，P99 {t['p99']:.3f} ms；两次均值 {g['run_means'][0]:.3f}/{g['run_means'][1]:.3f} ms。")
    ch=changes[f'combined_scale{scale}']
    lines += ['',f"组合版本相对 baseline 快 **{ch['speedup']:.2f} 倍**，总耗时下降 **{ch['feedback_reduction_percent']:.2f}%**；位置 RMSE 变化 {ch['position_change_percent']:+.3f}%，姿态 RMSE 变化 {ch['rotation_change_percent']:+.3f}%。",'']
lines += ['![总耗时](timing.png)','', '![跟踪误差](tracking.png)','', '## 耗时来源与复用次数','']
for m in METHODS:
    g=groups[f'{m}_scale1'];t=g['timings']
    lines.append(f"- {m}：模型构造 {t['model_ms']['mean']:.3f} ms，组装/消元/setup {t['setup_ms']['mean']:.3f} ms，求解/共同残差 {t['solve_ms']['mean']:.3f} ms，剩余反馈 {t['other_ms']['mean']:.3f} ms；平均迭代 {g['qp_iterations']['mean']:.2f}。")
for m in ['reuse','eliminate','combined']:
    g=groups[f'{m}_scale1'];t=g['timings']
    lines.append(f"- {m} 的 setup 细分：组装 {t['assembly_ms']['mean']:.3f} ms、消元 {t['reduction_ms']['mean']:.3f} ms、OSQP准备 {t['solver_prepare_ms']['mean']:.3f} ms；solve 内的乘子恢复 {t['dual_recovery_ms']['mean']:.3f} ms、原QP残差检查 {t['common_check_ms']['mean']:.3f} ms。")
lines += ['','baseline 未单独拆分 assembly/prepare/check 等内部细项，CSV 中这些新增细项的0是未测标记，不代表没有开销；其 setup/solve/feedback 总计时完整。','']
for scale in [1,2]:
    for m in METHODS:
        g=groups[f'{m}_scale{scale}']
        lines.append(f"- {m}，幅度{scale}：每次600周期中的 setup 次数 {g['setups_per_run']}，结构扩展次数 {g['growths_per_run']}；实际优化维度 {g['optimized_variables']}，约束行数 {g['optimized_constraints']}。")
lines += ['',
    '持久化保留稀疏结构与OSQP工作区；状态变化后的数值矩阵仍全部更新。内部缩放/自适应rho状态保留也会影响迭代数，因此不能把全部收益归于符号分析一项。与原方法一样，每周期使用平移后的控制序列作为primal热启动，dual初值为0。',
    '消元将每节点12个接触力和两组6维接口力用24个等式恢复：420→180变量，240个等式由参数化满足，1040行不等式全部保留。原420维QP仍构造并用于共同残差验收，相关开销未隐藏。','',
    '## 残差、接触与数值等价','']
for m in METHODS:
    ms=[groups[f'{m}_scale{s}'] for s in [1,2]]
    r={k:max(g['residual_max'][k] for g in ms) for k in RES}
    lines.append(f"- {m}：原QP相对 primal/dual/normal 最大值 {r[RES[0]]:.3e}/{r[RES[1]]:.3e}/{r[RES[2]]:.3e}；消元映射/偏置误差 {r['elimination_map_max']:.3e}/{r['elimination_offset_max']:.3e}。")
vr=verification['records']
lines += ['',
    '全部16次闭环、9600周期通过原问题共同残差门槛及物理门槛；1 kHz记录四足接触全部保持，没有力矩裁剪。',
    f"独立变换测试：E T 最大误差 {max(r['map_error'] for r in vr):.3e}，E c−b 最大误差 {max(r['offset_error'] for r in vr):.3e}；随机输入目标等价相对误差 {max(r['objective_transform_error'] for r in vr):.3e}，全部不等式变换误差 {max(r['inequality_transform_error'] for r in vr):.3e}。",
    f"在2e-11精度下，完整与优化版本首步力矩最大差异 {max(v['first_torque_difference_Nm'] for r in vr for v in r['variants'].values()):.6f} Nm；结构零变非零、重建、复用路径均经过测试。",
    f"本轮 baseline 与上一轮相同模型的 q/v/tau 最大差异为 {original_baseline_diff}，确认基线行为未变。",'',
    '## 有限精度审计','',
    '数学等价不意味着在有限残差门槛下每个力矩数值逐位相同。对相同回放状态、同一物理QP，用2e-12高精度消元解作参考，2e-7门槛下各方法首步力矩的差异如下。这是离线数值诊断，不计作闭环测速，也不假定高精度参考是数学精确解。','']
for m in METHODS:
    vals=[r['variants'][m]['first_torque_difference_Nm'] for r in audit['records']]
    lines.append(f"- {m}：6个状态上的最大差异 {max(vals):.6f} Nm，中位数 {np.median(vals):.6f} Nm。")
lines += ['',
    '上述差异解释了为什么同一代价/约束下仍可能出现小幅轨迹变化；本轮没有宣称改变模型或提高控制算法精度。原共同门槛保持不变；若后续要比较更严格的力矩一致性，应另做更紧门槛的四组对照。',
    '完整q/v/tau重复差异、每周期残差和源文件指纹见comparison.json及CSV。两次确定性仿真用于计时核查，不代表统计鲁棒性实验。','',
    '## 局限与下一步','',
    '实际计算延迟未注入仿真。是否达到20 ms实时周期必须同时看总耗时尾部；不能仅凭平均加速比宣称实时达标。当前只覆盖固定四足站立与两档参考幅度，不覆盖接触切换或奇异构型。',
    '下一步可把预测/代价/约束直接构造在180维消元坐标，减少先形成420维矩阵再投影的开销；但应保留独立原问题残差验收。当前结果证明工程加速有效，不代表OSQP复用或等式消元本身是新的论文贡献。','',
    '## 文件与复现','',
    '工程：`/home/ganhe/project/b2z1_mpc/relu_hgmpc/`。加载env.sh后运行test_osqp_acceleration.py、benchmark_osqp_acceleration.sh、audit_osqp_accuracy.py、report_osqp_acceleration.py。公式推导和边界见OSQP_ACCELERATION.md。',
    '实现遵循[OSQP固定稀疏结构数值更新接口](https://osqp.org/docs/interfaces/python.html)，不复用失效的数值矩阵。原始CSV/JSON和1 kHz NPZ全部保存在results/osqp_acceleration/。','']
(OUT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(dict(changes=changes,baseline_matches_previous=original_baseline_diff,
    total_ms={k:g['amortized_feedback_ms'] for k,g in groups.items()}),indent=2))
