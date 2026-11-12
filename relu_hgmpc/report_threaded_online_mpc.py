"""Report actual thread deadlines, full CPU cost and matched physical metrics."""
import pathlib,json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from report_online_wbc_mpc import rms,hf,clean
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'results/online_wbc_mpc_threads'
CASES=['normal','double','push'];MODES=['serial_quadprog','thread_quadprog','thread_osqp']
LABELS=['Serial / quadprog','Threads / quadprog','Threads / screened OSQP']
COLORS=['#8c8c8c','#df8b27','#1977bd']

def main():
    records=[];data={};summaries={}
    for case in CASES:
        for mode in MODES:
            for rep in [1,2]:
                name=f'{case}_{mode}_r{rep}';s=json.loads((OUT/f'{name}.json').read_text());summaries[name]=s
                a=np.atleast_1d(np.genfromtxt(OUT/f'{name}.csv',delimiter=',',names=True));data[name]=a
                tr=np.load(OUT/f'{name}_trace.npz');t=tr['trace'];plans=s['plans']
                complete=s['failed'] is None and s['simulated_cycles']==2400
                latency=np.array([p['completion_from_tick_ms'] for p in plans]);worker=np.array([p['worker_wall_ms'] for p in plans])
                times=np.array([p['request_time'] for p in plans])
                rec=dict(case=case,mode=mode,repeat=rep,complete=int(complete),duration=s['simulated_cycles']*.005,
                    planner_count=len(plans),skipped_requests=s['skipped_requests'],installed_plans=s['installed_plans'],
                    effective_request_hz=len(plans)/(s['simulated_cycles']*.005),
                    request_gap_max_ms=float(np.max(np.diff(times))*1000),
                    planner_deadline_misses=sum(p['deadline_miss'] for p in plans),
                    late_installs=sum(p['install_late_ms'] is not None and p['install_late_ms']>1e-6 for p in plans),
                    planner_latency_mean_ms=float(latency.mean()),planner_latency_p99_ms=float(np.quantile(latency,.99)),planner_latency_max_ms=float(latency.max()),
                    worker_wall_mean_ms=float(worker.mean()),worker_cpu_mean_ms=float(np.mean([p['worker_cpu_ms'] for p in plans])),
                    qp_solve_mean_ms=float(np.mean([p['solve_ms'] for p in plans])),construction_mean_ms=float(np.mean([p['construction_ms'] for p in plans])),
                    total_cpu_ms=s['main_cpu_ms']+s['worker_cpu_ms'],main_cpu_ms=s['main_cpu_ms'],planner_cpu_ms=s['worker_cpu_ms'],
                    initialization_ms=s['initialization_ms'],wall_duration_s=s['wall_duration_s'],
                    wbc_response_mean_ms=float(a['control_response_ms'].mean()),wbc_response_p99_ms=float(np.quantile(a['control_response_ms'],.99)),
                    wbc_response_max_ms=float(a['control_response_ms'].max()),wbc_deadline_misses=int(a['wbc_deadline_miss'].sum()),
                    wbc_qp_residual=max(float(a[k].max()) for k in ['qp_primal_relative','qp_dual_relative','qp_normal_relative']),
                    wbc_dynamics_residual=float(a['dynamics_residual'].max()),wbc_contact_residual=float(a['contact_residual'].max()),
                    wbc_constraint_violation=float(a['constraint_violation'].max()),hierarchy_lock_max=float(a['task_lock_max'].max()),
                    reference_continuity_max=float(a['reference_continuity'].max()),wbc_backup_solves=int(a['qp_fallbacks'].sum()),
                    mpc_qp_residual=max(p['qp_residual'] for p in plans),mpc_bound_violation=max(p['physical_bound_violation'] for p in plans),
                    nonlinear_dynamics_defect=max(p['nonlinear_dynamics_defect_max'] for p in plans),
                    nonlinear_contact_defect=max(p['nonlinear_contact_defect_max'] for p in plans),
                    clips=int(a['clips'].sum()),missing_contact_ms=int(np.sum(t[:,1:5].sum(axis=1)<4)),min_contacts=int(t[:,1:5].sum(axis=1).min()))
                if mode=='thread_osqp':
                    rec.update(osqp_constraints_mean=float(np.mean([p['osqp_constraints'] for p in plans])),
                        osqp_constraints_max=max(p['osqp_constraints'] for p in plans),osqp_rounds_max=max(p['osqp_rounds'] for p in plans),
                        osqp_setups=max(p['osqp_setups'] for p in plans),active_face_fastpath_count=sum(p['active_face_fastpath'] for p in plans))
                else:rec.update(osqp_constraints_mean=np.nan,osqp_constraints_max=np.nan,osqp_rounds_max=np.nan,osqp_setups=np.nan,active_face_fastpath_count=np.nan)
                for task in ['base','ee']:
                    for kind,unit in [('position','mm'),('rotation','deg')]:
                        rec[f'{task}_{kind}_rmse_{unit}']=rms(a[f'{task}_{kind}_error_{unit}']) if complete else np.nan
                for window,mask in [('all',t[:,0]>=2),('quiet',t[:,0]>=8)]:
                    for low in [5,10]:
                        for key,x in [('leg_velocity',t[:,111:123]),('arm_velocity',t[:,123:129]),('leg_torque',t[:,57:69]),('arm_torque',t[:,69:75])]:
                            rec[f'{key}_{window}_hf{low}']=hf(x[mask],low,True)
                records.append(rec)
    hashes={f:len({s['source_sha256'][f] for s in summaries.values()})==1 for f in next(iter(summaries.values()))['source_sha256']}
    assert all(hashes.values()),hashes
    means={};changes={}
    for case in CASES:
        assert len({summaries[f'{case}_{m}_r{r}']['target_sha256'] for m in MODES for r in [1,2]})==1
        means[case]={}
        for mode in MODES:
            rr=[r for r in records if r['case']==case and r['mode']==mode]
            means[case][mode]={k:float(np.mean([x[k] for x in rr])) for k in rr[0] if k not in ['case','mode','repeat']}
        changes[case]={}
        for baseline in MODES[:2]:
            b=means[case][baseline];e=means[case]['thread_osqp']
            changes[case][baseline]={k:100*(e[k]/b[k]-1) if b[k]!=0 and e['complete']==1 and b['complete']==1 else None for k in b}
    aggregate={}
    for mode in MODES:
        rr=[r for r in records if r['mode']==mode]
        aggregate[mode]={k:sum(r[k] for r in rr) for k in ['complete','planner_count','skipped_requests','planner_deadline_misses','late_installs','wbc_deadline_misses','missing_contact_ms','clips']}
        for k in ['planner_latency_max_ms','wbc_response_max_ms','mpc_qp_residual','mpc_bound_violation','wbc_qp_residual','wbc_contact_residual','hierarchy_lock_max','reference_continuity_max']:
            aggregate[mode][k]=max(r[k] for r in rr)
        if mode=='thread_osqp':aggregate[mode]['active_face_fastpath_count']=sum(r['active_face_fastpath_count'] for r in rr)
    native_checks=json.loads((OUT/'native_equivalence.json').read_text())
    result=dict(aggregate=aggregate,means=means,osqp_change_percent=changes,source_agreement=hashes,
        protocol=json.loads((OUT/'protocol.json').read_text()),native_checks=native_checks,
        fast_reference_verification=json.loads((OUT/'fast_reference_verification.json').read_text()),
        limitations='OSQP constraint generation has no proven fixed runtime bound; all original constraints/KKT checked before packet acceptance. Wall-paced WSL simulation, not a hard-real-time guarantee. Each mode has two repeats only.')
    with (OUT/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    (OUT/'comparison.json').write_text(json.dumps(clean(result),indent=2))
    plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.25,'figure.dpi':140})
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    fields=[('wbc_deadline_misses','WBC >5 ms / 2400 cycles'),('skipped_requests','Skipped MPC slots / 600'),
            ('total_cpu_ms','Controller CPU / 12 s (ms)'),('ee_position_rmse_mm','Tool position RMSE (mm)'),
            ('arm_velocity_quiet_hf5','Arm velocity 5-100 Hz (rad/s)'),('arm_torque_quiet_hf5','Arm torque 5-100 Hz (Nm)')]
    for ax,(key,title) in zip(axes.ravel(),fields):
        for i,mode in enumerate(MODES):
            ax.bar(np.arange(3)+(i-1)*.24,[means[c][mode][key] for c in CASES],.24,color=COLORS[i],label=LABELS[i])
        ax.set_xticks(range(3),CASES);ax.set_title(title,fontsize=10);ax.set_ylim(bottom=0)
    axes[0,0].legend(fontsize=7);fig.suptitle('Actual wall-paced threads; same model, horizon and original-QP accuracy gate')
    fig.tight_layout();fig.savefig(OUT/'comparison.png');fig.savefig(OUT/'comparison.pdf');plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(12,8),sharex='col')
    for j,case in enumerate(CASES):
        for i,mode in enumerate(MODES):
            a=data[f'{case}_{mode}_r1'];s=summaries[f'{case}_{mode}_r1']
            axes[0,j].plot(a['time'],a['ee_position_error_mm'],color=COLORS[i],lw=1,label=LABELS[i])
            axes[1,j].plot(a['time'],a['control_response_ms'],color=COLORS[i],lw=.45,alpha=.75)
            axes[2,j].plot([p['request_time'] for p in s['plans']],[p['completion_from_tick_ms'] for p in s['plans']],color=COLORS[i],lw=.6)
        axes[0,j].set_title(case.capitalize());axes[1,j].axhline(5,color='red',ls='--',lw=1)
        axes[2,j].axhline(20,color='red',ls='--',lw=1);axes[2,j].set_xlabel('Time (s)')
        if case=='push':
            for a in axes[:,j]:a.axvspan(6,6.15,color='red',alpha=.08)
    axes[0,0].set_ylabel('Tool position error (mm)');axes[1,0].set_ylabel('WBC response from tick (ms)')
    axes[2,0].set_ylabel('MPC completion from tick (ms)');axes[0,0].legend(fontsize=7)
    fig.suptitle('Tracking and actual completion times, repeat 1')
    fig.tight_layout();fig.savefig(OUT/'tracking_deadlines.png');fig.savefig(OUT/'tracking_deadlines.pdf');plt.close(fig)
    ao=aggregate['thread_osqp'];aq=aggregate['thread_quadprog'];ase=aggregate['serial_quadprog']
    cpu_change=[changes[c]['thread_quadprog']['total_cpu_ms'] for c in CASES]
    track_change=[abs(changes[c]['thread_quadprog']['ee_position_rmse_mm']) for c in CASES]
    lines=['# 双线程＋OSQP 的 50 Hz MPC／200 Hz WBC','',
        f"完成 {sum(r['complete'] for r in records)}/{len(records)} 次完整闭环。组合加速方案相对双线程 quadprog，三个工况总控制 CPU 时间降低 {min(-x for x in cpu_change):.2f}%–{max(-x for x in cpu_change):.2f}%，末端位置 RMSE 变化最大 {max(track_change):.4g}%。", 
        f"加速版实际完成 {ao['planner_count']}/3600 次计划请求，超过 20 ms {ao['planner_deadline_misses']} 次、跳过 {ao['skipped_requests']} 槽；WBC 超过 5 ms {ao['wbc_deadline_misses']}/14400 次（串行为 {ase['wbc_deadline_misses']} 次，双线程原求解器为 {aq['wbc_deadline_misses']} 次）。这是明显的调度改善，但尚未达到所有周期均准时的验收目标。", 
        '计算收益来自原生运算、线程隔离及经过完整验收的活跃约束复用。实际关节高频指标没有整体下降，尤其腿部力矩指标有升有降；不把本轮计算加速解释为普遍减振。','',
        '## 实现','',
        '- MPC 使用独立 ThreadPoolExecutor 工作线程与独立 Pinocchio／MuJoCo 数据、OSQP 工作区。主循环只消费已经完成的结果，不阻塞等待规划；超时结果在后续控制周期连续交接，忙时错过的 20 ms 请求槽明确记录。两线程绑定不同 CPU 核心，BLAS 单线程。',
        '- 预测矩阵构造移到 C++，计算期间释放 Python 执行锁。五次 SE(3) 参考求值及专用接触运动学也使用原生实现；仅需雅可比时不再求取质量矩阵与偏置力。独立比较原版／原生矩阵、目标函数、位姿／速度／加速度以及接触雅可比／漂移项。',
        '- OSQP：每次对当前 Hessian 做 Cholesky 变换；复用固定约束槽的符号结构，矩阵数值变化时仍重新分解；热启动按 20 ms／40 ms 的半节点时间移动。先求上一活跃集与无约束解违反的约束，检查全部 1340 行原约束后补入遗漏，直到完整原问题的原始／对偶／法锥残差均不超过 2e-7。没有放宽物理约束，也没有把未检查的约束当作可以删除。',
        '- 在 OSQP 前增加活跃约束预测：使用前次活跃面，通过小型 SVD 解等式约束 QP。只有全部原约束误差不超过 1e-9 且完整 KKT 不超过 1e-10 时直接接受；否则交给上述 OSQP。快路径命中率单独记录，因此不能把所有周期都称为执行了 ADMM 迭代。',
        '- 对照均采用相同原生构造、参考运算、模型、N=10、40 ms 预测步长、0.4 s 时域、权重、200 Hz 几何 WBC。三模式为串行 quadprog、双线程 quadprog、双线程筛选 OSQP。普通、双幅与 120 N 外力扰动各两次 12 s，第二轮反转运行顺序。',
        '- 预热／符号初始化在控制时钟启动前独立记录。计时既包含真实请求到完成的延迟，也记录主线程和规划线程各自的 CPU 时间；并行墙钟时间不相加冒充 CPU 消耗。总控制 CPU 包含构造、数据复制、全部求解、完整约束／节点检查、参考生成和 WBC，排除公共目标预计算、日志和物理引擎积分。','',
        '## 结果','']
    for case in CASES:
        lines += [f'### {case}','']
        for i,mode in enumerate(MODES):
            m=means[case][mode]
            lines.append(f"- {LABELS[i]}：完成率 {m['complete']:.0%}；平均每 12 s 求解 {m['planner_count']:.1f} 次、跳过 {m['skipped_requests']:.1f} 个规划槽；WBC 超时 {m['wbc_deadline_misses']:.1f}/2400；控制 CPU {m['total_cpu_ms']/1000:.4g} s。")
            lines.append(f"  MPC 从控制周期起点到完成平均 {m['planner_latency_mean_ms']:.4g} ms、两次运行 p99 平均 {m['planner_latency_p99_ms']:.4g} ms；末端位置／姿态 RMSE {m['ee_position_rmse_mm']:.4g} mm／{m['ee_rotation_rmse_deg']:.4g}°，底盘 {m['base_position_rmse_mm']:.4g} mm／{m['base_rotation_rmse_deg']:.4g}°。")
        for baseline in MODES[:2]:
            d=changes[case][baseline]
            if d['total_cpu_ms'] is None:continue
            lines.append(f"- 筛选 OSQP 相对 {baseline}：CPU {d['total_cpu_ms']:+.3g}%，末端位置 RMSE {d['ee_position_rmse_mm']:+.3g}%；安静窗口实际腿／臂速度高频 RMS {d['leg_velocity_quiet_hf5']:+.3g}%／{d['arm_velocity_quiet_hf5']:+.3g}%，施加腿／臂力矩高频 RMS {d['leg_torque_quiet_hf5']:+.3g}%／{d['arm_torque_quiet_hf5']:+.3g}%。")
        lines.append('')
    lines += ['## 时限、精度及局限','']
    lines.append(f"OSQP 组合方案中，{aggregate['thread_osqp']['active_face_fastpath_count']}/{aggregate['thread_osqp']['planner_count']} 次由完整验收的活跃面快路径直接完成，余下调用 OSQP。")
    for i,mode in enumerate(MODES):
        a=aggregate[mode]
        lines.append(f"- {LABELS[i]}：完整 {a['complete']}/6 次；规划 {a['planner_count']} 次，超过 20 ms {a['planner_deadline_misses']} 次、跳过 {a['skipped_requests']} 槽，最慢 {a['planner_latency_max_ms']:.4g} ms；WBC 超过 5 ms {a['wbc_deadline_misses']}/14400，最慢 {a['wbc_response_max_ms']:.4g} ms；失去四足接触 {a['missing_contact_ms']} ms，力矩裁剪 {a['clips']} 次。")
    for key in ['mpc_qp_residual','mpc_bound_violation','wbc_qp_residual','wbc_dynamics_residual','wbc_contact_residual','wbc_constraint_violation','hierarchy_lock_max','reference_continuity_max','nonlinear_dynamics_defect','nonlinear_contact_defect']:
        lines.append(f"- {key} 全部运行最大值：{max(r[key] for r in records):.6g}")
    lines += ['',
        '不能把名义 50 Hz 等同于每次均准时：必须同时查看跳槽、真实完成时刻和 WBC 时限。本实现没有证明固定最坏迭代次数或硬实时性；WSL／Python／宿主操作系统仍可能引入尾延迟。存在跳槽时有效规划频率低于严格 50 Hz，这一点不以平均耗时掩盖。',
        '这是按墙钟运行的仿真调度测试，不是实机时限认证。物理引擎仍按每次控制推进 5 个 1 ms 步长，错过墙钟时限时不另外插入硬件保持力矩的丢帧模型；跟踪指标应结合超时记录解释。',
        '任务与物理模型仍为单次冻结系数、SE(3) 一阶任务误差的参考生成 MPC；小 QP 残差不证明非线性预测严格可行。广义动力学缺陷含 N 与 N·m 分量，接触加速度缺陷单位 m/s²。',
        '实际关节高频量来自 1 kHz 真值速度与施加力矩；安静窗口 8–12 s，5–100 Hz；另存 10–100 Hz 与 2–12 s 全窗口指标。两个墙钟重复并不足以证明统计显著性。',
        '首版 Python 双线程、初版原生构造以及进一步参考优化前的运行保存在 diagnostics 和 smoke 日志，最终统计不混入这些结果。原来的低频反馈触发策略仍未纳入满足最低 50 Hz 要求的方案。','',
        '## 复现','',
        '工程：/home/ganhe/project/b2z1_mpc。激活 relu_hgmpc/env.sh，运行 test_native_online_mpc.py、test_fast_online_reference.py、benchmark_threaded_online_mpc.py、report_threaded_online_mpc.py。默认单次运行：python relu_hgmpc/run_threaded_online_mpc.py --case double --name demo。默认使用真正双线程与筛选 OSQP。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(clean(dict(aggregate=aggregate,means=means,changes=changes)),indent=2))

if __name__=='__main__':main()
