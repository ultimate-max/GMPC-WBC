"""Full-cost and accuracy audit of the 50 Hz MPC baseline."""
import csv,json,pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from report_online_wbc_mpc import hf,rms,clean

ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'results/online_wbc_mpc_50hz'
CASES=['normal','double','push'];POLICIES=['fixed10','fixed50']
LABELS={'fixed10':'10 Hz (historical ablation)','fixed50':'50 Hz'}
COLORS={'fixed10':'#d27928','fixed50':'#1673b9'}

def main():
    records=[];summaries={};data={};traces={};repeat_checks={}
    for case in CASES:
        for mode in POLICIES:
            for rep in [1,2]:
                name=f'{case}_{mode}_r{rep}'
                s=json.loads((OUT/f'{name}.json').read_text());summaries[name]=s
                a=np.atleast_1d(np.genfromtxt(OUT/f'{name}.csv',delimiter=',',names=True));data[name]=a
                tr=np.load(OUT/f'{name}_trace.npz');traces[name]=tr;t=tr['trace'];base=tr['base']
                plans=s['plans'];cost=np.array([p['wall_ms'] for p in plans]);times=np.array([p['time'] for p in plans])
                complete=s['failed'] is None and s['simulated_cycles']==2400
                windows=np.sum(a['total_ms'][:len(a)//4*4].reshape(-1,4),axis=1)
                row=dict(case=case,policy=mode,repeat=rep,complete=int(complete),duration=s['simulated_cycles']*.005,
                    replan_count=len(plans),total_control_ms=s['total_control_ms'],total_planner_ms=s['total_planner_ms'],
                    total_wbc_reference_ms=s['total_wbc_reference_ms'],
                    planner_mean_ms=float(cost.mean()),planner_p99_ms=float(np.quantile(cost,.99)),planner_max_ms=float(cost.max()),
                    construction_mean_ms=float(np.mean([p['construction_ms'] for p in plans])),
                    qp_solve_mean_ms=float(np.mean([p['solve_ms'] for p in plans])),
                    planner_over_20ms=int(np.sum(cost>20.)),
                    period_min_ms=float(np.min(np.diff(times))*1000),period_max_ms=float(np.max(np.diff(times))*1000),
                    single_core_window_mean_ms=float(windows.mean()),single_core_window_p99_ms=float(np.quantile(windows,.99)),
                    single_core_window_max_ms=float(windows.max()),single_core_window_over_20ms=int(np.sum(windows>20.)),
                    wbc_mean_ms=float(a['wbc_reference_ms'].mean()),wbc_p99_ms=float(np.quantile(a['wbc_reference_ms'],.99)),
                    wbc_max_ms=float(a['wbc_reference_ms'].max()),wbc_over_5ms=int(np.sum(a['wbc_reference_ms']>5.)),
                    clips=s['torque_clips'],missing_contact_ms=int(np.sum(t[:,1:5].sum(axis=1)<4)),
                    min_contacts=int(t[:,1:5].sum(axis=1).min()),
                    mpc_qp_residual=max(p['qp_residual'] for p in plans),mpc_bound_violation=max(p['physical_bound_violation'] for p in plans),
                    nonlinear_dynamics_defect=max(p['nonlinear_dynamics_defect_max'] for p in plans),
                    nonlinear_contact_defect=max(p['nonlinear_contact_defect_max'] for p in plans),
                    wbc_qp_residual=max(float(a[k].max()) for k in ['qp_primal_relative','qp_dual_relative','qp_normal_relative']),
                    wbc_dynamics_residual=float(a['dynamics_residual'].max()),wbc_contact_residual=float(a['contact_residual'].max()),
                    wbc_constraint_violation=float(a['constraint_violation'].max()),
                    hierarchy_lock_max=float(a['task_lock_max'].max()),reference_handover_max=float(a['reference_continuity'].max()),
                    wbc_backup_solves=int(a['qp_fallbacks'].sum()))
                for task in ['base','ee']:
                    for kind,unit in [('position','mm'),('rotation','deg')]:
                        values=a[f'{task}_{kind}_error_{unit}']
                        row[f'{task}_{kind}_rmse_{unit}']=rms(values) if complete else np.nan
                for window,mask in [('all',t[:,0]>=2),('quiet',t[:,0]>=8)]:
                    for low in [5,10]:
                        for label,x in [('leg_velocity',t[:,111:123]),('arm_velocity',t[:,123:129]),
                                        ('leg_torque',t[:,57:69]),('arm_torque',t[:,69:75])]:
                            row[f'{label}_{window}_hf{low}']=hf(x[mask],low,True)
                records.append(row)
            x=traces[f'{case}_{mode}_r1'];y=traces[f'{case}_{mode}_r2']
            repeat_checks[f'{case}_{mode}']={k:bool(np.array_equal(x[k],y[k])) for k in ['q','v','tau','acc','trace','base','reference_q']}
    agreement={f:len({s['source_sha256'][f] for s in summaries.values()})==1 for f in next(iter(summaries.values()))['source_sha256']}
    assert all(agreement.values()),agreement
    means={};changes={}
    for case in CASES:
        assert len({summaries[f'{case}_{m}_r{r}']['target_sha256'] for m in POLICIES for r in [1,2]})==1
        means[case]={}
        for mode in POLICIES:
            rr=[x for x in records if x['case']==case and x['policy']==mode]
            means[case][mode]={k:float(np.mean([x[k] for x in rr])) for k in rr[0] if k not in ['case','policy','repeat']}
        b=means[case]['fixed10'];e=means[case]['fixed50']
        changes[case]={k:100*(e[k]/b[k]-1) if b[k]!=0 and b['complete']==1 and e['complete']==1 else None for k in b}
    r50=[r for r in records if r['policy']=='fixed50']
    aggregate=dict(complete_runs=sum(r['complete'] for r in records),total_runs=len(records),
        complete_50hz_runs=sum(r['complete'] for r in r50),
        planner_calls_50hz=sum(r['replan_count'] for r in r50),
        planner_over_20ms_50hz=sum(r['planner_over_20ms'] for r in r50),
        single_core_windows_over_20ms_50hz=sum(r['single_core_window_over_20ms'] for r in r50),
        single_core_windows_50hz=sum(int(r['duration']/.02) for r in r50),
        wbc_over_5ms_50hz=sum(r['wbc_over_5ms'] for r in r50),
        min_contacts=min(r['min_contacts'] for r in records),clips=sum(r['clips'] for r in records))
    result=dict(aggregate=aggregate,means=means,change50_vs10_percent=changes,repeat_checks=repeat_checks,
        source_agreement=agreement,protocol=json.loads((OUT/'protocol.json').read_text()),
        verification=json.loads((OUT/'verification.json').read_text()),
        timing_limitation='Fixed logical release even on measured overruns. Single-core 20ms window sums are a necessary compute-load check, not an actual preemptive scheduler simulation or hard-real-time proof.')
    with (OUT/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    (OUT/'comparison.json').write_text(json.dumps(clean(result),indent=2))
    plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.25,'figure.dpi':140})
    fig,ax=plt.subplots(3,3,figsize=(12,8),sharex='col')
    for j,case in enumerate(CASES):
        for mode in POLICIES:
            a=data[f'{case}_{mode}_r1'];s=summaries[f'{case}_{mode}_r1'];col=COLORS[mode]
            ax[0,j].plot(a['time'],a['base_position_error_mm'],color=col,lw=1,label=LABELS[mode])
            ax[1,j].plot(a['time'],a['ee_position_error_mm'],color=col,lw=1)
            ax[2,j].plot([p['time'] for p in s['plans']],[p['wall_ms'] for p in s['plans']],color=col,lw=.7)
        ax[0,j].set_title(case.capitalize());ax[2,j].axhline(20,color='red',ls='--',lw=1)
        ax[2,j].set_xlabel('Time (s)')
        if case=='push':
            for a in ax[:,j]:a.axvspan(6,6.15,color='red',alpha=.08)
    ax[0,0].set_ylabel('Base position error (mm)');ax[1,0].set_ylabel('Tool position error (mm)')
    ax[2,0].set_ylabel('Full MPC computation (ms)');ax[0,0].legend(fontsize=8)
    fig.suptitle('50 Hz MPC: same 0.4 s horizon and 20 ms release latency for both policies')
    fig.tight_layout();fig.savefig(OUT/'tracking_timing.png');fig.savefig(OUT/'tracking_timing.pdf');plt.close(fig)
    fig,ax=plt.subplots(2,3,figsize=(12,7))
    fields=[('total_control_ms','Total online compute / 12 s (ms)'),('ee_position_rmse_mm','Tool position RMSE (mm)'),
        ('base_position_rmse_mm','Base position RMSE (mm)'),('single_core_window_p99_ms','20 ms single-core load: p99 (ms)'),
        ('arm_velocity_quiet_hf5','Arm velocity 5-100 Hz (rad/s)'),('arm_torque_quiet_hf5','Arm torque 5-100 Hz (Nm)')]
    for a,(key,title) in zip(ax.ravel(),fields):
        for i,mode in enumerate(POLICIES):
            a.bar(np.arange(3)+(i-.5)*.32,[means[c][mode][key] for c in CASES],.32,color=COLORS[mode],label=LABELS[mode])
        a.set_xticks(range(3),CASES);a.set_title(title,fontsize=10);a.set_ylim(bottom=0)
        if key=='single_core_window_p99_ms':a.axhline(20,color='red',ls='--',lw=1)
    ax[0,0].legend(fontsize=8);fig.suptitle('50 Hz baseline; 10 Hz is a historical frequency ablation only')
    fig.tight_layout();fig.savefig(OUT/'comparison.png');fig.savefig(OUT/'comparison.pdf');plt.close(fig)
    lines=['# 50 Hz 在线 MPC＋几何 WBC 验证','',
        f"完成 {aggregate['complete_runs']}/{len(records)} 次 12 s 闭环，其中 50 Hz 完成 {aggregate['complete_50hz_runs']}/6 次。",'',
        '## 设置与计时口径','',
        '- 当前默认 MPC 为 50 Hz（每 20 ms 重新读取状态并求解），WBC 为 200 Hz，MuJoCo 为 1 kHz。预测节点步长仍是 40 ms、N=10、时域 0.4 s、180 个力矩变量；预测步长与重规划周期不同。',
        '- 50 Hz 与历史 10 Hz 对照使用相同 20 ms 交接延迟、目标、预测模型、约束、权重、QP 精度、五次参考插值和几何 WBC。普通、双幅与 120 N 外力扰动工况各重复两次，第二次反转策略顺序。10 Hz 只是频率消融，不满足当前最低 50 Hz 要求。',
        '- 原先最低 5 Hz 的反馈触发策略不纳入合格方案。下一步的触发式加速必须在至少 50 Hz 更新约束下设计，不能把降至 5–10 Hz 本身称为同等性能加速。',
        '- 完整规划耗时包含参考前推、矩阵构造、数据传递、求解、预测节点非线性复核及轨迹构造。总在线计算还包含状态读取、交接、参考采样、WBC 和调度。公共目标几何预计算、日志和 MuJoCo 积分不计入控制器在线耗时。',
        '- 逐次记录规划超过 20 ms、WBC＋参考采样超过 5 ms，以及一个 20 ms 窗口内一次规划与四次 WBC 等在线工作的计算总和。单核窗口总和仅是计算负载审计，不等于真实可抢占调度验证。',
        '- 当前仍为串行仿真中的逻辑 20 ms 交接；即使测得超时，模拟释放时刻也不改变，超时显式报告。存在超时时不能由这些跟踪曲线宣称实际实时性能。','',
        '## 结果','']
    for case in CASES:
        lines += [f'### {case}','']
        for mode in POLICIES:
            m=means[case][mode]
            if m['complete']!=1:
                lines.append(f"- {LABELS[mode]}：有未完成运行，失败记录见原始 JSON；不作完整时域性能结论。");continue
            lines.append(f"- {LABELS[mode]}：每次 {m['replan_count']:.0f} 次 MPC；底盘位置 RMSE {m['base_position_rmse_mm']:.4g} mm、姿态 {m['base_rotation_rmse_deg']:.4g}°；末端位置 {m['ee_position_rmse_mm']:.4g} mm、姿态 {m['ee_rotation_rmse_deg']:.4g}°；每 12 s 总在线计算 {m['total_control_ms']/1000:.4g} s。")
            lines.append(f"  完整规划平均 {m['planner_mean_ms']:.4g} ms；两次运行 p99 的平均 {m['planner_p99_ms']:.4g} ms；QP 求解平均 {m['qp_solve_mean_ms']:.4g} ms。")
        d=changes[case]
        if d['total_control_ms'] is not None:
            lines.append(f"- 50 Hz 相对 10 Hz：总计算 {d['total_control_ms']:+.3g}%，底盘位置误差 {d['base_position_rmse_mm']:+.3g}%，末端位置误差 {d['ee_position_rmse_mm']:+.3g}%；安静窗口（8–12 s）实际腿/臂速度高频 RMS {d['leg_velocity_quiet_hf5']:+.3g}% / {d['arm_velocity_quiet_hf5']:+.3g}%，施加腿/臂力矩高频 RMS {d['leg_torque_quiet_hf5']:+.3g}% / {d['arm_torque_quiet_hf5']:+.3g}%。")
        lines.append('')
    lines += ['## 实时性与残差','',
        f"50 Hz 合计 {aggregate['planner_calls_50hz']} 次 MPC，超过 20 ms 共 {aggregate['planner_over_20ms_50hz']} 次；完整规划最慢 {max(r['planner_max_ms'] for r in r50):.4g} ms。", 
        f"50 Hz 单核 20 ms 计算窗口超预算 {aggregate['single_core_windows_over_20ms_50hz']}/{aggregate['single_core_windows_50hz']} 个；最慢窗口 {max(r['single_core_window_max_ms'] for r in r50):.4g} ms。WBC＋参考采样超 5 ms 共 {aggregate['wbc_over_5ms_50hz']} 次。", 
        f"全部运行最少接触数 {aggregate['min_contacts']}，力矩裁剪 {aggregate['clips']} 次。重复轨迹逐元素相同：{all(all(v.values()) for v in repeat_checks.values())}；这不是统计显著性的证明。",'']
    for key in ['mpc_qp_residual','mpc_bound_violation','wbc_qp_residual','wbc_dynamics_residual','wbc_contact_residual',
                'wbc_constraint_violation','hierarchy_lock_max','reference_handover_max','nonlinear_dynamics_defect','nonlinear_contact_defect']:
        lines.append(f"- {key}：{max(r[key] for r in records):.6g}")
    lines += ['',
        '这是单次冻结动力学/接触系数、任务误差作一阶 SE(3) 展开的参考生成 MPC；不是收敛 SQP-NMPC。MPC 初值为交接时刻已提交的参考 q/v/a，模型和执行边界使用最新实测状态。预测时域非线性缺陷仍单独记录，小 QP 残差不能视为严格非线性可行性。',
        'nonlinear_dynamics_defect 含 N 与 N·m 的广义残差分量，仅作数值诊断；nonlinear_contact_defect 单位为 m/s²。实际抖振指标来自 1 kHz 关节速度和施加力矩，Welch 频带为 5–100 Hz，同时保存 10–100 Hz 和包含扰动的 2–12 s 指标。',
        '本轮只变更重规划频率与交接延迟，尚未比较线性与五次插值，不把插值阶数本身解释为已验证的性能优势。','',
        '## 复现','',
        '工程：/home/ganhe/project/b2z1_mpc。激活 relu_hgmpc/env.sh 后运行 test_online_wbc_mpc_50hz.py、benchmark_online_wbc_mpc_50hz.py、report_online_wbc_mpc_50hz.py。原始日志和逐步数据保存在 results/online_wbc_mpc_50hz；旧低频结果仍保留。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(clean(dict(aggregate=aggregate,means=means,changes=changes)),indent=2))

if __name__=='__main__':main()
