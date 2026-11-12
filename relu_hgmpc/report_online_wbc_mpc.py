"""Matched-policy online MPC/WBC results, including failures and total cost."""
import csv,json,pathlib,collections,math
import numpy as np
from scipy.signal import welch
from scipy.integrate import trapezoid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'results/online_wbc_mpc'
CASES=['normal','double','push'];POLICIES=['fixed10','fixed5','event']
LABELS={'fixed10':'Fixed 10 Hz','fixed5':'Fixed 5 Hz','event':'Feedback-triggered'}
COLORS={'fixed10':'#d27928','fixed5':'#878787','event':'#1673b9'}

def rms(x):return float(np.sqrt(np.mean(np.asarray(x)**2)))
def hf(x,low=10.,mean=False):
    if len(x)<100:return np.nan
    n=min(2048,len(x));f,p=welch(x,fs=1000,nperseg=n,noverlap=n//2,axis=0)
    mask=(f>=low)&(f<=100);powers=trapezoid(p[mask],f[mask],axis=0)
    return float(np.sqrt(np.mean(powers) if mean else np.sum(powers)))
def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,list):return [clean(v) for v in x]
    if isinstance(x,float) and not math.isfinite(x):return None
    return x

def main():
    records=[];summaries={};data={};traces={};repeats={}
    for case in CASES:
        for policy in POLICIES:
            for rep in [1,2]:
                name=f'{case}_{policy}_r{rep}';s=json.loads((OUT/f'{name}.json').read_text());summaries[name]=s
                a=np.genfromtxt(OUT/f'{name}.csv',delimiter=',',names=True);data[name]=a
                tr=np.load(OUT/f'{name}_trace.npz');traces[name]=tr;t=tr['trace'];base=tr['base']
                complete=s['failed'] is None and s['simulated_cycles']==2400
                plans=s['plans'];times=np.array([p['time'] for p in plans]);cost=np.array([p['wall_ms'] for p in plans])
                row=dict(case=case,policy=policy,repeat=rep,complete=int(complete),duration=s['simulated_cycles']*.005,
                    replan_count=len(plans),feedback_trigger_count=sum(p['reason']=='wbc_feedback' for p in plans),
                    total_control_ms=s['total_control_ms'],total_planner_ms=s['total_planner_ms'],
                    total_wbc_reference_ms=s['total_wbc_reference_ms'],
                    avg_feedback_ms=float(a['total_ms'].mean()),wbc_p99_ms=float(np.quantile(a['wbc_reference_ms'],.99)),
                    wbc_max_ms=float(a['wbc_reference_ms'].max()),wbc_over_5ms_count=int(np.sum(a['wbc_reference_ms']>5.)),
                    planner_mean_ms=float(cost.mean()),planner_p99_ms=float(np.quantile(cost,.99)),planner_max_ms=float(cost.max()),
                    planner_over_release_budget=int(np.sum(cost>1000*s['args']['delay'])),
                    period_min_ms=float(np.min(np.diff(times))*1000),period_max_ms=float(np.max(np.diff(times))*1000),
                    clips=s['torque_clips'],missing_contact_ms=int(np.sum(t[:,1:5].sum(axis=1)<4)),
                    min_contacts=int(t[:,1:5].sum(axis=1).min()),
                    mpc_qp_residual=max(p['qp_residual'] for p in plans),
                    mpc_bound_violation=max(p['physical_bound_violation'] for p in plans),
                    nonlinear_dynamics_defect=max(p['nonlinear_dynamics_defect_max'] for p in plans),
                    nonlinear_contact_defect=max(p['nonlinear_contact_defect_max'] for p in plans),
                    wbc_qp_residual=max(float(a[k].max()) for k in ['qp_primal_relative','qp_dual_relative','qp_normal_relative']),
                    wbc_dynamics_residual=float(a['dynamics_residual'].max()),wbc_contact_residual=float(a['contact_residual'].max()),
                    hierarchy_lock_max=float(a['task_lock_max'].max()),reference_handover_max=float(a['reference_continuity'].max()),
                    wbc_backup_solves=int(a['qp_fallbacks'].sum()),
                    arm_joint_tracking_max_rad=float(np.max(np.abs(tr['q'][:,19:]-tr['reference_q'][:,19:]))))
                for task in ['base','ee']:
                    for kind,unit in [('position','mm'),('rotation','deg')]:
                        values=a[f'{task}_{kind}_error_{unit}']
                        row[f'{task}_{kind}_rmse_{unit}']=rms(values) if complete else np.nan
                        row[f'{task}_{kind}_max_{unit}']=float(values.max())
                        row[f'{task}_{kind}_push_window_rmse_{unit}']=rms(values[(a['time']>=6)&(a['time']<7)]) if complete else np.nan
                steady=t[:,0]>=2;quiet=t[:,0]>=8
                for window,mask in [('all',steady),('quiet',quiet)]:
                    for low in [5,10]:
                        row[f'base_omega_{window}_hf{low}_deg_s']=hf(base[mask,17:20],low)*180/np.pi
                        row[f'leg_velocity_{window}_hf{low}_rad_s']=hf(t[mask,111:123],low,True)
                        row[f'arm_velocity_{window}_hf{low}_rad_s']=hf(t[mask,123:129],low,True)
                        row[f'leg_torque_{window}_hf{low}_nm']=hf(t[mask,57:69],low,True)
                        row[f'arm_torque_{window}_hf{low}_nm']=hf(t[mask,69:75],low,True)
                row['leg_actual_acceleration_rms']=rms(np.diff(t[steady,111:123],axis=0)/.001)
                row['arm_actual_acceleration_rms']=rms(np.diff(t[steady,123:129],axis=0)/.001)
                row['leg_actual_torque_rate_rms']=rms(np.diff(t[steady,57:69],axis=0)/.001)
                row['arm_actual_torque_rate_rms']=rms(np.diff(t[steady,69:75],axis=0)/.001)
                records.append(row)
            x=traces[f'{case}_{policy}_r1'];y=traces[f'{case}_{policy}_r2']
            repeats[f'{case}_{policy}']={k:bool(np.array_equal(x[k],y[k])) for k in ['q','v','tau','acc','trace','base','reference_q']}
            repeats[f'{case}_{policy}']['schedule']=summaries[f'{case}_{policy}_r1']['plans'] and bool(np.array_equal(
                [p['time'] for p in summaries[f'{case}_{policy}_r1']['plans']],
                [p['time'] for p in summaries[f'{case}_{policy}_r2']['plans']]))
    means={};comparisons={}
    source_agreement={filename:len({s['source_sha256'][filename] for s in summaries.values()})==1
        for filename in next(iter(summaries.values()))['source_sha256']}
    assert all(source_agreement.values()),source_agreement
    for case in CASES:
        hashes={summaries[f'{case}_{policy}_r{rep}']['target_sha256'] for policy in POLICIES for rep in [1,2]};assert len(hashes)==1
        means[case]={}
        for policy in POLICIES:
            selected=[r for r in records if r['case']==case and r['policy']==policy]
            means[case][policy]={k:float(np.mean([r[k] for r in selected])) for k in selected[0] if k not in ['case','policy','repeat']}
        comparisons[case]={}
        for baseline in ['fixed10','fixed5']:
            b=means[case][baseline];e=means[case]['event']
            comparisons[case][baseline]={k:100*(e[k]/b[k]-1) if b[k]!=0 and e['complete']==1 and b['complete']==1 else None for k in b}
    with (OUT/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    result=dict(means=means,event_change_percent=comparisons,repeat_checks=repeats,source_agreement=source_agreement,
        protocol=json.loads((OUT/'protocol.json').read_text()),verification=json.loads((OUT/'verification.json').read_text()),
        complete_runs=sum(r['complete'] for r in records),total_runs=len(records),
        HF='1 kHz true joint velocities and applied torques; Welch 2048/1024; 5-100 Hz and 10-100 Hz. All window=2-12s includes push transients; quiet window=8-12s excludes push. Joint aggregate is RMS across channels.',
        timing='Online total includes state acquisition, reference propagation, matrices, CPU data movement, all solves, nonlinear node audit, packet creation, WBC, and scheduling. Goal geometry preparation and initialization are separate; no GPU.',
        limitation='Logical asynchronous release at 40 ms is simulated serially; no real OS background-thread deadline guarantee. Nonlinear horizon defects reported, not hidden by small linear-QP residuals.')
    (OUT/'comparison.json').write_text(json.dumps(clean(result),indent=2))
    plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.25,'figure.dpi':140})
    fig,axes=plt.subplots(3,3,figsize=(12,9),sharex='col')
    for j,case in enumerate(CASES):
        for policy in POLICIES:
            a=data[f'{case}_{policy}_r1'];color=COLORS[policy]
            axes[0,j].plot(a['time'],a['base_position_error_mm'],label=LABELS[policy],color=color,lw=1.)
            axes[1,j].plot(a['time'],a['ee_position_error_mm'],color=color,lw=1.)
            times=[p['time'] for p in summaries[f'{case}_{policy}_r1']['plans']]
            axes[2,j].scatter(times,np.full(len(times),POLICIES.index(policy)),marker='|',s=50,color=color)
        axes[0,j].set_title(case.capitalize());axes[2,j].set_xlabel('Time (s)')
        axes[2,j].set_yticks(range(3),[LABELS[p] for p in POLICIES],fontsize=8)
        if case=='push':
            for ax in axes[:,j]:ax.axvspan(6,6.15,color='red',alpha=.10)
    axes[0,0].set_ylabel('Base position error (mm)');axes[1,0].set_ylabel('Tool position error (mm)')
    axes[0,0].legend(fontsize=8);fig.suptitle('Online MPC + identical geometric WBC: tracking and replan times')
    fig.tight_layout();fig.savefig(OUT/'tracking_schedule.png');fig.savefig(OUT/'tracking_schedule.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    fields=[('total_control_ms','Total online compute / 12 s (ms)'),('replan_count','MPC solves / 12 s'),
        ('ee_position_rmse_mm','Tool position RMSE (mm)'),('base_position_rmse_mm','Base position RMSE (mm)'),
        ('arm_velocity_quiet_hf5_rad_s','Arm velocity 5-100 Hz, 8-12 s (rad/s)'),('arm_torque_quiet_hf5_nm','Arm torque 5-100 Hz, 8-12 s (Nm)')]
    for ax,(key,title) in zip(axes.ravel(),fields):
        for i,policy in enumerate(POLICIES):
            vals=[means[c][policy][key] if means[c][policy]['complete']==1 else np.nan for c in CASES]
            ax.bar(np.arange(3)+(i-1)*.24,vals,.24,color=COLORS[policy],label=LABELS[policy])
        ax.set_xticks(range(3),CASES);ax.set_title(title,fontsize=10);ax.set_ylim(bottom=0)
    axes[0,0].legend(fontsize=8);fig.suptitle('Same MPC horizon, solver, constraints and WBC; incomplete runs excluded')
    fig.tight_layout();fig.savefig(OUT/'comparison.png');fig.savefig(OUT/'comparison.pdf');plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(11,7),sharex=True)
    for case in CASES:
        a=data[f'{case}_event_r1'];axes[0].plot(a['time'],a['tracking_residual_score'],label=case)
        axes[1].plot(a['time'],a['joint_tracking_residual_score'],label=case)
        axes[2].plot(a['time'],np.minimum(np.minimum(a['torque_margin'],a['friction_margin']),a['normal_margin']),label=case)
    axes[0].axhline(1,color='black',ls=':');axes[1].axhline(1,color='black',ls=':');axes[2].axhline(.08,color='black',ls=':')
    axes[0].set_ylabel('Task tracking score');axes[1].set_ylabel('Joint tracking score');axes[2].set_ylabel('Minimum physical margin')
    axes[0].legend();axes[2].set_xlabel('Time (s)');fig.suptitle('Causal WBC signals used by the event scheduler (unfiltered)')
    fig.tight_layout();fig.savefig(OUT/'trigger_feedback.png');plt.close(fig)
    lines=['# 在线 MPC＋几何 WBC 重规划对照','',f"已完成 {result['complete_runs']}/{len(records)} 次完整运行。每次目标时长 12 s；不完整运行不参与完整时域误差与计算节省比例。",'',
        '## 当前结论','',
        '在线 MPC、连续几何参考和 WBC 的闭环已接通。当前反馈触发方案可以减少重规划，但尚未实现同等精度下的加速，更没有表现出一致的实际关节减振优势。相比固定 10 Hz，求解次数从每 12 s 的 120 次降至 62–67 次，规划计算减少约 43%–48%，总在线计算减少约 8%–10%。代价在双幅工况尤其明显：末端位置 RMSE 从约 2.57 mm 增至 3.69 mm，底盘约 5.57 mm 增至 8.12 mm。',
        '固定 5 Hz 消融表明，大部分计算节省来自降频。触发方案相对固定 5 Hz 只带来小幅任务跟踪改善，实际速度与力矩高频指标有升有降，不能宣称优于固定低频调度。单次约 1% 的总耗时差异不应过度解释为稳定的性能优势。',
        '当前触发信号主要描述 WBC 能否执行已下发参考。若旧 MPC 参考本身已经偏离全局目标，WBC 仍可能很好地跟随该参考，因此仅凭执行残差和约束余量会漏检规划质量下降。后续应把全局目标误差和模型预测失配纳入触发判据，并验证可接受误差界；本轮尚未实现这些扩展。','',
        '## 共同实现与比较口径','',
        '- MPC 每次在线重新读取 MuJoCo 真值，更新 Pinocchio 全身动力学、接触、科氏项及执行器约束，求解 180 变量 QP。N=10、步长 40 ms、预测时域 0.4 s。三策略使用相同目标、约束、精度与求解器。',
        '- 本次是单次线性化的轨迹生成 MPC，不是收敛到非线性最优解的 SQP-NMPC，也不是原层级 GMPC 控制器原封不动改频率。SE(3) 任务误差在当前位姿作一阶展开，动力学系数在每次重规划更新并在该时域冻结。',
        '- MPC 初值取交接时刻已提交参考的 q/v/a，以免实测跟踪滞后引起参考重置尖峰；实测状态用于模型、约束和已施加力矩更新，WBC 闭环处理实际跟踪误差。支撑腿执行参考沿非线性球足无滑动运动学重建。',
        '- 200 Hz WBC，1 kHz MuJoCo 与实际执行器记录。统一层级：动力学/接触/执行边界 > 底盘 > 末端 > 姿态与力矩平滑。规划第一步力矩变化范围按 40 ms 后交接计算，实际 WBC 仍按 5 ms 执行周期限制变化率。',
        '- 三策略共同加入机械臂参考偏差约束：a-a_ref+16(v-v_ref)+64(q-q_ref) 在 ±5.12 rad/s² 内。它限制任务空间弱可观测方向的关节漂移；不能宣称在模型误差下已严格证明偏差不变集。',
        '- 固定 10 Hz、固定 5 Hz、反馈触发；每个工况两次串行重复，第二次倒序。普通、双幅、外力扰动：6.00–6.15 s 在底盘质心施加世界 x 正向 120 N，不提前告知控制器。','',
        '## 触发规则','',
        '使用上一控制周期 WBC 输出，避免读取未来结果。取任务加速度未实现量、底盘/末端跟踪误差、关节参考偏差的归一化最大值，经系数 0.2 的 EWMA 滤波后超过 1，或力矩/摩擦/法向力余量小于 8%、关节线性化范围余量小于 20%，连续三个控制周期后触发。最短间隔 50 ms，最长 200 ms。',
        '归一化尺度：线加速度 0.5 m/s²、角加速度 2 rad/s²、底盘 2 mm/0.5°、末端 5 mm/2°、关节 0.03 rad。这些为工程阈值，经过调试后冻结用于本轮；本轮结果不是未见工况验证，也不是阈值最优性的证明。','',
        '## 结果','']
    for case in CASES:
        lines += [f'### {case}','']
        for policy in POLICIES:
            m=means[case][policy]
            if m['complete']!=1:
                fail=summaries[f'{case}_{policy}_r1']['failed'];lines.append(f'- {LABELS[policy]}：未完成，失败信息 {fail}。');continue
            lines.append(f"- {LABELS[policy]}：MPC {m['replan_count']:.0f} 次，其中反馈触发 {m['feedback_trigger_count']:.0f} 次；总在线计算 {m['total_control_ms']/1000:.4f} s，规划 {m['total_planner_ms']/1000:.4f} s；底盘位置/姿态 RMSE {m['base_position_rmse_mm']:.5g} mm / {m['base_rotation_rmse_deg']:.5g}°；末端 {m['ee_position_rmse_mm']:.5g} mm / {m['ee_rotation_rmse_deg']:.5g}°。")
        for baseline in ['fixed10','fixed5']:
            d=comparisons[case][baseline]
            if d['total_control_ms'] is None:continue
            lines.append(f"- 触发方案相对 {LABELS[baseline]}：总计算 {d['total_control_ms']:+.3g}%，规划计算 {d['total_planner_ms']:+.3g}%，底盘位置 RMSE {d['base_position_rmse_mm']:+.3g}%，末端位置 RMSE {d['ee_position_rmse_mm']:+.3g}%；安静时段腿/臂实际速度高频 RMS {d['leg_velocity_quiet_hf5_rad_s']:+.3g}% / {d['arm_velocity_quiet_hf5_rad_s']:+.3g}%，腿/臂施加力矩高频 RMS {d['leg_torque_quiet_hf5_nm']:+.3g}% / {d['arm_torque_quiet_hf5_nm']:+.3g}%。")
        lines.append('')
    lines += ['## 残差、计时与限制','']
    for key in ['mpc_qp_residual','mpc_bound_violation','wbc_qp_residual','wbc_dynamics_residual','wbc_contact_residual','hierarchy_lock_max','reference_handover_max','nonlinear_dynamics_defect','nonlinear_contact_defect']:
        lines.append(f'- {key} 最大值：{max(r[key] for r in records):.6g}')
    lines += ['',f"两次重复的状态、实际执行器记录与触发时刻逐元素一致：{all(all(v.values()) for v in repeats.values())}。确定性重复不等于统计显著性。",
        f"所有运行合计失去四足接触的记录时长 {sum(r['missing_contact_ms'] for r in records)} ms，力矩裁剪 {sum(r['clips'] for r in records)} 次。规划超过 40 ms 交接预算共 {sum(r['planner_over_release_budget'] for r in records)} 次。WBC＋参考采样超过 5 ms 共 {sum(r['wbc_over_5ms_count'] for r in records)} 次，最大 {max(r['wbc_max_ms'] for r in records):.4g} ms。WBC 备用求解共 {sum(r['wbc_backup_solves'] for r in records)} 次。",'',
        'WBC 主动集求解遇到数值退化时，显式记录并用 SLSQP 求解同一个子问题，通过非负活跃乘子重建后，仍须通过同一未加数值正则的 KKT 检查、原物理边界检查和层级保持检查；不会把求解器返回成功本身视为验收。备用求解时间已计入总计算。',
        '总计算包括状态读取、参考前推、矩阵构造、CPU 数据传递、求解、非线性节点复核、轨迹生成、WBC 以及触发检测；不只比较 QP 求解时间。目标曲线几何预计算与初始化单独记录，不是在线 MPC 计算的替代。',
        '规划完成后按统一 40 ms 延迟交接，等待期间 WBC 继续执行已提交轨迹。这里是串行运行的逻辑并发模拟，尚未部署实际操作系统的异步线程，不能宣称已证明硬实时。WBC 瞬时约束检查与 MPC 线性 QP 残差均有记录；预测时域的非线性动力学/接触缺陷单独报告，不能用小 QP 残差掩盖模型近似。',
        'nonlinear_dynamics_defect 是拼接广义动力学残差的最大分量，包含 N 与 N·m，仅作为近似模型数值诊断，不是具有统一单位的物理范数；nonlinear_contact_defect 的单位为 m/s²。',
        '高频指标来自 1 kHz 实际关节速度和施加力矩。报告同时保留 5–100 Hz、10–100 Hz、实际速度差分加速度 RMS 与力矩变化率；安静窗口 8–12 s 排除了外力脉冲，完整 2–12 s 窗口含扰动响应，不能全部解释成周期抖振。','',
        '## 调试发现与复现','',
        '直接在交接时重置到实测 q/v，会把执行滞后变成参考加速度尖峰；只有位姿和速度连续并不足够。另一问题是末端误差较小不能排除机械臂关节偏离参考。仅增加重规划次数未消除后者，需要三策略共同的执行偏差约束。相关失败结果保存在 diagnostics，这些修复不能计作触发调度算法单独带来的收益。',
        '源代码：/home/ganhe/project/b2z1_mpc/relu_hgmpc。运行 env.sh 后执行 test_online_wbc_mpc.py、benchmark_online_wbc_mpc.py、report_online_wbc_mpc.py。原始数据、协议、源码和目标轨迹哈希：results/online_wbc_mpc。原 native.cpp 和 geometric_wbc.py 已备份至 relu_hgmpc/backup_before_online。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(clean(dict(complete=result['complete_runs'],means=means,changes=comparisons)),indent=2))

if __name__=='__main__':main()
