"""Matched full-duration reporting; incomplete runs never enter rankings."""
import csv,json,pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from report_online_wbc_mpc import rms,hf,clean
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'results/shared_geometric_wbc'
CASES=['normal','double','push'];MODES=['allocation','mpc'];CLOCKS=['fixed','wall']
LABELS={'allocation':'Prioritized instantaneous allocation','mpc':'Hierarchical geometric MPC'}
COLORS={'allocation':'#dd872b','mpc':'#237eae'}

def main():
    records=[];summaries={};data={};traces={}
    for clock in CLOCKS:
        for case in CASES:
            for mode in MODES:
                for rep in [1,2]:
                    name=f'{clock}_{case}_{mode}_r{rep}';s=json.loads((OUT/f'{name}.json').read_text());summaries[name]=s
                    a=np.atleast_1d(np.genfromtxt(OUT/f'{name}.csv',delimiter=',',names=True));data[name]=a
                    with np.load(OUT/f'{name}_trace.npz') as z:trace=z['trace'];base=z['base']
                    if rep==1:traces[name]=trace
                    complete=s['failed'] is None and s['simulated_cycles']==2400
                    plans=s['plans'];times=np.array([p['worker_wall_ms'] for p in plans])
                    contact=trace[:,1:5]>0;speed=np.linalg.norm(trace[:,21:33].reshape(-1,4,3),axis=2)
                    r=dict(clock=clock,case=case,mode=mode,repeat=rep,complete=int(complete),duration=s['simulated_cycles']*.005,
                        total_cpu_s=(s['main_cpu_ms']+s['worker_cpu_ms'])/1000 if complete else np.nan,
                        main_cpu_s=s['main_cpu_ms']/1000 if complete else np.nan,worker_cpu_s=s['worker_cpu_ms']/1000 if complete else np.nan,
                        response_mean_ms=float(np.mean(a['control_response_ms'])),response_p99_ms=float(np.quantile(a['control_response_ms'],.99)),
                        response_max_ms=float(np.max(a['control_response_ms'])),wbc_deadline_misses=int(np.sum(a['wbc_deadline_miss'])),
                        planner_count=len(plans),installed_plans=s['installed_plans'],skipped_slots=s['skipped_requests'],
                        actual_plan_hz=s['installed_plans']/(s['simulated_cycles']*.005),planner_deadline_misses=sum(p['deadline_miss'] for p in plans),
                        planner_wall_mean_ms=float(np.mean(times)),planner_wall_p99_ms=float(np.quantile(times,.99)),planner_wall_max_ms=float(np.max(times)),
                        planner_cpu_mean_ms=float(np.mean([p['worker_cpu_ms'] for p in plans])),
                        upper_construction_mean_ms=float(np.mean([p['construction_ms'] for p in plans])),
                        upper_solve_mean_ms=float(np.mean([p['solve_ms'] for p in plans])),
                        qp_kkt_max=max(p['qp_residual'] for p in plans),upper_task_lock_max=max(p['hierarchy_lock'] for p in plans),
                        upper_bound_max=max(p['physical_bound_violation'] for p in plans),
                        wbc_kkt_max=max(float(np.max(a[k])) for k in ['qp_primal_relative','qp_dual_relative','qp_normal_relative']),
                        wbc_task_lock_max=float(np.max(a['task_lock_max'])),wbc_bound_max=float(np.max(a['constraint_violation'])),
                        contact_equation_max=float(np.max(a['contact_residual'])),dynamics_equation_max=float(np.max(a['dynamics_residual'])),
                        nonlinear_prediction_contact_max=max(p['nonlinear_contact_defect_max'] for p in plans),
                        nonlinear_prediction_dynamics_max=max(p['nonlinear_dynamics_defect_max'] for p in plans),
                        reference_continuity_max=float(np.max(a['reference_continuity'])),
                        missing_contact_ms=int(np.sum(contact.sum(axis=1)<4)),min_contacts=int(np.min(contact.sum(axis=1))),
                        contact_material_speed_rms_mm_s=rms(speed[contact])*1000,clips=int(np.sum(a['clips'])),
                        osqp_calls=sum(p['hierarchy_osqp_calls'] for p in plans),active_face_fast=sum(p['hierarchy_active_face_fast'] for p in plans),
                        exact_ls_levels=sum(p['hierarchy_fast_levels'] for p in plans),polish_fallbacks=sum(p['hierarchy_fallbacks'] for p in plans))
                    for frame in ['base','ee']:
                        for kind,unit in [('position','mm'),('rotation','deg')]:
                            x=a[f'{frame}_{kind}_error_{unit}'];key=f'{frame}_{kind}_rmse_{unit}'
                            r[key]=rms(x) if complete else np.nan
                            mask=(a['time']>=6)&(a['time']<7)
                            r[f'{frame}_{kind}_push_window_rmse_{unit}']=rms(x[mask]) if np.sum(mask)==200 else np.nan
                    for window,mask in [('full',trace[:,0]>=2),('late',trace[:,0]>=8)]:
                        for low in [5,10]:
                            for label,x in [('leg_velocity',trace[:,111:123]),('arm_velocity',trace[:,123:129]),
                                            ('leg_torque',trace[:,57:69]),('arm_torque',trace[:,69:75])]:
                                r[f'{label}_{window}_hf{low}']=hf(x[mask],low,True) if complete else np.nan
                            r[f'base_omega_{window}_hf{low}_deg_s']=hf(base[mask,17:20],low)*180/np.pi if complete else np.nan
                    records.append(r)
    for case in CASES:
        assert len({s['target_sha256'] for s in summaries.values() if s['args']['case']==case})==1
    sources={k:len({s['source_sha256'][k] for s in summaries.values()})==1 for k in next(iter(summaries.values()))['source_sha256']}
    assert all(sources.values()),sources
    for name,s in summaries.items():
        if s['args']['clock']=='fixed' and s['failed'] is None:
            assert len(s['plans'])==600 and s['skipped_requests']==0
            assert max(p['install_late_ms'] or 0 for p in s['plans'])<1e-8
    with (OUT/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    means={};changes={};ranges={}
    for clock in CLOCKS:
        means[clock]={};changes[clock]={};ranges[clock]={}
        for case in CASES:
            means[clock][case]={};ranges[clock][case]={}
            for mode in MODES:
                rr=[r for r in records if r['clock']==clock and r['case']==case and r['mode']==mode]
                fields=[k for k in rr[0] if k not in ['clock','case','mode','repeat']]
                means[clock][case][mode]={k:float(np.mean([r[k] for r in rr])) for k in fields}
                ranges[clock][case][mode]={k:[float(min(r[k] for r in rr)),float(max(r[k] for r in rr))] for k in fields}
            b,m=[means[clock][case][x] for x in MODES]
            changes[clock][case]={k:100*(m[k]/b[k]-1) if b[k] and b['complete']==1 and m['complete']==1 else None for k in b}
    result=clean(dict(means=means,ranges=ranges,mpc_change_percent=changes,source_agreement=sources,
                      protocol=json.loads((OUT/'protocol.json').read_text())))
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2))
    plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.22,'figure.dpi':150})
    fields=[('ee_position_rmse_mm','EE position RMSE (mm)'),('ee_rotation_rmse_deg','EE rotation RMSE (deg)'),
            ('total_cpu_s','Controller CPU / 12 s (s)'),('base_position_rmse_mm','Base position RMSE (mm)'),
            ('arm_velocity_full_hf5','Actual arm velocity 5-100 Hz (rad/s)'),('arm_torque_full_hf5','Applied arm torque 5-100 Hz (Nm)')]
    for clock in CLOCKS:
        fig,axes=plt.subplots(2,3,figsize=(12,6.5))
        for ax,(field,title) in zip(axes.ravel(),fields):
            for i,mode in enumerate(MODES):
                y=[means[clock][c][mode][field] if means[clock][c][mode]['complete']==1 else np.nan for c in CASES]
                ax.bar(np.arange(3)+(i-.5)*.32,y,.32,color=COLORS[mode],label=LABELS[mode])
                for j,c in enumerate(CASES):
                    if means[clock][c][mode]['complete']!=1:ax.text(j+(i-.5)*.32,.03,'incomplete',rotation=90,transform=ax.get_xaxis_transform())
            ax.set_xticks(range(3),['Normal','2x motion','Push']);ax.set_title(title);ax.set_ylim(bottom=0)
        fig.legend([LABELS[m] for m in MODES],loc='lower center',ncol=2)
        title='Identical logical 50 Hz releases (offline)' if clock=='fixed' else 'Actual threaded schedule (50 Hz requested; late plans counted)'
        fig.suptitle(title+' | Shared geometric WBC, goals and task hierarchy')
        fig.tight_layout(rect=(0,.055,1,.95));fig.savefig(OUT/f'comparison_{clock}.png');fig.savefig(OUT/f'comparison_{clock}.pdf');plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(12,8),sharex='col')
    for j,case in enumerate(CASES):
        for mode in MODES:
            a=data[f'fixed_{case}_{mode}_r1']
            for i,field in enumerate(['base_position_error_mm','ee_position_error_mm','ee_rotation_error_deg']):
                axes[i,j].plot(a['time'],a[field],lw=1,color=COLORS[mode],label=LABELS[mode])
            failure=summaries[f'fixed_{case}_{mode}_r1']['failed']
            if failure is not None:
                for ax in axes[:,j]:ax.axvline(failure['time'],color=COLORS[mode],ls=':',lw=1)
                axes[0,j].text(failure['time']+.25,.78,f"{mode} stopped\n{failure['time']:.3f} s",
                    transform=axes[0,j].get_xaxis_transform(),fontsize=8,color=COLORS[mode])
        axes[0,j].set_title(case);axes[-1,j].set_xlabel('Time (s)')
        if case=='push':
            for ax in axes[:,j]:ax.axvspan(6,6.15,color='red',alpha=.1)
    for ax,label in zip(axes[:,0],['Base position (mm)','EE position (mm)','EE rotation (deg)']):ax.set_ylabel(label)
    axes[0,0].legend(fontsize=7);fig.suptitle('Same WBC and exact 50 Hz reference release: tracking errors')
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(OUT/'tracking_fixed.png');fig.savefig(OUT/'tracking_fixed.pdf');plt.close(fig)
    lines=['# 共用几何 WBC 的优先级运动分配与层级 GMPC 对照','',
           '## Material Passport','',
           'mode: run; status: measured; source: local MuJoCo/Pinocchio simulation; raw data: matching CSV/NPZ/JSON files; scope: controlled engineering experiment, not a hardware or original-paper replication claim.','',
           f"共 {len(records)} 次正式运行，{sum(r['complete'] for r in records)} 次完成 12 s。三工况、两种上层、固定逻辑频率与实际墙钟两种调度、各两次重复。",'',
           '主要结论：同一几何 WBC 和固定逻辑 50 Hz 下，MPC 在普通／双幅轨迹上提高底盘与末端跟踪精度，机械臂实际速度的高频量下降；力矩及腿部高频量并非全面下降，总计算 CPU 明显增加。实际多线程没有在所有工况稳定达到 50 Hz，扰动工况仍存在失败。不能据此宣称已经实现“跟踪更好、执行器全面更平滑、计算更快”三者兼得。','',
           '## 实际改动与公平性','',
           '两组共用 CommonWBC → GeometricWBC、同一完整目标、SE(3) 误差、同一增益、同一物理边界、球足无滑动滚动接触、C2 拼接和 advance_reference 关节伺服接口。每组独立状态，源文件和目标哈希一致。结构测试验证同一状态与参考产生逐项相同的 WBC 输出，以及上层首个 40 ms 节点的全部物理约束矩阵／上下界一致。',
           '共同优先级为：物理约束 > 底盘位姿 > 末端位姿 > 关节姿态／力和力矩平滑。上层和 WBC 通过任务输出锁定实现词典序优先级，不把大权重加权和称为严格层级。',
           '运动分配是本次适配基线：当前目标的优先级微分逆运动学，通过接触消元后的力矩坐标表达下一命令区间的速度，并检查共同物理边界。其速度反馈增益为 10；只读取当前释放时刻的目标。它没有保留原论文“机械臂优先启用、底盘开关、可操作度第三任务”的原样结构。因此本轮结果不能称为直接战胜原论文算法。原 prioritized.py、原 WBC 以及上轮结果均保留。',
           'MPC 使用相同几何任务及物理边界，10×40 ms 的未来目标和冻结动力学的一阶预测，严格分层求解。两组上层目标函数分别是当前速度分配与预测位姿／速度跟踪；这是方法差异，不是完全相同的 QP。上层对 40 ms 命令节点检查边界，WBC 每 5 ms 在实际状态重新检查；节点可行不保证插值全段或实际非线性系统严格可行。',
           '固定频率组在离线仿真中等待规划包，于准确的 20 ms 释放时刻应用，确实每秒 50 次；它用于控制效果比较，不能证明实时性。墙钟组使用实际独立线程、20 ms 请求周期、200 Hz WBC，迟到包按实际完成时间拼接，忙时跳过请求。报告真实完成率而不把请求频率当成实际更新率。',
           '两组都移除了额外硬性的机械臂参考误差包络；物理约束仍是硬约束。两组都使用相同位置／速度关节伺服，避免未观测方向漂移。此设置在正式批次前冻结。','',
           '## 固定逻辑频率：主要控制效果','']
    for case in CASES:
        lines += [f'### {case}','']
        for mode in MODES:
            r=means['fixed'][case][mode]
            if r['complete']!=1:
                lines.append(f'- {mode} 未全部完成；不作完整时域排名。');continue
            lines.append(f"- {mode}：底盘位置／姿态 RMSE {r['base_position_rmse_mm']:.5g} mm／{r['base_rotation_rmse_deg']:.5g}°；末端 {r['ee_position_rmse_mm']:.5g} mm／{r['ee_rotation_rmse_deg']:.5g}°；12 s 总控制 CPU {r['total_cpu_s']:.5g} s，其中主线程 {r['main_cpu_s']:.5g} s、上层 {r['worker_cpu_s']:.5g} s。")
            lines.append(f"  2–12 s 实际腿／臂速度 5–100 Hz RMS {r['leg_velocity_full_hf5']:.5g}／{r['arm_velocity_full_hf5']:.5g} rad/s；实际腿／臂力矩 {r['leg_torque_full_hf5']:.5g}／{r['arm_torque_full_hf5']:.5g} N·m。")
        d=changes['fixed'][case]
        if d['total_cpu_s'] is not None:
            lines.append(f"- MPC 相对分配：末端位置 {d['ee_position_rmse_mm']:+.3g}%，姿态 {d['ee_rotation_rmse_deg']:+.3g}%；腿／臂速度高频量 {d['leg_velocity_full_hf5']:+.3g}%／{d['arm_velocity_full_hf5']:+.3g}%；腿／臂力矩高频量 {d['leg_torque_full_hf5']:+.3g}%／{d['arm_torque_full_hf5']:+.3g}%；总 CPU {d['total_cpu_s']:+.3g}%。")
        lines.append('')
    lines+=['## 实际多线程时限','']
    for case in CASES:
        for mode in MODES:
            r=means['wall'][case][mode]
            lines.append(f"- {case}/{mode}：完成率 {r['complete']:.0%}，实际安装计划频率 {r['actual_plan_hz']:.3g} Hz（含起始 20 ms 等待）；每次运行规划迟到平均 {r['planner_deadline_misses']:.1f} 次、跳过 {r['skipped_slots']:.1f} 槽；单次规划墙钟均值／P99／最大值的两次平均 {r['planner_wall_mean_ms']:.4g}/{r['planner_wall_p99_ms']:.4g}/{r['planner_wall_max_ms']:.4g} ms；WBC 响应超 5 ms 平均 {r['wbc_deadline_misses']:.1f}/{r['duration']/.005:.0f} 次。")
            if r['complete']==1:lines.append(f"  实际调度末端误差 {r['ee_position_rmse_mm']:.5g} mm／{r['ee_rotation_rmse_deg']:.5g}°，12 s 控制 CPU {r['total_cpu_s']:.5g} s。")
    lines+=['','## 扰动失败诊断','']
    for name,s in summaries.items():
        if s['failed'] is not None:lines.append(f"- 正式运行 {name} 在 {s['failed']['time']:.4g} s 停止：{s['failed']['error']}。")
    for mode in MODES:
        path=OUT/f'diagnosis_{mode}.json'
        if path.exists():
            diagnosis=json.loads(path.read_text())
            for item in diagnosis:
                lines.append(f"- {mode} 固定频率独立重放：{item['layer']} 层、{item.get('level','n/a')} 级，{item['variables']} 变量；{item['error']}。独立线性可行性检查：{item['feasibility_status']}。")
    lines+=['失败不等同于任务本身或原论文算法不可行；需要区分当前线性化、承诺的参考初值与边界、求解器和恢复策略。本原型遇到未通过验收的控制问题会停止实验，没有安全回退控制器。失败时段不纳入全程误差／耗时排名。','', '## 验收与解释边界','']
    for clock in CLOCKS:
        for mode in MODES:
            rr=[r for r in records if r['clock']==clock and r['mode']==mode]
            lines.append(f"- {clock}/{mode}：累计失去四足接触 {sum(r['missing_contact_ms'] for r in rr)} ms，执行力矩裁剪 {sum(r['clips'] for r in rr)} 次；上层／WBC KKT 最大 {max(r['qp_kkt_max'] for r in rr):.4g}/{max(r['wbc_kkt_max'] for r in rr):.4g}；任务锁定最大 {max(r['upper_task_lock_max'] for r in rr):.4g}/{max(r['wbc_task_lock_max'] for r in rr):.4g}；上层／WBC 原单位边界残差 {max(r['upper_bound_max'] for r in rr):.4g}/{max(r['wbc_bound_max'] for r in rr):.4g}；WBC 接触方程残差 {max(r['contact_equation_max'] for r in rr):.4g}。")
    lines += ['',
        '这些数据用于判断本实现的权衡，不预设 MPC 必然减少所有关节抖振或提高所有跟踪指标。频带能量来自 1 kHz 实际关节速度和施加力矩；按关节功率平均再开方，保留 5–100 Hz、10–100 Hz 和后段 8–12 s 敏感性指标。它包含参考插值、扰动与非线性瞬态，不能直接当作持续极限环抖振的证明。',
        '几何误差、执行器平滑和 WBC 均共同使用，因此本轮不能估计 Lie 群方法相对欧拉方法的独立贡献，也不能说明一般 MPC 对所有优先级运动分配都占优。',
        '总计算耗时统计主线程与规划线程实际 CPU 时间之和，包含状态读取、参考生成、数据复制、模型与矩阵构造、求解、校验和 WBC；不把并行墙钟相加。共同目标预计算及初始化单独记录，MuJoCo 积分、其中的关节伺服仿真和结果日志不计入在线控制 CPU。',
        'OSQP 用于剩余空间内的正定平滑 QP，配合原约束验收的活跃面快速解；高优先级半正定问题使用活跃集 QP 与原始目标校正。准确名称是混合层级求解器，不能称为所有层都由 OSQP 求解，也不是同一个 QP 的纯求解器加速测试。',
        '大轨迹的约束激活会增加求解时间。本轮没有硬实时证书，也没有把真实墙钟超时映射为实机保持旧力矩的掉帧模型；MuJoCo 每个控制调用仍推进五个 1 ms 步。两次重复仅检查数值重复性与调度波动，不做显著性结论。',
        '诊断试跑及修复前批次保留，未混入正式指标。完整目标与源文件哈希在各次 JSON；固定频率的逐时序重复差异另见 repeatability.json。','',
        '## 文件与复现','',
        'WSL 工程：/home/ganhe/project/b2z1_mpc。激活 relu_hgmpc/env.sh，执行 test_shared_wbc.py、benchmark_shared_wbc.py、report_shared_wbc.py。正式结果在 results/shared_geometric_wbc。',
        'comparison.json 包含两次均值、范围和相对变化；metrics.csv 保留逐运行结果；comparison_fixed/wall 与 tracking_fixed 提供 PNG/PDF。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    repeats={}
    for case in CASES:
        for mode in MODES:
            names=[f'fixed_{case}_{mode}_r{rep}' for rep in [1,2]]
            with np.load(OUT/f'{names[0]}_trace.npz') as a,np.load(OUT/f'{names[1]}_trace.npz') as b:
                repeats[f'{case}_{mode}']={k:float(np.max(np.abs(a[k]-b[k]))) if a[k].shape==b[k].shape else 'shape mismatch' for k in ['q','v','tau','acc','trace']}
    (OUT/'repeatability.json').write_text(json.dumps(repeats,indent=2))
    print(json.dumps(clean(dict(complete=sum(r['complete'] for r in records),means=means,changes=changes,repeatability=repeats)),indent=2))

if __name__=='__main__':main()
