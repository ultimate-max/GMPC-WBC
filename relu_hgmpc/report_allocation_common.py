import pathlib,json,csv
import numpy as np
import pinocchio as pin
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from report_online_wbc_mpc import rms,hf,clean
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'results/allocation_vs_online_mpc'
CASES=['normal','double','push'];MODES=['allocation_legacy','allocation_rolling','mpc']
LABELS=['Allocation+WBC / hysteresis','Allocation+WBC / rolling correction','Online MPC+geometric WBC']
COLORS=['#888888','#df8b27','#1977bd']

def rotation(q):return pin.Quaternion(q[6],*q[3:6]).matrix()

def main():
    records=[];summaries={};data={};basedata={}
    for case in CASES:
        for mode in MODES:
            for rep in [1,2]:
                name=f'{case}_{mode}_r{rep}';s=json.loads((OUT/f'{name}.json').read_text());summaries[name]=s
                a=np.atleast_1d(np.genfromtxt(OUT/f'{name}.csv',delimiter=',',names=True));data[name]=a
                with np.load(OUT/f'{name}_trace.npz') as z:
                    trace=z['trace'];base=z['base'];q=z['q']
                if rep==1:basedata[name]=base
                complete=s['failed'] is None and s['simulated_cycles']==2400
                home=base[0,1:8];offset=np.linalg.norm(base[:,1:4]-home[:3],axis=1)*1000
                R0=rotation(home);angles=np.array([np.linalg.norm(pin.log3(R0.T@rotation(x))) for x in base[:,1:8]])*180/np.pi
                contact=trace[:,1:5]>0;speed=np.linalg.norm(trace[:,21:33].reshape(-1,4,3),axis=2)
                rec=dict(case=case,mode=mode,repeat=rep,complete=int(complete),duration=s['simulated_cycles']*.005,
                    total_cpu_ms=s['main_cpu_ms']+s['worker_cpu_ms'] if mode=='mpc' else s['total_cpu_ms'],
                    response_mean_ms=float(a['control_response_ms'].mean()),response_p99_ms=float(np.quantile(a['control_response_ms'],.99)),
                    response_max_ms=float(a['control_response_ms'].max()),wbc_deadline_misses=int(a['wbc_deadline_miss'].sum()),
                    base_offset_rms_mm=rms(offset) if complete else np.nan,base_rotation_offset_rms_deg=rms(angles) if complete else np.nan,
                    base_height_offset_rms_mm=rms((base[:,3]-home[2])*1000) if complete else np.nan,
                    min_contacts=int(contact.sum(axis=1).min()),missing_contact_ms=int(np.sum(contact.sum(axis=1)<4)),
                    contact_speed_rms_mm_s=rms(speed[contact])*1000,clips=int(a['clips'].sum()))
                for kind,unit in [('position','mm'),('rotation','deg')]:
                    x=a[f'ee_{kind}_error_{unit}'];rec[f'ee_{kind}_rmse_{unit}']=rms(x) if complete else np.nan
                    early=a['time']<8.;rec[f'ee_{kind}_first8_rmse_{unit}']=rms(x[early]) if np.sum(early)==1600 else np.nan
                    mask=(a['time']>=6)&(a['time']<7);rec[f'ee_{kind}_push_window_rmse_{unit}']=rms(x[mask]) if np.sum(mask)==200 else np.nan
                if mode=='mpc':
                    rec.update(switches=0,internal_base_position_rmse_mm=np.nan,internal_base_rotation_rmse_deg=np.nan,
                        prescribed_base_position_rmse_mm=rms(a['base_position_error_mm']),prescribed_base_rotation_rmse_deg=rms(a['base_rotation_error_deg']),
                        final_contact_acc_residual=float(a['contact_residual'].max()),float_dynamics_residual=float(a['dynamics_residual'].max()),
                        final_qp_primal=float(a['constraint_violation'].max()),
                        planner_count=len(s['plans']),skipped_planner_requests=s['skipped_requests'],planner_deadline_misses=sum(p['deadline_miss'] for p in s['plans']))
                else:
                    rec.update(switches=int(np.sum(np.diff(a['base_active'])!=0)),
                        internal_base_position_rmse_mm=rms(a['internal_base_position_error_mm']),internal_base_rotation_rmse_deg=rms(a['internal_base_rotation_error_deg']),
                        prescribed_base_position_rmse_mm=np.nan,prescribed_base_rotation_rmse_deg=np.nan,
                        final_contact_acc_residual=float(a['wbc_contact_acc_residual'].max()),float_dynamics_residual=float(a['wbc_float_dynamics_residual'].max()),
                        final_qp_primal=float(a['primal'].max()),planner_count=0,skipped_planner_requests=0,planner_deadline_misses=0)
                for window,mask in [('all',trace[:,0]>=2),('quiet',trace[:,0]>=8)]:
                    for low in [5,10]:
                        rec[f'base_omega_{window}_hf{low}_deg_s']=hf(base[mask,17:20],low)*180/np.pi
                        for label,x in [('leg_velocity',trace[:,111:123]),('arm_velocity',trace[:,123:129]),('leg_torque',trace[:,57:69]),('arm_torque',trace[:,69:75])]:
                            rec[f'{label}_{window}_hf{low}']=hf(x[mask],low,True)
                records.append(rec)
    shared_sources={key:len({s['source_sha256'][key] for s in summaries.values()})==1 for key in ['native.cpp','geometric_wbc.py']}
    assert all(shared_sources.values()),shared_sources
    means={};changes={};aggregate={}
    for case in CASES:
        assert len({summaries[f'{case}_{m}_r{rep}']['target_sha256'] for m in MODES for rep in [1,2]})==1
        means[case]={}
        for mode in MODES:
            rr=[r for r in records if r['case']==case and r['mode']==mode]
            means[case][mode]={k:float(np.mean([x[k] for x in rr])) for k in rr[0] if k not in ['case','mode','repeat']}
        b=means[case]['allocation_rolling'];m=means[case]['mpc']
        changes[case]={k:100*(m[k]/b[k]-1) if b[k]!=0 and b['complete']==1 and m['complete']==1 else None for k in b}
    for mode in MODES:
        rr=[r for r in records if r['mode']==mode]
        aggregate[mode]={k:sum(r[k] for r in rr) for k in ['complete','missing_contact_ms','clips','wbc_deadline_misses','skipped_planner_requests','planner_deadline_misses']}
        aggregate[mode]['cycles']=sum(round(r['duration']/.005) for r in rr)
    with (OUT/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    diagnosis=json.loads((OUT/'boundary_diagnosis.json').read_text()) if (OUT/'boundary_diagnosis.json').exists() else None
    result=dict(means=means,mpc_vs_rolling_change_percent=changes,aggregate=aggregate,shared_source_agreement=shared_sources,boundary_diagnosis=diagnosis,
        protocol=json.loads((OUT/'protocol.json').read_text()),
        limitations='Shared tool target, not identical base/posture objectives, hierarchy, constraints or low-level reference generation. Allocation200Hz vs MPC50Hz; both WBC200Hz. No geometry-only or same-QP acceleration claim.')
    (OUT/'comparison.json').write_text(json.dumps(clean(result),indent=2))
    plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.25,'figure.dpi':140})
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    fields=[('ee_position_rmse_mm','Tool position RMSE (mm)'),('ee_rotation_rmse_deg','Tool rotation RMSE (deg)'),
        ('total_cpu_ms','Controller CPU / 12 s (ms)'),('base_omega_quiet_hf10_deg_s','Base omega 10-100 Hz (deg/s)'),
        ('leg_velocity_quiet_hf5','Leg velocity 5-100 Hz (rad/s)'),('arm_velocity_quiet_hf5','Arm velocity 5-100 Hz (rad/s)')]
    for ax,(key,title) in zip(axes.ravel(),fields):
        for i,mode in enumerate(MODES):
            ax.bar(np.arange(3)+(i-1)*.24,[means[c][mode][key] if means[c][mode]['complete']==1 else np.nan for c in CASES],.24,color=COLORS[i],label=LABELS[i])
            for j,c in enumerate(CASES):
                if means[c][mode]['complete']!=1:ax.text(j+(i-1)*.24,.02,'failed',rotation=90,fontsize=7,color=COLORS[i],transform=ax.get_xaxis_transform())
        ax.set_xticks(range(3),CASES);ax.set_title(title,fontsize=10);ax.set_ylim(bottom=0)
    fig.legend(LABELS,loc='lower center',ncol=3,fontsize=8,bbox_to_anchor=(.5,.005))
    fig.suptitle('Same tool task; different base objectives and complete controller architectures')
    fig.tight_layout(rect=(0,.05,1,.97));fig.savefig(OUT/'comparison.png');fig.savefig(OUT/'comparison.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,6),sharex='col')
    for j,case in enumerate(CASES):
        for i,mode in enumerate(MODES):
            a=data[f'{case}_{mode}_r1'];base=basedata[f'{case}_{mode}_r1']
            axes[0,j].plot(a['time'],a['ee_position_error_mm'],color=COLORS[i],lw=1,label=LABELS[i])
            offset=np.linalg.norm(base[:,1:4]-base[0,1:4],axis=1)*1000
            axes[1,j].plot(base[::5,0],offset[::5],color=COLORS[i],lw=1)
        axes[0,j].set_title(case.capitalize());axes[1,j].set_xlabel('Time (s)')
        if case=='push':
            for ax in axes[:,j]:ax.axvspan(6,6.15,color='red',alpha=.08)
    axes[0,0].set_ylabel('Tool position error (mm)');axes[1,0].set_ylabel('Base offset from initial pose (mm)')
    axes[0,0].legend(fontsize=7);fig.suptitle('Base displacement is coordination motion, not a common base tracking error')
    fig.tight_layout();fig.savefig(OUT/'tracking_base.png');fig.savefig(OUT/'tracking_base.pdf');plt.close(fig)
    lines=['# 当前在线 MPC＋几何 WBC 与优先级运动分配＋WBC','',
        f"已执行 {len(records)} 次试验，其中 {sum(r['complete'] for r in records)} 次跑满 12 s，其余提前终止并保留失败记录。三工况、各两次，第二轮反序串行运行。",'',
        '## 当前结论','',
        '当前完整 MPC＋几何 WBC 方案的实际高频运动较小、接触保持更好，但计算量更高，末端姿态误差也明显大于运动分配基线，不能判定全面优越。上轮约 19% 的加速是 MPC 内部求解器实现对照，不是相对运动分配＋WBC 的加速。',
        f"普通工况：接触修正的滞环基线与当前 MPC 的末端位置 RMSE 分别为 {means['normal']['allocation_rolling']['ee_position_rmse_mm']:.3f}/{means['normal']['mpc']['ee_position_rmse_mm']:.3f} mm，姿态分别为 {means['normal']['allocation_rolling']['ee_rotation_rmse_deg']:.4f}/{means['normal']['mpc']['ee_rotation_rmse_deg']:.4f}°，每 12 s 总控制 CPU 分别为 {means['normal']['allocation_rolling']['total_cpu_ms']/1000:.3f}/{means['normal']['mpc']['total_cpu_ms']/1000:.3f} s。", 
        f"完整曲线显示，基线在大部分正常运动时段的末端跟踪更准，边界附近的误差尖峰拉高了其全程 RMSE。普通工况前 0–8 s 位置 RMSE：基线 {means['normal']['allocation_rolling']['ee_position_first8_rmse_mm']:.4g} mm，MPC {means['normal']['mpc']['ee_position_first8_rmse_mm']:.4g} mm。这个窗口仅用于解释全程指标，不替代完整时域结论。", 
        '高频统计的 8–12 s 窗口排除了 6 s 的外力脉冲，但包含基线边界附近的切换／恢复瞬态。因此本轮证据支持运动尖峰和接触异常较少，不能由频带能量下降直接推出持续周期抖振被普遍消除。',
        '当前 MPC 六次全部完成；接触修正基线完成普通和扰动的四次运行，双幅两次在约 5.25 s 终止；保留旧接触稳定项的基线六次均提前终止。未完成时域不用于全程误差或计算节省排名。',
        '独立重放确认，双幅接触修正基线失败时底盘 y=0.151367 m，超过本站立移植设置的 +0.15 m 界约 1.37 mm。一控制步返回边界需要速度≤−0.27346 m/s，而速度下界为−0.25 m/s，故当前状态下生成的速度区间为空。这是当前实现的边界恢复失败，不能据此断言末端任务本身不可行或原论文方法必然失败。MPC 的外部底盘目标及边界结构不同，此失败也不能单独作为预测优势的因果证据。','',
        '## 比较边界','',
        '这是完整控制架构比较，统一的是机器人、末端位姿／体速度目标、MuJoCo 真值、120 N 外力扰动、200 Hz WBC 和 1 kHz 物理记录。目标轨迹哈希逐项相同。两种基线都保留 0.10／0.12 滞环，不用已知有启停抖动的单阈值版本充当弱基线。',
        '运动分配和 WBC 均以 200 Hz 运行；当前 MPC 以 50 Hz 规划、几何 WBC 以 200 Hz 执行。基线使用原优先级算法的 6D 工程扩展，原论文的 3D 实验并没有被称为已复现的 6D 原方法。',
        '额外保留原球心三轴位置稳定项和仅法向位置稳定项（允许无滑动滚动）两个基线，后者用于主要比较，避免把接触建模修正的收益归因于 MPC。两版本的原分配、层级、参数与最终基座加速度修正 QP 均不改动。',
        '两套方法目标和实现仍不同：基线仅获得当前末端目标，自行分配底盘运动；MPC 还获得未来末端目标以及外部底盘／关节姿态参考。其 WBC 的任务层级、加速度／力矩变化率约束及关节伺服参考生成也不同。因此不称为相同 QP 加速比，不由底盘偏移量判断谁的底盘跟踪更准，也不把差异单独归因于李群。','',
        '## 共同末端任务结果','']
    for case in CASES:
        lines += [f'### {case}','']
        for i,mode in enumerate(MODES):
            m=means[case][mode]
            if m['complete']!=1:
                failures=[summaries[f'{case}_{mode}_r{rep}']['failed'] for rep in [1,2]]
                lines.append(f"- {LABELS[i]}：存在未完成运行，失败记录 {failures}。不使用不完整时域的误差或 CPU 耗时作优劣排名。")
                continue
            lines.append(f"- {LABELS[i]}：完成率 {m['complete']:.0%}；末端位置／姿态 RMSE {m['ee_position_rmse_mm']:.5g} mm／{m['ee_rotation_rmse_deg']:.5g}°；每 12 s 控制 CPU {m['total_cpu_ms']/1000:.4g} s，WBC 超时平均 {m['wbc_deadline_misses']:.1f}/2400 次。")
            lines.append(f"  底盘位置／姿态运动 RMS {m['base_offset_rms_mm']:.5g} mm／{m['base_rotation_offset_rms_deg']:.5g}°；后段 8–12 s 底盘角速度 10–100 Hz RMS {m['base_omega_quiet_hf10_deg_s']:.5g}°/s；实际腿／臂速度 5–100 Hz RMS {m['leg_velocity_quiet_hf5']:.5g}／{m['arm_velocity_quiet_hf5']:.5g} rad/s；力矩高频 RMS {m['leg_torque_quiet_hf5']:.5g}／{m['arm_torque_quiet_hf5']:.5g} N·m。")
            if mode!='mpc':lines.append(f"  对自身生成底盘目标的位置／姿态 RMSE {m['internal_base_position_rmse_mm']:.5g} mm／{m['internal_base_rotation_rmse_deg']:.5g}°，机身启停切换平均 {m['switches']:.1f} 次。")
        d=changes[case]
        if d['total_cpu_ms'] is not None:
            lines.append(f"- 当前 MPC 相对接触修正的强基线：末端位置误差 {d['ee_position_rmse_mm']:+.3g}%，姿态误差 {d['ee_rotation_rmse_deg']:+.3g}%，总 CPU {d['total_cpu_ms']:+.3g}%；底盘角速度高频量 {d['base_omega_quiet_hf10_deg_s']:+.3g}%，实际腿／臂速度高频量 {d['leg_velocity_quiet_hf5']:+.3g}%／{d['arm_velocity_quiet_hf5']:+.3g}%。")
        lines.append('')
    lines += ['## 接触、扰动与时限','']
    for i,mode in enumerate(MODES):
        rr=[r for r in records if r['mode']==mode];agg=aggregate[mode]
        lines.append(f"- {LABELS[i]}：已执行时域内失去四足接触合计 {agg['missing_contact_ms']} ms；力矩裁剪 {agg['clips']} 次；最终接触加速度残差最大 {max(r['final_contact_acc_residual'] for r in rr):.5g} m/s²；接触材料点速度 RMS 最大 {max(r['contact_speed_rms_mm_s'] for r in rr):.5g} mm/s；WBC 超时 {agg['wbc_deadline_misses']}/{agg['cycles']} 次。")
        m=means['push'][mode];lines.append(f"  外力施加后 6–7 s 的末端位置／姿态 RMSE {m['ee_position_push_window_rmse_mm']:.5g} mm／{m['ee_rotation_push_window_rmse_deg']:.5g}°。")
    lines += ['',f"本轮 MPC 规划超 20 ms 共 {aggregate['mpc']['planner_deadline_misses']} 次，跳过 {aggregate['mpc']['skipped_planner_requests']} 个请求槽。",'',
        '总 CPU 按各实际控制线程计时，包括状态／模型、数值梯度、运动分配、矩阵、求解、验收和几何参考；初始化／公共目标预计算、日志与物理引擎积分不计入在线工作量。基线一核，MPC＋WBC 两核；报告工作量而非用并行墙钟相加。',
        '墙钟调度和迟到计划处理实际运行；物理引擎仍以每次控制推进五个 1 ms 步长，未加入实机丢帧保持力矩模型。两次重复不构成统计显著性或硬实时保证。',
        '基线最终 QP 仅修正基座加速度，可能改变前级已满足的接触加速度。当前几何 WBC 在最终求解中把接触作为硬等式。这是执行层结构区别，并不等同于 MPC 预测的独立收益。',
        '位移与姿态偏移以初始底盘为参照，含主动协调运动；内部目标跟踪误差单独列出。高频指标取实际 1 kHz 速度／施加力矩，后段窗口 8–12 s 包含基线边界附近的瞬态，另保存 2–12 s 全窗口及不同频带。',
        '若要证明 MPC 的独立收益，需要让两种上层共用相同几何 WBC、相同任务优先级及边界，再比较接近工作空间／关节／力矩边界或未来参考突变等需要预见的工况。本轮不能直接给出这个因果结论。','',
        '## 复现','',
        '工程 /home/ganhe/project/b2z1_mpc，激活 relu_hgmpc/env.sh，运行 benchmark_allocation_common.py 和 report_allocation_common.py。原始时序、代码及目标哈希保存在 results/allocation_vs_online_mpc。当前 MPC 控制器和基线算法源码没有因这次对比而修改。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(clean(dict(aggregate=aggregate,means=means,changes=changes)),indent=2))

if __name__=='__main__':main()
