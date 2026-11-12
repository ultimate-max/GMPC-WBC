"""Scientific plots and an auditable report for the geometry-only WBC ablation."""
import csv,json,pathlib
import numpy as np
from scipy.signal import welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'results/geometric_wbc'
CASES=['normal','double','offset'];MODES=['product','se3']
COLORS={'product':'#e18a24','se3':'#1673b9'}
LABELS={'product':'Position + SO(3) log','se3':'SE(3) log'}

def hf(x,average=False):
    freq,power=welch(x,fs=1000,nperseg=2048,noverlap=1024,axis=0)
    mask=(freq>=10)&(freq<=100)
    variances=np.trapezoid(power[mask],freq[mask],axis=0)
    return float(np.sqrt(np.mean(variances) if average else np.sum(variances)))

def rms(x):return float(np.sqrt(np.mean(np.asarray(x)**2)))

def main():
    metrics=[];data={};summaries={};traces={}
    for case in CASES:
        for mode in MODES:
            for rep in [1,2]:
                name=f'{case}_{mode}_r{rep}'
                summary=json.loads((OUT/f'{name}.json').read_text());summaries[name]=summary
                assert summary['failed'] is None and summary['simulated_cycles']==2400,name
                a=np.genfromtxt(OUT/f'{name}.csv',delimiter=',',names=True)
                tr=np.load(OUT/f'{name}_trace.npz');data[name]=a;traces[name]=tr
                t=tr['trace'];base=tr['base'];steady=t[:,0]>=2.
                row=dict(case=case,mode=mode,repeat=rep)
                for task in ['base','ee']:
                    for measure,unit in [('position','mm'),('rotation','deg')]:
                        col=f'{task}_{measure}_error_{unit}'
                        row[f'{task}_{measure}_rmse_{unit}']=rms(a[col])
                        row[f'{task}_{measure}_steady_rmse_{unit}']=rms(a[col][a['time']>=2.])
                        row[f'{task}_{measure}_first_half_second_rmse_{unit}']=rms(a[col][a['time']<.5])
                row.update(base_omega_hf_deg_s=hf(base[steady,17:20])*180/np.pi,
                    leg_velocity_hf_rad_s=hf(t[steady,111:123],True),arm_velocity_hf_rad_s=hf(t[steady,123:129],True),
                    leg_torque_hf_nm=hf(t[steady,57:69],True),arm_torque_hf_nm=hf(t[steady,69:75],True),
                    leg_torque_rate_rms_nm_s=rms(np.diff(t[steady,57:69],axis=0)/.001),
                    arm_torque_rate_rms_nm_s=rms(np.diff(t[steady,69:75],axis=0)/.001),
                    missing_contact_ms=int(np.sum(t[:,1:5].sum(axis=1)<4)),min_contacts=int(t[:,1:5].sum(axis=1).min()),
                    clips=summary['torque_clips'],feedback_mean_ms=float(a['feedback_ms'].mean()),
                    feedback_p95_ms=float(np.quantile(a['feedback_ms'],.95)),feedback_p99_ms=float(np.quantile(a['feedback_ms'],.99)),
                    feedback_max_ms=float(a['feedback_ms'].max()),over_5ms_count=int(np.sum(a['feedback_ms']>5)),
                    model_mean_ms=float(a['model_ms'].mean()),qp_residual_max=max(float(a[k].max()) for k in ['qp_primal_relative','qp_dual_relative','qp_normal_relative']),
                    contact_residual_max=float(a['contact_residual'].max()),dynamics_residual_max=float(a['dynamics_residual'].max()),
                    task_lock_max=float(a['task_lock_max'].max()),constraint_violation_max=float(a['constraint_violation'].max()))
                pointvel=t[:,21:33].reshape(-1,4,3);contact=t[:,1:5]>0
                row['contact_material_speed_rms_mm_s']=rms(np.linalg.norm(pointvel[contact],axis=1))*1000
                metrics.append(row)
    repeat_checks={};pairs={}
    for case in CASES:
        hashes={summaries[f'{case}_{mode}_r{rep}']['reference_sha256'] for mode in MODES for rep in [1,2]}
        assert len(hashes)==1
        for mode in MODES:
            first=traces[f'{case}_{mode}_r1'];second=traces[f'{case}_{mode}_r2']
            repeat_checks[f'{case}_{mode}']={key:bool(np.array_equal(first[key],second[key])) for key in ['q','v','tau','acc','trace','base']}
        means={mode:{k:float(np.mean([r[k] for r in metrics if r['case']==case and r['mode']==mode])) for k in metrics[0] if k not in ['case','mode','repeat']} for mode in MODES}
        changes={k:100*(means['se3'][k]/means['product'][k]-1) if means['product'][k]!=0 else None for k in means['product']}
        pairs[case]=dict(means=means,se3_change_percent=changes,common_reference_sha256=next(iter(hashes)))
    with (OUT/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    comparison=dict(pairs=pairs,repeat_checks=repeat_checks,verification=json.loads((OUT/'verification.json').read_text()),
        scope='Offline common dynamically feasible reference, no online MPC; both error choices already use SO(3); identical 3-level hierarchy and constraints.',
        hf_definition='Steady window 2-12 s; true 1 kHz channels; Welch 2048 points, overlap 1024, integrated 10-100 Hz PSD. Base = vector RMS; joint metrics = mean channel RMS.',
        timing='Serialized runs, order reversed in repeat 2; total state read + model + derivatives + three QPs + checks; no compute-delay injection. Offline preparation separately reported.',
        contact_change='New experiment only: retain native spherical no-slip contact Jacobian/drift, remove incompatible horizontal sphere-center position stabilization; keep normal position and full contact velocity stabilization.',
        residuals='KKT relative residuals use positive scalar-normalized objective and original constraint rows; physical/hierarchy residuals separately checked in original units.')
    (OUT/'comparison.json').write_text(json.dumps(comparison,indent=2))
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.25,'figure.dpi':140})
    fig,axes=plt.subplots(4,3,figsize=(13,10),sharex='col')
    fields=[('base_position_error_mm','Base position (mm)'),('base_rotation_error_deg','Base orientation (deg)'),
        ('ee_position_error_mm','Tool position (mm)'),('ee_rotation_error_deg','Tool orientation (deg)')]
    for j,case in enumerate(CASES):
        for i,(key,label) in enumerate(fields):
            ax=axes[i,j]
            for mode in MODES:
                a=data[f'{case}_{mode}_r1'];ax.plot(a['time'],a[key],color=COLORS[mode],lw=1.2,label=LABELS[mode])
            ax.set_ylabel(label if j==0 else '');ax.set_title(case.capitalize() if i==0 else '')
            if i==3:ax.set_xlabel('Time (s)')
            if case=='offset':ax.set_yscale('symlog',linthresh=.01)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Matched-reference WBC: true tracking error (offset column uses symlog)',y=.995)
    fig.tight_layout();fig.savefig(OUT/'tracking.png');fig.savefig(OUT/'tracking.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    barfields=[('base_omega_hf_deg_s','Base angular velocity HF (deg/s)'),('leg_velocity_hf_rad_s','Leg joint velocity HF (rad/s)'),
        ('arm_velocity_hf_rad_s','Arm joint velocity HF (rad/s)'),('leg_torque_hf_nm','Leg torque HF (Nm)'),
        ('arm_torque_hf_nm','Arm torque HF (Nm)'),('feedback_mean_ms','Total feedback mean (ms)')]
    for ax,(key,title) in zip(axes.ravel(),barfields):
        x=np.arange(3)
        for i,mode in enumerate(MODES):ax.bar(x+(i-.5)*.34,[pairs[c]['means'][mode][key] for c in CASES],.34,label=LABELS[mode],color=COLORS[mode])
        ax.set_xticks(x,CASES);ax.set_title(title,fontsize=10);ax.set_ylim(bottom=0)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Same constraints, priorities and gains; vibration measured over 2-12 s')
    fig.tight_layout();fig.savefig(OUT/'comparison.png');fig.savefig(OUT/'comparison.pdf');plt.close(fig)
    lines=['# 共同轨迹下的几何 WBC 验证','',
        '本轮只验证规划与执行分离方案中的执行层。使用离线共同可行轨迹，尚未接入在线 MPC，也未模拟计算延迟。不能据此宣称 MPC 加速或完整架构稳定性。','',
        '## 当前结论','',
        '12 次、每次 12 s 的闭环全部完成，共 28,800 次反馈、144,000 次物理步。两种方法均保持四足接触，没有力矩裁剪。SE(3) 总反馈均值约 0.97–0.99 ms，基线约 1.10–1.11 ms；这是当前执行层实现的耗时，不能换算成 MPC 加速比。',
        '完整 SE(3) 误差没有呈现跨工况的全面优势。普通幅度下底盘姿态 RMSE 下降约 5.3%，双幅下却增加约 2.5%；普通幅度底盘高频角速度下降约 7.7%，双幅下增加约 13.8%。三个工况的腿部力矩高频 RMS 均下降（约 8.5%–30%），但不等于实际关节速度抖振也始终下降。普通与双幅的底盘位置误差几乎相同。',
        '因此，这轮支持快速、受约束的几何执行层可以运行，不支持“仅替换为 SE(3) 就能同时提高跟踪与全面消除抖振”的强结论。基线本身已正确使用 SO(3)，小误差时两种表示接近；这是解释结果接近的数学依据，不能单独解释所有非线性接触波动差异。','',
        '## 对照条件','',
        '- 对照为位置差＋SO(3) 对数姿态误差，与完整 SE(3) 对数误差；两者都包含正确的运动参考速度搬移及二阶误差导数。基线不是欧拉角误差。',
        '- 相同 B2+Z1、Pinocchio、MuJoCo 真值、200 Hz 反馈、1 kHz 物理与执行器记录；相同力矩、摩擦锥、接触力、速度、位置及力矩变化率边界。两者同加关节加速度设计边界：腿 50、臂 30 rad/s²；这些是控制设计值，非硬件标称值。',
        '- 严格层级：硬动力学与无滑动接触 > 底盘位姿 > 末端位姿 > 姿态/接触力/力矩平滑。后层以零空间保持前层实际实现的任务值。',
        '- 普通幅度、双幅、初始参考偏差恢复；各 12 s，各方法两次串行重复，第二次倒序。恢复工况初始参考相对实际状态偏移约 24.5 mm、9.9°，不是外力冲击试验。',
        '- 轨迹在每个 5 ms 节点检查接触运动学、逆动力学、摩擦锥、力矩和关节边界；不是连续时间可行性证明。原始试验和控制器未覆盖。','',
        '## 接触模型排查','',
        '原 native Jc 对应球形足底瞬时接触点速度 v_center + omega × (0,0,-r)，而 ce 原先约束球心三轴位置。球体转动时，球心固定与接触点无滑动不一致。新实验保留球底速度约束，仅保留球心法向位置稳定项，并通过积分无滑动运动学构造腿部参考；允许无滑动滚动引起的水平球心位移。此修正仅用于新对照，不能把它造成的改善归于 SE(3)。','',
        '## 数值检查','',
        '所有 KKT 检查使用同一目标标量归一化规则；不把归一化后的对偶残差冒充原目标单位残差。动力学、接触与层级保持另行按原单位检查。Jlog 的方向导数通过中心差分计算，未宣称全解析二阶导数。',
        '两者统一使用 quadprog 主动集求解器。半正定任务 Hessian 加 1e-10 数值对角项，验收仍检查未加该项的目标。前层正对偶乘子对应的激活边界随任务输出一起锁定，下层可用维数允许减少到零。仅对继承解的 2e-7 内浮点边界误差归零，最终仍核验原始物理边界。',
        '调试中发现原始仅靠一步速度上限的底盘优先级会将轻质腕部快速加速；恢复工况首步真实关节速度超限约 0.029 rad/s。增加两方法共同的加速度设计边界后重新开展全部对照。此改进来自约束设计，不能归于李群表示。失败证据保存在 diagnostics。','']
    maxkeys=['qp_residual_max','contact_residual_max','dynamics_residual_max','task_lock_max','constraint_violation_max']
    for key in maxkeys:lines.append(f'- {key}: {max(r[key] for r in metrics):.6g}')
    lines += ['',f'两次重复轨迹逐元素完全相同：{all(all(v.values()) for v in repeat_checks.values())}。这验证确定性重复，不代表统计显著性。','', '## 结果','']
    for case in CASES:
        lines.append(f'### {case}')
        lines.append('')
        for mode in MODES:
            m=pairs[case]['means'][mode]
            lines.append(f"- {LABELS[mode]}：底盘位置 RMSE {m['base_position_rmse_mm']:.6g} mm，姿态 {m['base_rotation_rmse_deg']:.6g}°；末端位置 {m['ee_position_rmse_mm']:.6g} mm，姿态 {m['ee_rotation_rmse_deg']:.6g}°；总反馈均值 {m['feedback_mean_ms']:.4g} ms，P99 {m['feedback_p99_ms']:.4g} ms；少于四足接触 {m['missing_contact_ms']:.0f} ms，力矩裁剪 {m['clips']:.0f} 次。")
        d=pairs[case]['se3_change_percent']
        lines.append(f"- SE(3) 相对变化：底盘位置 RMSE {d['base_position_rmse_mm']:+.4g}%，姿态 {d['base_rotation_rmse_deg']:+.4g}%；底盘角速度高频 RMS {d['base_omega_hf_deg_s']:+.4g}%；腿/臂实际速度高频 RMS {d['leg_velocity_hf_rad_s']:+.4g}% / {d['arm_velocity_hf_rad_s']:+.4g}%；腿/臂施加力矩高频 RMS {d['leg_torque_hf_nm']:+.4g}% / {d['arm_torque_hf_nm']:+.4g}%。")
        lines.append('')
    lines += ['## 计时口径与复现','',
        '总反馈计时包括读取状态、模型与矩阵、几何导数、三级 QP、物理及层级检查。离线参考生成约 1–2 s，实际值见 reference_*.json，不能把该离线过程当作实时规划器。CSV 的 feedback_ms 不含 MuJoCo 推进、日志和初始化，初始化与预加载另列。未模拟超时或计算延迟，P99 不能当作最坏执行时间保证。','',
        '运行：在 /home/ganhe/project/b2z1_mpc 下启用 relu_hgmpc/env.sh 后，依次执行 test_geometric_wbc.py、benchmark_geometric_wbc.py、report_geometric_wbc.py。源代码在 relu_hgmpc；全部原始记录、轨迹哈希和源码哈希在 results/geometric_wbc。','',
        '结论必须结合上述数值：这一对照可以判断完整 SE(3) 表示对强几何基线的增益，不能解释为“几何 WBC 对普通 WBC”的全面优越性，也不能归因于 MPC。','',
        '## 后续在线 MPC 接口','',
        '上层应输出一致的 q、v、a、接触力与接触模式，并由同一运动学计算底盘/末端的位姿、速度和加速度参考；不能独立插值出彼此冲突的任务目标。下层反馈已实现的任务加速度、任务残差、激活约束和力矩/摩擦余量，使上层更新可执行轨迹。',
        '下一项需要单独验证在线 MPC 规划耗时、重规划频率、参考更新平滑性和计算延迟下的闭环，不能用当前离线参考回放代替。论文创新可继续检验“执行可行性反馈如何减少规划更新并保持误差界”，当前结果尚未证明该创新。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(pairs,indent=2))

if __name__=='__main__':main()
