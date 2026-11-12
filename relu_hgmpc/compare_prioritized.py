import json,pathlib,shutil,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=pathlib.Path(__file__).resolve().parents[1];out=root/'results'/'prioritized';old=root/'results'/'relu_hgmpc'
specs=[('ReLU-QP MPC',old,'baseline_i100_final_12s','#2464a5'),
       ('Prioritized GMPC',old,'hgmpc_i100_final_12s','#d26628'),
       ('Xie WBC: 6D extension',out,'xie_6d_final_r1','#38814a')]
data={};summary={};repro={}
for label,folder,name,color in specs:
    data[label]=np.genfromtxt(folder/f'{name}.csv',delimiter=',',names=True)
    summary[label]=json.loads((folder/f'{name}.json').read_text())
for dim in (3,6):
    a=np.genfromtxt(out/f'xie_{dim}d_final_r1.csv',delimiter=',',names=True)
    b=np.genfromtxt(out/f'xie_{dim}d_final_r2.csv',delimiter=',',names=True)
    cols=['position_error_mm','rotation_error_deg','base_z','tilt_rad','foot_drift_mm','contacts','base_active','sigma_min']
    diff={k:float(np.max(np.abs(a[k]-b[k]))) for k in cols}
    assert all(v==0 for v in diff.values()),diff
    repro[f'{dim}D']=diff
(out/'reproducibility.json').write_text(json.dumps(repro,indent=2))

fig,ax=plt.subplots(2,2,figsize=(12,7.5),layout='constrained')
for label,_,_,color in specs:
    x=data[label]
    for a,key,ylabel in zip(ax.flat,['position_error_mm','rotation_error_deg','foot_drift_mm','feedback_ms'],
                           ['Position error (mm)','Orientation error (deg)','Maximum foot drift (mm)','Feedback computation time (ms)']):
        a.plot(x['time'],x[key],label=label,color=color,lw=1.2);a.set_xlabel('Simulation time (s)');a.set_ylabel(ylabel);a.grid(alpha=.2)
for a in ax.flat:a.legend(fontsize=8)
ax[1,1].axhline(20,color='black',ls='--',lw=1);ax[1,1].set_yscale('log')
fig.suptitle('B2+Z1: common 6D target, 12 s standing manipulation\nMPC: 100 iterations / QP; WBC: converged 18-variable active-set QP; no computation delay injection',fontsize=11)
fig.savefig(out/'three_method_comparison.png',dpi=180);plt.close(fig)

x3=np.genfromtxt(out/'xie_3d_final_r1.csv',delimiter=',',names=True);x6=data['Xie WBC: 6D extension']
fig,ax=plt.subplots(3,1,figsize=(10,7),layout='constrained')
for x,label,color in [(x3,'Original 3D task','#707070'),(x6,'6D extension','#38814a')]:
    ax[0].plot(x['time'],x['position_error_mm'],label=label,color=color)
    ax[1].plot(x['time'],x['sigma_min'],label=label,color=color)
    ax[2].step(x['time'],x['contacts'],where='post',label=label,color=color)
ax[1].axhline(.1,ls='--',color='black',lw=1,label='Activation threshold')
for a,label in zip(ax,['Position error (mm)','Smallest singular value','Number of contacting feet']):
    a.set_xlabel('Simulation time (s)');a.set_ylabel(label);a.legend(fontsize=8);a.grid(alpha=.2)
ax[2].set_ylim(-.2,4.3);ax[2].set_yticks(range(5))
fig.suptitle('Xie et al. algorithm structure on B2+Z1\n3D original task and 6D extension are reported separately',fontsize=12)
fig.savefig(out/'original_and_extension.png',dpi=180);plt.close(fig)

lines=['# 第三种方法：任务优先级运动分配＋WBC 对照','',
       '2026-11-09。B2+Z1 / MuJoCo 真值 / 12 s / 20 ms 控制周期。原有两种 MPC 使用每层 100 次 ReLU-QP 迭代，WBC 的 18 变量 QP 使用 CPU quadprog 收敛解。',
       '', '**复现范围是论文算法结构在 B2+Z1 站立模式的移植**。3D 是原文任务维度；6D 是为了与现有 MPC 位姿任务衔接的扩展。原机器人、原低层参数和 trot 实验未逐项复现。','',
       '## 同一 6D 目标下的结果','']
compact=[]
for label,_,_,_ in specs:
    r=summary[label];x=data[label]
    m=dict(method=label,position_rmse_mm=r['position_rmse_mm'],rotation_rmse_deg=r['rotation_rmse_deg'],
           feedback_mean_ms=r['feedback_ms']['mean'],feedback_p95_ms=r['feedback_ms']['p95'],
           max_foot_drift_mm=float(np.max(x['foot_drift_mm'])),min_contacts=int(np.min(x['contacts'])),
           incomplete_contact_cycles=int(np.sum(x['contacts']<4)),torque_clips=r['torque_clips'])
    compact.append(m)
    lines += [f'**{label}**：位置 RMSE {m["position_rmse_mm"]:.3f} mm；姿态 RMSE {m["rotation_rmse_deg"]:.3f}°；反馈计算平均 {m["feedback_mean_ms"]:.3f} ms，P95 {m["feedback_p95_ms"]:.3f} ms。',
              f'最大足端漂移 {m["max_foot_drift_mm"]:.3f} mm；最少接触足数 {m["min_contacts"]}；不足四足接触的采样周期 {m["incomplete_contact_cycles"]}/600；力矩截断 {m["torque_clips"]} 次。','']
r3=json.loads((out/'xie_3d_final_r1.json').read_text())
lines += ['## 原文 3D 位置任务','',
          f'位置 RMSE {r3["position_rmse_mm"]:.3f} mm；反馈计算平均 {r3["feedback_ms"]["mean"]:.3f} ms；P95 {r3["feedback_ms"]["p95"]:.3f} ms。',
          f'最大足端漂移 {r3["foot_drift_mm"]["max"]:.3f} mm；最少接触足数 {int(np.min(x3["contacts"]))}；基座激活比例 {100*r3["base_active_fraction"]:.1f}%。',
          '姿态不是该版本的任务，不能拿其姿态误差与两个 6D MPC 排名。本条小幅轨迹未触发机械臂饱和；另有真实 Z1 近奇异位形、饱和、phi/psi 缩放和零空间投影的测试。','',
          '## 结论的适用范围','',
          '本例中速度层运动分配＋小型 WBC QP 的计算成本明显较小，末端跟踪也有竞争力。这是控制器结构、反馈/前馈设计和任务参数共同产生的结果，不是同一个优化问题更换求解器后的加速比。',
          '2026-11-10 校核更新：1 kHz 接触诊断与单因素消融确认，原 6D 扩展的机身单阈值启停抖动是主要触发因素。仅增加 0.10 开启／0.12 关闭滞环，同一 6D 任务和 WBC 即全程四足接触。此处保留原设置的数据，不应将其问题归因于 6D 任务本身。另须注意，足端漂移是球心位移，包含滚动，不能直接视为打滑；完整证据见工程 results/contact_audit/RESULTS.md。',
          '原任务与扩展各独立运行两次，非计时物理轨迹逐项精确一致；所有真实关节位置/速度越界检查通过。计算时间受运行时调度影响。',
          '现有 MPC 对照沿用上轮相同硬件与模型的已保存实验；它们是固定迭代近似解，原始/对偶残差和详细计时见随附摘要，未伪装成相同最优性精度。',
          '仿真未注入计算延迟；GMPC 超过 20 ms 不能据此声称满足 50 Hz 实时控制。','',
          '## 数学和数值核对','',
          '式 (17) 的原表达与交集零空间投影等价；式 (7) 的饱和分配需显式扣除已固定自由度的任务贡献；P_N 的维数按整机器人雅可比解释。全部差异、参数和未覆盖范围见 PRIORITIZED.md。',
          '验证包含真实 Z1 近奇异位形激活基座、饱和再分配、最大 phi、可行 psi、被冻结自由度不受第三任务改变、动态一致 WBC 层级保持，以及实际困难 QP 与独立 SLSQP 的目标值核对。',
          '', '![三种方法对比](three_method_comparison.png)','', '![原任务和扩展](original_and_extension.png)','',
          '数据：各方法摘要 JSON、逐周期 CSV、verification.json、reproducibility.json。',
          '方法来源：[Xie et al., 2025](https://doi.org/10.34133/cbsystems.0203)；本地 Zotero 附件 X8WCA9UY。']
(out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(out/'comparison.json').write_text(json.dumps(compact,indent=2))
print(json.dumps(compact,indent=2));print('Original 3D RMSE / time:',r3['position_rmse_mm'],r3['feedback_ms']['mean'])
if len(sys.argv)>1:
    dst=pathlib.Path(sys.argv[1]);dst.mkdir(parents=True,exist_ok=True)
    for name in ['RESULTS.md','comparison.json','verification.json','reproducibility.json','three_method_comparison.png','original_and_extension.png','failure_6d.npz']:
        shutil.copy2(out/name,dst/name)
    for p in out.glob('xie_*_final_r*.*'):shutil.copy2(p,dst/p.name)
    shutil.copy2(root/'relu_hgmpc/PRIORITIZED.md',dst/'PRIORITIZED.md')
    for _,folder,name,_ in specs[:2]:
        for suffix in ('.json','.csv'):shutil.copy2(folder/(name+suffix),dst/(name+suffix))
