import pathlib,json,sys,shutil
import numpy as np
from scipy.optimize import linprog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/contact_audit'
names=['threshold','hysteresis','always','arm']
summaries={n:json.loads((out/f'{n}.json').read_text()) for n in names}
arrays={n:np.genfromtxt(out/f'{n}.csv',delimiter=',',names=True) for n in names}
traces={n:np.load(out/f'{n}_trace.npz') for n in names}
base=traces['threshold'];j=140;A=base['A'][j];low=base['low'][j];upp=base['upp'][j]
results={}
for label,rows in [('all',np.arange(6,44)),('no_torque',np.arange(6,26)),('no_contact_force_limits',np.arange(26,44))]:
    C=A[rows,6:];lo=low[rows];hi=upp[rows];ml=np.isfinite(lo);mu=np.isfinite(hi)
    r=linprog(np.zeros(12),A_eq=A[:6,6:],b_eq=low[:6],A_ub=np.vstack([-C[ml],C[mu]]),
              b_ub=np.r_[-lo[ml],hi[mu]],bounds=[(None,None)]*12,method='highs')
    results[label]=dict(feasible=r.success,status=r.message)
(out/'frozen_snapshot_feasibility.json').write_text(json.dumps(results,indent=2))

fig,ax=plt.subplots(3,1,figsize=(11,8),sharex=True,layout='constrained')
for name,label,color in [('threshold','Single threshold (original 6D extension)','#b65035'),('hysteresis','Hysteresis; same WBC and trajectory','#2464a5')]:
    x=arrays[name];z=traces[name]
    ax[0].plot(x['time'],x['sigma'],label=label,color=color)
    ax[1].step(z['trace'][:,0],z['trace'][:,1:5].sum(axis=1),where='post',color=color,label=label)
    ax[2].plot(x['time'],x['contact_post'],color=color,label=label)
ax[0].axhline(.1,color='gray',ls='--',lw=1);ax[0].axhline(.12,color='gray',ls=':',lw=1)
ax[0].set_ylabel('Smallest arm singular value');ax[1].set_ylabel('Contacting feet (1 kHz)');ax[1].set_yticks([2,3,4]);ax[1].set_ylim(1.8,4.15)
ax[2].set_ylabel('Post-QP contact residual (m/s²)');ax[2].set_xlabel('Simulation time (s)')
for a in ax:a.grid(alpha=.2);a.legend(fontsize=8);a.set_xlim(2.5,4.4)
fig.suptitle('6D contact audit: switching triggers the contact failure\nOnly activation logic changes; identical task, WBC, bounds and plant',fontsize=12)
fig.savefig(out/'switching_cause.png',dpi=180);plt.close(fig)

fig,ax=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
for name,label,color in [('threshold','Single threshold','#b65035'),('hysteresis','Hysteresis','#2464a5')]:
    z=traces[name]['trace'];mask=z[:,1:5]>0;vc=z[:,21:33].reshape(-1,4,3);vc[~mask]=0
    cumulative=np.cumsum(np.linalg.norm(vc[:,:,:2],axis=2),axis=0) # 1 ms * 1000 mm/m
    ax[0].plot(arrays[name]['time'],arrays[name]['center_drift_mm'],label=label,color=color)
    ax[1].plot(z[:,0],cumulative.max(axis=1),label=label,color=color)
ax[0].set_ylabel('Maximum foot-center displacement (mm)');ax[1].set_ylabel('Maximum accumulated sliding path (mm)')
for a in ax:a.set_xlabel('Simulation time (s)');a.grid(alpha=.2);a.legend(fontsize=8)
fig.suptitle('Foot-center displacement includes rolling; it is not a sliding metric',fontsize=12)
fig.savefig(out/'rolling_vs_sliding.png',dpi=180);plt.close(fig)

old=np.genfromtxt(root/'results/prioritized/xie_6d_final_r1.csv',delimiter=',',names=True)
repro_error=float(np.max(np.abs(old['position_error_mm']-arrays['threshold']['position_error_mm'])))
threshold=summaries['threshold'];hys=summaries['hysteresis'];arm=summaries['arm']
lines=['# 6D 足端接触问题：原因审计','',
 '2026-11-10。使用当前 B2+Z1 / MuJoCo / 12 s 同一 6D 目标和 20 ms 控制周期，新增 1 ms 接触诊断。',
 f'原始 6D 结果已重现；位置误差曲线与上轮逐点最大差 {repro_error:.3g} mm。','',
 '**结论：主要故障是本次 6D 扩展中机身激活的硬切换，不是 6D 任务天然不能保持接触。此前将 WBC 结构笼统地作为主要原因，证据不足。球心位移也不能直接作为打滑指标。**','',
 '## 已验证的因果链','',
 '1. 将原 3D 位置任务的 0.1 奇异值阈值直接用于含平移/转动的 6D 雅可比，没有重新归一化或标定；两种雅可比的奇异值不具相同尺度意义。本轨迹单臂实际上可以完成，单臂可行性对照已经验证。',
 '2. 每个周期独立判断 sigma<0.1，没有启停滞环。机身在约 2.76–4.2 s 附近频繁加入/退出分配，12 s 共切换 41 次。末端指令连续，但关节/基座速度的分配解不连续。',
 '3. 代码用 `(command-previous_velocity)/0.02` 生成前馈加速度。切换造成最高约 2.51 m/s² 基座平移加速度、4.47 rad/s² 基座角加速度和14.71 rad/s² 臂关节加速度，远高于平滑轨迹本身需要的变化。速度界里的制动距离项不是相邻时刻的加速度变化约束。',
 '4. 在 2.80 s 冻结同一状态和运动指令，独立 LP 证明：不允许基座加速度修正时，原指令不能同时满足本实现的浮动基座动力学、摩擦/法向力及力矩界。WBC 因此大幅修正基座加速度；递归层的接触残差原本约 1e-14，最终 QP 后升至 0.695 m/s²，预测最小法向力碰到 10 N 下界。',
 '5. 首次丢接触发生在 2.807 s。连续时间离散诊断显示至少一足无接触共 130 个 1 ms 步长，瞬时最少两足接触；上次每 20 ms 采样只看到了 7 个异常周期、最少三足，漏掉了更短的瞬态。','',
 '因此，缺少最终硬接触等式是使突变指令传到物理接触的一个通道；直接触发因素是速度分配硬切换与差分加速度，而不是“使用六维末端任务”本身。','',
 '## 单因素对照','']
labels={'threshold':'原单阈值','hysteresis':'0.10 开启／0.12 关闭滞环','always':'全程允许基座参与','arm':'单臂可行性对照'}
for name in names:
    s=summaries[name]
    lines += [f'**{labels[name]}**：切换 {s["switch_count"]} 次；最少 {s["contacts_min"]} 足接触；不足四足接触累计 {s["missing_contact_steps"]} ms；位置 RMSE {s["position_rmse_mm"]:.3f} mm；最大球心位移 {s["max_center_drift_mm"]:.3f} mm；最大单足累计切向滑动路程 {max(s["per_foot_slip_path_mm"]):.3f} mm。','']
lines += ['只改变机身激活逻辑，滞环配置便将切换从 41 次降为 1 次，12 s 全程保持四足接触；WBC 公式、力矩/摩擦界、原末端轨迹和伺服均保持相同。这是主因的直接消融证据。',
 '单臂对照仍保留秩不足/饱和时启用机身的算法保护；本条轨迹没有触发这些分支。它验证本轨迹不需要因复制的 0.1 阈值而频繁调用机身，不代表所有任务都应禁用机身。','',
 '## 球心移动不等于接触点打滑','',
 '当前足端是半径 32 mm 的球体。接触点物质速度满足 `v_contact=v_center+omega cross r`。保持接触且无滑动时，球体仍可滚动，球心也会移动。',
 f'滞环配置的球心最大位移仍为 {hys["max_center_drift_mm"]:.2f} mm，但四足均有接触，最大单足累计滑动路程只有 {max(hys["per_foot_slip_path_mm"]):.3f} mm；位移主要是滚动。',
 '原设置最严重的右后足最终 x 位移约 -14.29 mm，可分解为约 -5.81 mm 接触滑动、-3.99 mm 滚动、-4.47 mm 离地运动。这里是有方向位移分量，不能把各自欧氏长度直接相加。积分为 1 ms 离散估计，冲击瞬间存在离散误差。',
 '所以旧报告的 14.7 mm 是球心位移，既包含真实接触问题，也包含滚动，不能直接称为 14.7 mm 打滑。若要求足印完全固定，还应检查足端几何和滚动约束；这与是否控制 6D 位姿是不同要求。','',
 '## 实现与后续边界','',
 '保留原单阈值默认值以重现旧实验，已添加可选滞环参数。例：',
 '```bash',
 'source /home/ganhe/project/b2z1_mpc/relu_hgmpc/env.sh',
 'python relu_hgmpc/run_prioritized.py --task-dim 6 --activation hysteresis --duration 12 --name xie6_hysteresis',
 'python relu_hgmpc/audit_contact.py --activation threshold',
 'python relu_hgmpc/audit_contact.py --activation hysteresis',
 '```','',
 '本轮证实滞环能修复这条轨迹的接触丢失，不宣称这组阈值适用于全部任务。下一步工程上应标定 6D 雅可比尺度、限制分配解的加速度变化，并考虑让最后一级优化也显式保持接触。只有 6 个基座加速度修正变量时，直接附加接触等式可能令 QP 不可行，需要允许接触一致的全身加速度修正或对任务指令作可行性调整。',
 '原始数据和旧结果未覆盖；目录含 1 kHz 轨迹 NPZ、每周期诊断 CSV、摘要 JSON、冻结状态可行性检查和修改前源码。','',
 '![切换与接触](switching_cause.png)','', '![滚动与滑动](rolling_vs_sliding.png)']
(out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(reproduction_error_mm=repro_error,snapshot_feasibility=results),indent=2))
if len(sys.argv)>1:
    dst=pathlib.Path(sys.argv[1]);dst.mkdir(exist_ok=True,parents=True)
    for pattern in ['*.md','*.png','*.json','*.csv']:
        for p in out.glob(pattern):shutil.copy2(p,dst/p.name)
