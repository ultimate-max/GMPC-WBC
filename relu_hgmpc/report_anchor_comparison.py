"""Actual measurements; failed runs are censored, never ranked as successes."""
import pathlib,json,csv,hashlib,platform,sys
import numpy as np
from scipy.signal import butter,sosfiltfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/anchored_mpc'
names=['single200_normal','single200_paper_shared','single200_paper','single100_normal',
       'single200_hold','single200_push','single200_double','single200_repeat']
labels=['Anchored GMPC + geometric WBC','Paper allocation + shared geometric WBC','Paper allocation + paper WBC (6D adaptation)']
def read(name):
    d=json.loads((out/f'{name}.json').read_text())
    with (out/f'{name}.csv').open() as f:r=list(csv.DictReader(f))
    return d,{key:np.array([float(x[key]) for x in r]) for key in r[0]},np.load(out/f'{name}_trace.npz')
data={name:read(name) for name in names if (out/f'{name}.json').exists()}
common={}
for name in names[:3]:
    d,r,z=data[name];mask=r['time']<8.;trace=z['trace'];tm=trace[:,0]<8.;tr=trace[tm];steady=tr[:,0]>=1.
    result=dict(interval_seconds=[0,8],jitter_interval_seconds=[1,8],
        ee_position_rmse_mm=float(np.sqrt(np.mean(r['ee_position_error_mm'][mask]**2))),
        ee_rotation_rmse_deg=float(np.sqrt(np.mean(r['ee_rotation_error_deg'][mask]**2))),
        base_position_rmse_mm=float(np.sqrt(np.mean(r['base_position_error_mm'][mask]**2))),
        total_cpu_ms=float(r['cpu_ms'][mask].sum()),
        cycle_ms_mean=float(r['total_ms'][mask].mean()),cycle_ms_p95=float(np.quantile(r['total_ms'][mask],.95)),
        foot_velocity_rms=float(np.sqrt(np.mean(tr[:,21:33]**2))),
        contact_loss_fraction=float(np.mean(tr[:,1:5]<.5)))
    for key,col in [('leg_velocity',slice(111,123)),('arm_velocity',slice(123,129)),('leg_torque',slice(57,69)),('arm_torque',slice(69,75))]:
        x=tr[:,col]
        result[key]={}
        for low in (5,10):
            y=sosfiltfilt(butter(4,[low,100],fs=1000,btype='bandpass',output='sos'),x,axis=0)
            result[key][f'hf_{low}_100_rms']=float(np.sqrt(np.mean(y[steady]**2)))
        result[key]['step_rms']=float(np.sqrt(np.mean(np.diff(x[steady],axis=0)**2)))
    flags=[p['base_active'] for p in d['plans'] if 'base_active' in p and p['time']<8]
    if flags:result['base_activation_switches']=int(np.count_nonzero(np.diff(flags)))
    common[name]=result

fig,axs=plt.subplots(3,1,figsize=(11,9),sharex=True,layout='constrained')
colors=['#1976bd','#df8b20','#777777']
for name,label,color in zip(names[:3],labels,colors):
    d,r,_=data[name]
    for ax,key in zip(axs,['ee_position_error_mm','base_position_error_mm','total_ms']):
        if key=='total_ms':
            idx=np.arange(0,len(r['time']),4);ax.plot(r['time'][idx],r[key][idx],color=color,lw=.7,alpha=.7,label=label)
        else:ax.plot(r['time'],r[key],color=color,lw=1.3,label=label)
        if d['failure']:ax.axvline(d['completed_seconds'],color=color,ls=':',alpha=.7)
axs[0].set_ylabel('External EE position error (mm)');axs[0].legend(fontsize=8)
axs[1].set_ylabel('Nominal base position error (mm)')
axs[2].set_ylabel('Serial replan-cycle time (ms)');axs[2].axhline(5,color='red',ls='--',lw=.8,label='200 Hz cycle budget')
axs[2].set_xlabel('Simulation time (s)')
for ax in axs:ax.grid(alpha=.2);ax.set_xlim(0,12)
fig.suptitle('Single-thread comparison; vertical dotted lines mark controller failures\nSimulation waits during solves: these are not real-time tracking results',fontsize=11)
fig.savefig(out/'singlethread_tracking.png',dpi=170);plt.close(fig)

fig,axs=plt.subplots(1,4,figsize=(12,3.7),layout='constrained')
for ax,key,title in zip(axs,['leg_velocity','arm_velocity','leg_torque','arm_torque'],
    ['Leg velocity (rad/s)','Arm velocity (rad/s)','Leg torque (N m)','Arm torque (N m)']):
    vals=[common[n][key]['hf_5_100_rms'] for n in names[:3]]
    bars=ax.bar(range(3),vals,color=colors)
    ax.set_xticks(range(3),['GMPC','SNS + GWBC','Paper 6D'],rotation=25,ha='right');ax.set_title(title,fontsize=10)
    for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,b.get_height(),f'{v:.3g}',ha='center',va='bottom',fontsize=8)
    ax.set_ylim(0,max(vals)*1.2);ax.grid(axis='y',alpha=.2)
fig.suptitle('Actual actuator signals: 5–100 Hz RMS on the common 1–8 s interval\nBand energy includes task transients; it is not proof of selection-switch chatter',fontsize=10)
fig.savefig(out/'singlethread_actuators.png',dpi=170);plt.close(fig)

result=dict(common_prefix=common,full_runs={name:{k:v for k,v in d.items() if k not in ('plans','source_sha256','schedule_events')} for name,(d,_,_) in data.items()})
for name,(d,r,z) in data.items():
    if d['args']['method']=='anchor':
        result['full_runs'][name]['mpc_original_kkt_max']=max(p.get('hierarchy_kkt',0.) for p in d['plans'])
        result['full_runs'][name]['mpc_lock_max']=max(p.get('hierarchy_lock',0.) for p in d['plans'])
        result['full_runs'][name]['arm_weight_range']=[min(p.get('arm_motion_weight',0.) for p in d['plans']),max(p.get('arm_motion_weight',0.) for p in d['plans'])]
        result['full_runs'][name]['base_preference_gate_max']=max(p.get('base_nominal_gate',0.) for p in d['plans'])
(out/'comparison_summary.json').write_text(json.dumps(result,indent=2))

lines=['# 单线程末端锚定 GMPC＋几何 WBC：实现与实测',
       '', '本报告对应 B2＋Z1 的 6D 站立仿真；使用 MuJoCo 真值、Pinocchio 动力学。原论文是运动分配＋WBC，不是 MPC；本文的 paper 结果是对其方程的 B2＋Z1、6D 适配，不能代替作者代码复现，也不能据此认定原论文方法本身失败。',
       '', '论文来源：[Xie 等，2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12087763/)。',
       '', '## 已实现的联合任务',
       '', '安装关系为 Y=T_WB T_BA，机械臂相对运动为 F=T_AE(q_a)，目标为 D=T_WE^d。反推安装基座目标 Y_req=D F^-1，并使用同一个闭合条件 Y=Y_req；不会给底盘和机械臂各自指定一个互不相关的终点。',
       '', '原始接口误差 e_I=Log(Y_req^-1 Y)=Ad_F Log(D^-1 YF)。F 随机械臂构型变化，导数必须包含这一项。本实现使用拉回到统一末端度量的误差 r=Ad_F^-1 e_I，因此和直接的 SE(3) 末端误差严格等价。换对接位置本身不构成性能优势。',
       '', 'MPC 层级：硬物理约束 > 末端位姿/误差速度 > 连续运动分配＋柔性底盘姿态＋局部避奇异偏好 > 姿态/加速度/力矩平滑。WBC 层级：硬物理约束 > 规划末端 > 规划底盘 > 关节与力矩平滑。WBC 跟踪一致的规划参考，但全部结果对外部原始末端目标统计。',
       '', '分配使用归一化底盘增量与机械臂增量的代价。机械臂权重由 Jacobian 最小奇异值平滑调整，范围 0.2–0.8；关节限位通过硬约束限制。末端误差足够小后，底盘回正权重平滑降至零；仍保留运动抑制与必要的避奇异偏好。这里不是“谁先到就硬切掉谁”的逻辑。',
       '', '可操作度使用正则化 log-volume 的局部一阶偏好，不是严格的远离奇异点保证。所有动态权重、门函数与尺度均在 anchored_mpc.py 中明示。',
       '', '## 单线程调度',
       '', '主循环先运行 WBC，再运行到期的 MPC。MPC 只能在整数个 WBC tick 上触发。默认 WBC 200 Hz、MPC 50 Hz（4 分频），WBC 100 Hz 对应 2 分频。保留用户要求的 MPC 最低 50 Hz；预算模式用耗时 P95 决定分频并使用滞回，但不会自动降到 50 Hz 以下。',
       '', '预测规模固定为 10×40 ms；改变执行频率不会改变预测时域。约束、模型、求解精度保持一致。串行总耗时包含状态读取、参考生成、QP 矩阵构造、分解/求解、认证和轨迹恢复。目标缓存与模型初始化单列在计时范围外。',
       '', '**这些闭环使用受控仿真时钟，求解时仿真会等待。降低 MPC 调用频率不能消除单次求解对 WBC 的阻塞；以下超时统计不等于实际机器人延迟实验。当前版本不能宣称达到 200 Hz WBC 的硬实时要求。**',
       '', '## 完整运行结果（目标均为 12 秒）', '']
for name,(d,r,z) in data.items():
    status='完成' if not d['failure'] else '停止：'+d['failure']['reason']
    lines.append(f"- {name}：{d['completed_seconds']:.3f} s，{status}。末端位置 RMS={d['ee_position_error_mm']['rms']:.3f} mm，姿态 RMS={d['ee_rotation_error_deg']['rms']:.3f}°。失败运行的误差只覆盖已完成片段，不能与完整运行直接排名。")
lines+=['','## 相同时间段对照','', '正常工况统一取 0–8 s；实际关节信号的频带指标统一取 1–8 s，避免把失败后的缺失数据当成零误差。这是用于定位差异的共同前缀，不能掩盖后续失败。','']
for name,label in zip(names[:3],labels):
    r=common[name]
    lines.append(f"- {label}：末端位置 RMS {r['ee_position_rmse_mm']:.3f} mm，姿态 RMS {r['ee_rotation_rmse_deg']:.3f}°；总控制 CPU {r['total_cpu_ms']/1000:.3f} s；每周期总耗时均值/P95 {r['cycle_ms_mean']:.3f}/{r['cycle_ms_p95']:.3f} ms。")
    lines.append(f"  实际机械臂速度/力矩的 5–100 Hz RMS：{r['arm_velocity']['hf_5_100_rms']:.6f} rad/s / {r['arm_torque']['hf_5_100_rms']:.6f} N·m。实际腿速度/力矩：{r['leg_velocity']['hf_5_100_rms']:.6f} rad/s / {r['leg_torque']['hf_5_100_rms']:.6f} N·m。")
    if 'base_activation_switches' in r:lines.append(f"  该区间记录到底盘启停切换 {r['base_activation_switches']} 次；这不是完整选择矩阵所有元素的切换计数。")
lines+=['','![跟踪与耗时](singlethread_tracking.png)','','![实际关节信号](singlethread_actuators.png)',
       '', '## 基线适配与结论边界','',
       '- paper：保留论文的机械臂优先、基座激活、饱和重分配、零空间可操作度和原 WBC 方程。分配 50 Hz，WBC 200 Hz；已验证抽取 allocate() 前后原控制输出完全一致。原 WBC 的基座加速度修正可能改变此前接触任务结果，相关接触残差有单独记录。',
       '- paper_shared：上述运动分配接入同一个末端优先几何 WBC，增加发布时刻参考补偿和有界加速度的速度指令转换，属于明确标注的适配实验。它和原 WBC 基线共同报告，不能只选较差的一组比较。',
       '- MPC 和瞬时运动分配优化的是不同问题，耗时比较是完整控制栈成本，不是同一个 QP 的求解器加速比。',
       '- 原论文基线没有被要求跟踪这里的名义底盘轨迹。图中的底盘偏差用于展示运动分配结果，不能直接作为该基线跟踪失败的证据。耗时曲线显示每四个周期抽取的重规划周期；均值统计包含全部 WBC 周期。',
       '- 在共同前 8 秒，MPC 的末端误差优于共用几何 WBC 的运动分配适配版，但劣于原 WBC 的论文 6D 适配版；MPC 的实际速度和力矩频带分量也没有普遍减小。因此本次试验没有证明全面性能优势。',
       '- 共同前缀中的频带 RMS 不是纯粹的“选择矩阵抖振”。控制增益、预测、插值、物理约束和低层控制都可能影响它；还需选择矩阵开关的单因素消融。',
       '- 候选 MPC 在双幅工况仍有层级 QP 数值失败。已证明其中一个被报告不可行的低层子问题存在可行零增量；1e-10 原单位的舍入处理可恢复该点，但后续点仍会失败。始终使用原始目标/约束的 2e-7 KKT 与任务锁标准验收，未发布不合格解。',
       '- 历史双线程测试失败和迟到丢弃测试都保留；按照最新要求，推荐入口已经改为单线程。',
       '- 当前结果支持继续研究连续冗余分配与预测控制，不能据此宣称 GMPC 已全面优于论文、严格避奇异、必然抑制全部关节抖振，或满足硬实时。',
       '', '## 运行入口','', '工程路径：/home/ganhe/project/b2z1_mpc。',
       '', '```bash','cd /home/ganhe/project/b2z1_mpc','source relu_hgmpc/env.sh',
       'python relu_hgmpc/run_singlethread_anchor.py --method anchor --case normal --wbc-hz 200 --schedule budget --duration 12 --name my_anchor',
       'python relu_hgmpc/run_singlethread_anchor.py --method paper_shared --case normal --wbc-hz 200 --schedule ratio --duration 12 --name my_paper_shared',
       'python relu_hgmpc/run_singlethread_anchor.py --method paper --case normal --wbc-hz 200 --schedule ratio --duration 12 --name my_paper',
       '```','', '原始日志、CSV、1 kHz 实际速度/施加力矩、源文件哈希、失败时刻均在 results/anchored_mpc；comparison_summary.json 提供可机器读取的汇总。']
(out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
import pinocchio,mujoco,osqp,scipy
passport=dict(platform=platform.platform(),python=sys.version,packages={k:v.__version__ for k,v in [('pinocchio',pinocchio),('mujoco',mujoco),('osqp',osqp),('numpy',np),('scipy',scipy)]},
    project=str(root),model_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'models').glob('*') if p.is_file()},
    validations={n:json.loads((out/n).read_text()) for n in ('test_geometry.json','test_integration.json')},
    hardware='simulation only; configurable fixed mount; simulator truth; no physical robot actuation')
(out/'MATERIAL_PASSPORT.json').write_text(json.dumps(passport,indent=2))
print(json.dumps(dict(common_prefix=common,selected={n:{k:d[k] for k in ('completed_seconds','ee_position_error_mm','upper_total_ms','singlethread_timing')} for n,(d,_,_) in data.items()}),indent=2))
