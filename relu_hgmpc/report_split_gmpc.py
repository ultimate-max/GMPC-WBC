import pathlib,json,sys,shutil
import numpy as np
from scipy.signal import welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/split_gmpc'
names=['central_12s','distributed30_12s','distributed100_12s','distributed30_60s','distributed30_scale2']
summaries={n:json.loads((out/f'{n}.json').read_text()) for n in names if (out/f'{n}.json').exists()}
data={n:np.genfromtxt(out/f'{n}.csv',delimiter=',',names=True) for n in summaries}
actual={}
for name,s in summaries.items():
    z=np.load(out/f'{name}_trace.npz')['trace'];assert z.shape[1]==129
    metrics={}
    for label,sl in [('legs',slice(0,12)),('arm',slice(12,18))]:
        v=z[:,111:129][:,sl];u=z[:,57:75][:,sl]
        def band(x):
            f,P=welch(x[500:],fs=1000,nperseg=2048,noverlap=1024,axis=0)
            m=(f>=10)&(f<=100)
            return float(np.sqrt(np.trapezoid(P[m],f[m],axis=0).mean()))
        metrics[label]=dict(velocity_increment_rms=float(np.sqrt(np.mean(np.diff(v,axis=0)**2))),
            torque_increment_rms=float(np.sqrt(np.mean(np.diff(u,axis=0)**2))),
            velocity_band_10_100_hz=band(v),torque_band_10_100_hz=band(u))
    actual[name]=metrics
    s['measured_1khz']=metrics
(out/'actual_metrics.json').write_text(json.dumps(actual,indent=2))
(out/'comparison.json').write_text(json.dumps(summaries,indent=2))
labels=['Central GMPC','Distributed GMPC: 30 rounds','Distributed GMPC: 100 rounds']
colors=['#253c78','#d26628','#36866b']
fig,axes=plt.subplots(4,1,figsize=(11,10),sharex=True,layout='constrained')
for n,label,color in zip(names[:3],labels,colors):
    x=data[n];z=np.load(out/f'{n}_trace.npz')['trace']
    axes[0].plot(x['time'],x['position_error_mm'],label=label,color=color)
    axes[1].plot(x['time'],x['rotation_error_deg'],color=color)
    axes[2].step(z[:,0],(z[:,1:5]>0).sum(axis=1),where='post',color=color)
    axes[3].semilogy(x['time'],np.maximum(x['interface_scaled'],1e-10),color=color)
for a,label in zip(axes,['Position error (mm)','Orientation error (deg)','Contacting feet (1 kHz)','Scaled interface residual']):
    a.set_ylabel(label);a.grid(alpha=.2)
axes[0].legend();axes[2].set_ylim(3.5,4.5);axes[-1].set_xlabel('Simulation time (s)')
fig.suptitle('Predictive base/legs + arm GMPC: same dynamics, objectives and constraints')
fig.savefig(out/'split_tracking.png',dpi=170);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
for ax,key,title in zip(axes,['controller_ms','dual_residual'],['End-to-end feedback computation','ADMM dual residual']):
    for n,label,color in zip(names[:3],labels,colors):
        if key=='dual_residual' and n=='central_12s':continue
        ax.plot(data[n]['time'],data[n][key],label=label,color=color)
    ax.set_title(title);ax.set_xlabel('Simulation time (s)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    ax.set_ylabel('ms' if key=='controller_ms' else 'Scaled residual')
    if key=='controller_ms':ax.axhline(20,color='gray',ls='--',label='20 ms period')
    else:ax.set_yscale('log');ax.axhline(1e-4,color='gray',ls='--')
fig.suptitle('Finite iteration and compute limits remain visible; simulated time excludes latency')
fig.savefig(out/'split_compute_residuals.png',dpi=170);plt.close(fig)

verification=json.loads((out/'verification.json').read_text())
lines=['# 完整底盘＋机械臂分布式 GMPC：实测结果','',
 '2026-11-10。B2+Z1、Pinocchio、MuJoCo 真值；10×30 ms 预测、20 ms 控制、1 kHz 接触/执行力矩/关节速度记录。',
 '已从瞬时速度分配扩展为两个局部动力学 OCP，优化腿/臂力矩、地面接触力与显式接口力，通过共识 ADMM 协调共同机身运动和作用力。每周期重线性化，预测域内冻结动力学系数；并非多次非线性 SQP。当前采用加权任务代价，未声称严格词典序优先级。','',
 '## 同一优化问题的对照','']
for name in names:
    if name not in summaries:continue
    s=summaries[name]
    lines += [f'**{name}**：完成 {s["cycles"]} 周期，失败记录 {s["failed"]}。','',
        f'- 位置 RMSE {s["position_rmse_mm"]:.4f} mm；姿态 RMSE {s["rotation_rmse_deg"]:.5f}°。',
        f'- 最少 {s["min_contacts"]} 足接触，少于四足累计 {s["missing_contact_ms"]} ms；最大单足累计滑动 {s["max_slip_path_mm"]:.5f} mm；力矩截断 {s["torque_clips"]} 次。',
        f'- 反馈总耗时平均 {s["controller_ms"]["mean"]:.2f} ms，P95 {s["controller_ms"]["p95"]:.2f} ms。',
        f'- 最大接口归一化残差 {s["interface_scaled"]["max"]:.3g}；最大对偶残差 {s["dual_residual"]["max"]:.3g}（集中式此字段不适用）。',
        f'- 冻结完整模型动力学最大分量残差 {s["full_dynamics_max"]["max"]:.3g}；接口净力最大分量 {s["wrench_force_max"]["max"]:.3g} N；接口净力矩最大分量 {s["wrench_moment_max"]["max"]:.3g} Nm。']
    for group in ['legs','arm']:
        m=actual[name][group]
        lines += [f'- {group} 实测 1 ms 速度增量 RMS {m["velocity_increment_rms"]:.6g} rad/s；执行力矩增量 RMS {m["torque_increment_rms"]:.6g} Nm；10–100 Hz 速度 RMS {m["velocity_band_10_100_hz"]:.6g} rad/s。']
    if 'nonlinear_base_force_max' in s:
        lines += [f'- 预测轨迹上重新计算非线性动力学：基座力最大残差 {s["nonlinear_base_force_max"]["max"]:.4g} N、基座力矩 {s["nonlinear_base_moment_max"]["max"]:.4g} Nm、关节力矩 {s["nonlinear_joint_torque_max"]["max"]:.4g} Nm。这些是冻结模型误差，不能用很小的 ADMM 接口残差替代。',
                  f'- 同时达到原始/对偶停止条件的周期比例 {s["admm_converged"]["mean"]*100:.2f}%。']
    lines+=['']
lines += ['## 校核与结论边界','',
 f'惯量/偏置分解最大误差 {verification["split_model_max_error"]:.3g}；几何误差雅可比有限差分误差 {verification["geometric_jacobian_error"]:.3g}；集中式与充分迭代 ADMM 的绝对目标差 {verification["objective_gap"]:.3g}。',
 '30 次和 100 次版本的跟踪接近集中式，不能据此把有限迭代当作完全收敛。两种分布式设置的反馈耗时都超过 20 ms，也没有超过集中式的计算速度。计时为本轮串行运行各配置的初步实测，非重复统计基准；模型构造、求解器设置、全部协调轮次及诊断均计入。',
 '仿真未注入计算延迟，因此通过闭环不能证明在当前计算平台上可实现 50 Hz 控制。实际波形记录来自 MuJoCo，当前没有硬件实验。',
 '新 MPC 与速度层 WBC 的关节参考生成方式不同。速度层三种分配的对照用于隔离切换机制，新 MPC 对照用于核实预测与分布式求解；不将跨架构的误差或抖振差异单独归因于 ADMM 或 Lie 群。',
 '完整 GMPC 的机械臂执行力矩增量 RMS 约 0.0564 Nm，低于原硬切换的 0.1221 Nm，但高于连续速度层原型的 0.0210 Nm；位置 RMSE 约 0.330 mm，也高于速度层原型的 0.120 mm。扩展已跑通，并未证明新架构在所有指标上更优。',
 '本次局部 QP 使用 OSQP；旧 ReLU-QP 基线保持独立。虚拟子系统的公共机身惯量分配方式、完整公式、求解停止条件和运行命令见 SPLIT_GMPC.md。','',
 '![跟踪与接触](split_tracking.png)','',
 '![计时与残差](split_compute_residuals.png)']
if 'interface_kkt_acceleration_error' in verification:
    lines.insert(lines.index('## 校核与结论边界')+2,f'显式接口 KKT 与完整模型加速度最大误差 {verification["interface_kkt_acceleration_error"]:.3g}。')
(out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
shutil.copy2(root/'relu_hgmpc/SPLIT_GMPC.md',out/'SPLIT_GMPC.md')
if len(sys.argv)>1:
    dst=pathlib.Path(sys.argv[1]);dst.mkdir(parents=True,exist_ok=True)
    for f in out.iterdir():
        if f.suffix in ('.md','.json','.png','.csv'):shutil.copy2(f,dst/f.name)
print(json.dumps({n:dict(position_mm=s['position_rmse_mm'],contacts=s['min_contacts'],
    missing_ms=s['missing_contact_ms'],time_ms=s['controller_ms']['mean'],actual=actual[n]) for n,s in summaries.items()},indent=2))
