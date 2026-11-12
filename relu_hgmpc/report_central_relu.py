"""Summarize the ABBA closed-loop solver comparison without omitting setup."""
import json,pathlib,shutil,sys,platform
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from relu_solver import torch
import scipy,osqp,pinocchio

root=pathlib.Path(__file__).resolve().parents[1];src=root/'results/split_gmpc';out=root/'results/central_relu'
backends=['osqp','reluqp'];runs={};data={};aggregate={}
for backend in backends:
    names=[f'central_{backend}_12s_r{i}' for i in [1,2]]
    ss=[json.loads((src/f'{name}.json').read_text()) for name in names]
    xx=[np.genfromtxt(src/f'{name}.csv',delimiter=',',names=True) for name in names]
    for s in ss:
        assert s['failed'] is None and s['cycles']==600 and s['qp_converged']['mean']==1.,s
        assert s['args']['nodes']==10 and s['args']['step']==.03 and s['args']['scale']==1.
    assert ss[0]['source_sha256']==ss[1]['source_sha256']
    for name,s,x in zip(names,ss,xx):runs[name]=s;data[name]=x
    merged=np.concatenate(xx)
    result=dict(position_rmse_mm=float(np.sqrt(np.mean(merged['position_error_mm']**2))),
        orientation_rmse_deg=float(np.sqrt(np.mean(merged['rotation_error_deg']**2))),
        min_contacts=min(s['min_contacts'] for s in ss),missing_contact_ms=sum(s['missing_contact_ms'] for s in ss),
        torque_clips=sum(s['torque_clips'] for s in ss),max_slip_path_mm=max(s['max_slip_path_mm'] for s in ss),
        repeat_position_curve_max_difference_mm=float(np.max(np.abs(xx[0]['position_error_mm']-xx[1]['position_error_mm']))),
        repeat_orientation_curve_max_difference_deg=float(np.max(np.abs(xx[0]['rotation_error_deg']-xx[1]['rotation_error_deg']))),
        first_cycle_ms=[float(x['controller_ms'][0]) for x in xx])
    for key in ['controller_ms','model_ms','setup_ms','solve_ms','preprocessing_ms','kernel_setup_ms',
                'qp_primal_raw','qp_primal_relative','qp_dual_raw','qp_dual_relative','qp_normal_relative',
                'qp_iterations','qp_converged','interface_scaled','full_dynamics_max']:
        a=merged[key];result[key]=dict(mean=float(a.mean()),p95=float(np.quantile(a,.95)),max=float(a.max()))
    result['over_20ms_fraction']=float(np.mean(merged['controller_ms']>=20))
    aggregate[backend]=result

assert runs['central_osqp_12s_r1']['source_sha256']==runs['central_reluqp_12s_r1']['source_sha256']
verification=json.loads((out/'verification.json').read_text())
hardware=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
              numpy=np.__version__,scipy=scipy.__version__,osqp=osqp.__version__,pinocchio=pinocchio.__version__,
              python=platform.python_version(),platform=platform.platform())
comparison=dict(aggregate=aggregate,runs=runs,hardware=hardware,
   relu_to_osqp_total_time_ratio=aggregate['reluqp']['controller_ms']['mean']/aggregate['osqp']['controller_ms']['mean'],
   relu_to_osqp_solve_time_ratio=aggregate['reluqp']['solve_ms']['mean']/aggregate['osqp']['solve_ms']['mean'],
   verification=verification,
   timing_scope='feedback model/matrix construction, exact equality reduction and whitening (ReLU-QP), '
     'solver setup and CPU-GPU transfer, iterations, common residual checks, GPU-CPU retrieval and model diagnostics. '
     'Includes first-cycle CUDA startup inside feedback; excludes imports, truth acquisition and MuJoCo stepping. '
     'No computation delay injected into simulated plant.',
   solver_difference='Both solve the same 420-variable, 1280-row physical QP (240 equality rows). '
     'ReLU-QP uses an exactly equivalent 180-variable equality-nullspace parameterization and Hessian whitening; '
     'OSQP uses the unreduced QP. No constraints/objective weights are dropped or relaxed. '
     'Thus the solver/preprocessing packages differ; equality accuracy is not an intrinsic GPU benefit.')
(out/'comparison.json').write_text(json.dumps(comparison,indent=2))

colors=['#245a91','#d15f25'];labels=['OSQP','ReLU-QP (exact reduction)']
fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
keys=['position_error_mm','rotation_error_deg','qp_primal_relative','qp_dual_relative']
ylabels=['Position error (mm)','Orientation error (deg)','Common primal residual','Common stationarity residual']
for ax,key,yl in zip(axes.ravel(),keys,ylabels):
    for backend,label,color in zip(backends,labels,colors):
        x=data[f'central_{backend}_12s_r1'];ax.plot(x['time'],np.maximum(x[key],1e-16),label=label,color=color,lw=1.4)
    ax.set_xlabel('Simulation time (s)');ax.set_ylabel(yl);ax.grid(alpha=.25)
    if 'residual' in yl:ax.set_yscale('log');ax.axhline(2e-7,color='gray',ls='--',lw=1)
axes[0,0].legend(fontsize=9);fig.suptitle('Central GMPC: identical physical QP, different solver preprocessing')
fig.savefig(out/'tracking_residuals.png',dpi=170);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
for backend,label,color in zip(backends,labels,colors):
    x=data[f'central_{backend}_12s_r1'];axes[0].plot(x['time'],x['controller_ms'],label=label,color=color)
axes[0].axhline(20,color='gray',ls='--');axes[0].set_xlabel('Simulation time (s)');axes[0].set_ylabel('Full feedback time (ms)')
axes[0].set_title('Full time, including first cycle');axes[0].legend(fontsize=8);axes[0].grid(alpha=.25)
bottom=np.zeros(2)
for key,label,color in [('model_ms','Model / QP assembly','#b4c9da'),('setup_ms','Preprocess / setup / upload','#dd9a69'),('solve_ms','Solve / checks / download','#548c77')]:
    vals=np.array([aggregate[b][key]['mean'] for b in backends]);axes[1].bar(backends,vals,bottom=bottom,label=label,color=color);bottom+=vals
vals=np.array([aggregate[b]['controller_ms']['mean'] for b in backends])-bottom
axes[1].bar(backends,vals,bottom=bottom,label='Other diagnostics',color='#a1a2ae')
axes[1].axhline(20,color='gray',ls='--');axes[1].set_ylabel('Mean milliseconds / control cycle')
axes[1].set_title('Two runs per backend, all cycles');axes[1].legend(fontsize=8)
fig.suptitle('Setup remains online; simulation does not include computation delay')
fig.savefig(out/'timing.png',dpi=170);plt.close(fig)

lines=['# 集中式 GMPC：ReLU-QP 与 OSQP 闭环对照','',
 '2026-11-10；B2+Z1、Pinocchio、MuJoCo 真值。12 秒轨迹、10×30 ms 预测、20 ms 控制周期；按 OSQP/ReLU-QP/ReLU-QP/OSQP 顺序各跑两次，共 2400 个控制周期。','',
 '## 保持一致的内容','',
 '模型、几何误差、所有代价权重、约束、初始状态、参考轨迹、预测时域和原有控制序列移位热启动均不变。源码语法树核对 build_part/build/warm 与接入前完全一致；五个同状态快照的 H/g/A/l/u/E/d 逐元素一致。',
 '原 QP 有 420 个变量、1280 行约束，其中 240 行为等式。ReLU-QP 先精确消去等式，再对白化后的 180 维等价 QP 执行 GPU ReLU-QP 递推，最终恢复完整变量和乘子。没有放宽接触、接口、摩擦、关节或力矩约束，没有修改物理代价，也没有 OSQP 回退。',
 '当前 ReLU-QP 为工程中论文递推的实现，使用固定 rho=0.1、FP64；不是宣称完整复现上游自适应 rho 实现。两种后端的预处理不同，因此这是求解器及预处理组合的比较。','',
 '## 统一残差与计时口径','',
 '令 r=1/max(1,||A行||₂)，s_p=max(1,||rAx||∞,||r投影(Ax)||∞)，s_d=max(1,||Hx||∞,||g||∞,||Aᵀλ||∞)。原始可行性残差为 ||r(Ax-投影(Ax))||∞/s_p；驻点残差为 ||Hx+g+Aᵀλ||∞/s_d；另检查乘子符号与互补性的投影残差。三项均不大于 2e-7 才接受，超出预算即停止，不带病执行。OSQP 若内部停止但未满足统一标准，则继续收紧内部精度。',
 '反馈总时间包括模型/矩阵构造、等式消元及白化、求解器设置、CPU→GPU 传输、全部迭代、GPU→CPU 取回、统一残差和模型诊断。包括首周期 CUDA 初始化；不含程序导入、真值读取、MuJoCo 推进及记录导出。GPU 计时做同步。','',
 '## 实测结果','']
for backend in backends:
    s=aggregate[backend]
    lines += [f'**{backend.upper()}**','',
        f'- 位置 RMSE {s["position_rmse_mm"]:.6f} mm；姿态 RMSE {s["orientation_rmse_deg"]:.6f}°。',
        f'- 总耗时均值 {s["controller_ms"]["mean"]:.3f} ms，P95 {s["controller_ms"]["p95"]:.3f} ms；超过 20 ms 的比例 {100*s["over_20ms_fraction"]:.1f}%。',
        f'- 模型/矩阵构造 {s["model_ms"]["mean"]:.3f} ms；预处理/设置/上传 {s["setup_ms"]["mean"]:.3f} ms；求解/校核/取回 {s["solve_ms"]["mean"]:.3f} ms。',
        f'- 统一原始残差最大 {s["qp_primal_relative"]["max"]:.3g}；驻点残差最大 {s["qp_dual_relative"]["max"]:.3g}；法锥投影残差最大 {s["qp_normal_relative"]["max"]:.3g}；全部周期满足标准。',
        f'- 原始约束违反最大分量 {s["qp_primal_raw"]["max"]:.3g}（混合单位，仅作为数值诊断）；接口归一化残差最大 {s["interface_scaled"]["max"]:.3g}。',
        f'- 最少 {s["min_contacts"]} 足接触；累计缺失接触 {s["missing_contact_ms"]} ms；力矩截断 {s["torque_clips"]} 次；最大单次运行足端累计滑动 {s["max_slip_path_mm"]:.6f} mm。',
        f'- 两次运行位置误差曲线最大差 {s["repeat_position_curve_max_difference_mm"]:.3g} mm；首周期耗时 {s["first_cycle_ms"]} ms。','']
lines += ['## 解释与边界','',
 f'ReLU-QP 总耗时 / OSQP = {comparison["relu_to_osqp_total_time_ratio"]:.3f}；迭代求解部分耗时之比 = {comparison["relu_to_osqp_solve_time_ratio"]:.3f}。矩阵重建必须计入在线开销。',
 'ReLU-QP 等式残差很小主要来自精确等式消元，不能归因于 GPU 本身。有限容差和很弱的内部力正则化允许控制变量有小差异；完整约束、目标函数和末端跟踪均独立核对。',
 f'活跃不等式与冗余等式测试中，ReLU-QP 对高精度 OSQP 的最大变量误差 {max(t["control_error"] for t in verification["random_active_constraint_tests"]):.3g}；五个同 QP 快照的最大绝对目标差 {max(t["objective_gap"] for t in verification["snapshots"]):.3g}。',
 '只测试当前站立末端跟踪轨迹；两次重复不足以支持广泛统计显著性。仿真未注入计算延迟，闭环通过不代表满足 50 Hz 实时控制。','',
 '![跟踪与统一残差](tracking_residuals.png)','', '![总耗时与分解](timing.png)','',
 '## 复现','', '```bash','cd /home/ganhe/project/b2z1_mpc','source relu_hgmpc/env.sh',
 'python relu_hgmpc/test_central_relu.py','bash relu_hgmpc/benchmark_central_relu.sh','python relu_hgmpc/report_central_relu.py','```','',
 '源码：relu_hgmpc/central_relu.py、split_gmpc.py、run_split_gmpc.py。原始逐周期 CSV、摘要 JSON 和 1 kHz NPZ 在 results/split_gmpc/central_{osqp,reluqp}_12s_r{1,2}*；本报告与校核在 results/central_relu/。']
(out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if len(sys.argv)>1:
    dest=pathlib.Path(sys.argv[1]);dest.mkdir(parents=True,exist_ok=True)
    for name in runs:
        for ext in ['.json','.csv']:shutil.copy2(src/f'{name}{ext}',dest/f'{name}{ext}')
    for name in ['RESULTS.md','comparison.json','verification.json','tracking_residuals.png','timing.png']:
        shutil.copy2(out/name,dest/name)
print(json.dumps(dict(aggregate=aggregate,hardware=hardware),indent=2),flush=True)
