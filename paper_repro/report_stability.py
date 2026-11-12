from pathlib import Path
import json,hashlib,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];D=R/'results/interface_audit';P=R/'results/paper_repro'
def read(path): return np.atleast_1d(np.genfromtxt(path,delimiter=',',names=True))
def summarize(a):
    return dict(samples=len(a),rmse_mm=float(np.sqrt(np.mean(a['error_m']**2))*1000),max_error_mm=float(a['error_m'].max()*1000),max_foot_drift_mm=float(a['foot_drift_m'].max()*1000),max_tilt_deg=float(np.rad2deg(a['tilt_rad'].max())),four_contacts_fraction=float(np.mean(a['contacts']==4)),solver_failures=int(a['solver_failures'].sum()),torque_clips=int(a['torque_clips'].sum()),interface_l2_max=float(a['interface_l2'].max()),interface_max_pair_node=float(a['interface_max'].max()),full_defect_l2_max=float(a['full_defect_l2'].max()))
summary={}
for name in ['stable_repeat1','stable_repeat2','stable_repeat3','stable_serial','stable_long60']:
    path=D/f'{name}.csv'
    if not path.exists(): shutil.copy(P/f'{name}.csv',path)
    a=read(path);stats=summarize(a)
    expected=3000 if name=='stable_long60' else 300 if name=='stable_serial' else 600
    assert stats['samples']==expected and stats['max_error_mm']<5 and stats['max_foot_drift_mm']<5 and stats['max_tilt_deg']<3 and stats['four_contacts_fraction']==1 and stats['solver_failures']==stats['torque_clips']==0,(name,stats)
    summary[name]=stats
a=read(D/'stable_repeat1.csv');b=read(D/'stable_serial.csv');fields=[f'q{i}' for i in range(25)]+['error_m','interface_l2','full_defect_l2']
summary['serial_parallel_max_abs']=max(float(np.max(np.abs(a[f][:len(b)]-b[f]))) for f in fields)
for mode in ['central','distributed']:
    paths=sorted(D.glob(f'latency_{mode}_*.csv'))
    if not paths: continue
    x=np.concatenate([read(p) for p in paths]);m=summarize(x)
    assert len(paths)==3 and m['samples']==900 and m['solver_failures']==m['torque_clips']==0 and m['four_contacts_fraction']==1,(mode,m)
    for key in ['controller_ms','max_local_solve_ms','update_ms','parallel_wall_ms','critical_lin_ms','critical_qp_ms']:
        m[key]=dict(mean=float(x[key].mean()),p95=float(np.percentile(x[key],95)))
    m['deadline_miss_fraction']=float(np.mean(x['controller_ms']>20));summary['latency_'+mode]=m
(D/'stability_summary.json').write_text(json.dumps(summary,indent=2))
for name in ['INTERFACE_AUDIT.md','runtime.cpp','verify_interfaces.py','validate_stability.sh','compare_stable.sh','report_stability.py']:
    shutil.copy(R/'paper_repro'/name,D/name)
(D/'source_sha256.json').write_text(json.dumps({str(f.relative_to(R)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (R/'paper_repro').glob('*') if f.is_file()},indent=2))
long=read(D/'stable_long60.csv');fig,axs=plt.subplots(2,2,figsize=(10,6),layout='constrained')
for ax,key,factor,label in zip(axs.flat,['error_m','foot_drift_m','interface_l2','full_defect_l2'],[1000,1000,1,1],['Tool position error (mm)','Maximum foot displacement (mm)','Interface velocity residual, L2','Full-model prediction defect, L2']):
    ax.plot(long['time'],factor*long[key]);ax.set(xlabel='Simulation time (s)',ylabel=label);ax.grid(alpha=.2)
fig.suptitle('Distributed B2 + Z1: 60 s stable simulation\n10 coordination steps, primal step 0.2, dual step 0.05')
fig.savefig(D/'stable60.png',dpi=180);plt.close(fig)
m=summary['stable_long60'];timing=''
if 'latency_central' in summary and 'latency_distributed' in summary:
    c=summary['latency_central'];d=summary['latency_distributed']
    timing=f"""\n## 稳定后重新计时\n\n在构建与其他仿真全部结束后，交替运行三次各 6 秒。集中式平均总耗时 {c['controller_ms']['mean']:.2f} ms（P95 {c['controller_ms']['p95']:.2f} ms），分布式稳定配置为 {d['controller_ms']['mean']:.2f} ms（P95 {d['controller_ms']['p95']:.2f} ms）。后者包括全部十次局部求解、参数交换和合并，没有只计最后一次迭代。\n\n集中式跟踪 RMSE 为 {c['rmse_mm']:.3f} mm，分布式为 {d['rmse_mm']:.3f} mm；预测缺陷也不同，因此这是一组明确配置的运行成本对照，不是同等最优性精度的求解器速度结论。当前稳定方案没有复现论文的加速结论，也没有达到 20 ms 实时预算。\n"""
report=f"""# 接口与 ADMM 排查结果\n\n分布式闭环已在本任务中跑稳定，但尚未复现论文单次完整更新的结果。完整推导与测试依据见 [INTERFACE_AUDIT.md](INTERFACE_AUDIT.md)。\n\n## 核实与修复\n\n1. 接口消元按带符号关联矩阵独立重建。24 组非一致状态的 Schur/KKT 对照误差约 5e-13，没有发现需要整体反转接口力符号的证据。\n2. 改为两两接口速度残差对应的聚合边乘子更新；明确采用 r+d 的平方项与 d+=r 的一致符号。\n3. 修复终端对偶错误：终端无一致性罚项，乘子却被更新并向有效预测区间复制。现在终端及移位后无前驱约束的尾部乘子置零，实际 C++ 回归测试通过。\n4. 修复不是唯一充分条件。完整步长单次更新仍失败。稳定配置采用原始步长 0.2、对偶步长 0.05、每周期 10 次并行协调；对偶更新保持开启，没有集中式回退或固定机身。\n\n## 闭环验证\n\n通过三次 12 秒、一次 6 秒串行对照和一次 60 秒长测。长测 {m['samples']} 个控制周期：\n\n- 末端位置 RMSE **{m['rmse_mm']:.3f} mm**，最大 **{m['max_error_mm']:.3f} mm**。\n- 所有控制采样点保持四足接触；最大足端位移 **{m['max_foot_drift_mm']:.3f} mm**，最大机身倾角 **{m['max_tilt_deg']:.4f}°**。\n- 求解失败 **{m['solver_failures']}** 次，力矩截断 **{m['torque_clips']}** 次。\n- 接口整体 L2 最大 {m['interface_l2_max']:.6g}，单个子系统对/节点最大 {m['interface_max_pair_node']:.6g}。完整动力学预测缺陷 L2 最大 {m['full_defect_l2_max']:.6g}，有限迭代的预测仍有误差，不能宣称完全收敛。\n- 串行与并行的状态、误差及残差记录最大绝对差为 {summary['serial_parallel_max_abs']:.3g}。\n\n以上是具体仿真范围内的实证结果，不是闭环稳定性定理。预测节点仍为 50、步长 10 ms、仿真控制周期 20 ms；仿真未注入计算延迟。\n{timing}\n## 运行\n\n```bash\ncd /home/ganhe/project/b2z1_mpc\nsource paper_repro/env.sh\npaper_repro/build/paper_rti \"$PWD\" --mode distributed --profile stable --duration 60 --name my_stable_run\n```\n\n`--profile single_step` 保留完整步长、一次协调的未通过配置，用于继续研究，不能把它与稳定配置混称。所有公式测试、回归测试、原始日志、稳定性门槛与计时边界均保留在工程中。\n"""
(D/'RESULTS.md').write_text(report)
print(json.dumps(summary,indent=2))
