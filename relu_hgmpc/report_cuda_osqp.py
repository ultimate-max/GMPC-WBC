"""Report matched short closed loops and final FP64 CUDA replay audit."""
import pathlib,json,csv,hashlib,shutil,sys,subprocess
import numpy as np
import osqp_cuda
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/cuda_osqp'
audit=json.loads((out/'verification.json').read_text())
assert len(audit['records'])==10 and all(r['passed'] for r in audit['records'])
assert audit['cuda_use_float']==0
assert audit['cuda_binary_sha256']==hashlib.sha256(pathlib.Path(osqp_cuda.__file__).read_bytes()).hexdigest()
groups={};traces={}
for scale in (1,2):
    for alg in ('builtin','cuda'):
        name=f'{alg}_scale{scale}_2s';d=json.loads((out/f'{name}.json').read_text())
        assert d['failed'] is None and d['simulated_cycles']==100
        assert d['min_contacts']==4 and d['torque_clips']==0
        assert max(d[k]['max'] for k in ('qp_primal_relative','qp_dual_relative','qp_normal_relative'))<=2e-7
        assert d['task_lock_relative']['max']<=2e-7
        with (out/f'{name}.csv').open() as f:rows=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
        traces[name]=rows
        groups[name]=dict(position_rmse_mm=d['position_rmse_mm'],rotation_rmse_deg=d['rotation_rmse_deg'],
            feedback_mean_ms=d['amortized_feedback_ms'],feedback_p95_ms=d['feedback_ms']['p95'],feedback_p99_ms=d['feedback_ms']['p99'],
            feedback_max_ms=d['feedback_ms']['max'],residual_max=max(d[k]['max'] for k in ('qp_primal_relative','qp_dual_relative','qp_normal_relative')),
            task_lock_relative_max=d['task_lock_relative']['max'],min_contacts=d['min_contacts'],torque_clips=d['torque_clips'],
            primary_iterations=d['primary_iterations']['mean'],secondary_iterations=d['secondary_iterations']['mean'],
            primary_solver_api_ms=d['primary_solver_api_ms']['mean'],secondary_solver_api_ms=d['secondary_solver_api_ms']['mean'],
            primary_prepare_ms=d['primary_setup_ms']['mean'],secondary_prepare_ms=d['secondary_setup_ms']['mean'],
            reference_controller_initialization_ms=d['precompute_ms'],osqp_setups=d['osqp_setups'],pattern_growths=d['pattern_growths'])
for scale,index in [(1,0),(1,80),(1,200),(1,490),(2,200)]:
    a=np.load(out/f'replay_s{scale}_i{index}_builtin_primary.npz');b=np.load(out/f'replay_s{scale}_i{index}_cuda_primary.npz')
    assert all(np.array_equal(a[k],b[k]) for k in ('H','g','A','l','u'))
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv'],text=True)
summary=dict(groups=groups,replay=audit,primary_replay_matrices='bitwise identical',gpu=gpu,
    scope='Four 2-second runs (one per backend/scale), 400 control cycles and 8000 low-level samples. Five replay states per backend. Not a 12-second/full-trajectory benchmark; no compute-delay injection.',
    timing='State read, model, matrix construction, both QPs, explicit CUDA synchronization, host/device transfer, multiplier recovery and original-coordinate checks; startup amortized. Simulator advance excluded.',
    precision='CPU FP64 and locally built CUDA FP64. Original-coordinate KKT and task-lock tolerance 2e-7.',
    modifications='Official OSQP core v1.0.0 + Python v1.1.3. OSQP_USE_FLOAT=OFF; CG tolerance floor 1e-7->1e-12; reference-counted global CUDA library handle for serial multiple workspaces.',
    sources={f:hashlib.sha256((root/'relu_hgmpc'/f).read_bytes()).hexdigest() for f in
        ['osqp_acceleration.py','hierarchical_gmpc.py','run_hierarchical_gmpc.py','env_cuda_osqp.sh','cuda_fp64_precision.patch']})
(out/'comparison.json').write_text(json.dumps(summary,indent=2))
lines=['# CUDA OSQP 接入层级 GMPC 的试验','',
    '已接入官方 CUDA 后端的本地双精度构建，完成相同模型、约束、时域、物理权重和验收门槛的短闭环比较。CPU 仍为默认后端。',
    '', '环境：WSL2、RTX 4060 Laptop 8GB、驱动555.97、CUDA 12.1运行库，OSQP Python/CUDA扩展1.1.3、核心1.0.0。每次2秒、100个20ms控制周期、2000个1ms物理样本，N=10、dt=30ms。两档幅值、两个后端各一次，共400个控制周期。',
    '', '## 闭环结果','']
for scale in (1,2):
    lines += [f'### {"普通" if scale==1 else "双幅"}参考','']
    for alg,label in [('builtin','CPU'),('cuda','GPU')]:
        d=groups[f'{alg}_scale{scale}_2s']
        lines += [f'- {label}：位置RMSE {d["position_rmse_mm"]:.8f} mm，姿态RMSE {d["rotation_rmse_deg"]:.8f}°；完整反馈均值 {d["feedback_mean_ms"]:.3f} ms，P95 {d["feedback_p95_ms"]:.3f} ms，P99 {d["feedback_p99_ms"]:.3f} ms。',
            f'  - 两层外层ADMM平均迭代总数（含精化）{d["primary_iterations"]:.1f}/{d["secondary_iterations"]:.1f}；求解API墙钟时间（含同步/返回）{d["primary_solver_api_ms"]:.3f}/{d["secondary_solver_api_ms"]:.3f} ms；准备/更新API及格式构造 {d["primary_prepare_ms"]:.3f}/{d["secondary_prepare_ms"]:.3f} ms。',
            f'  - 原问题相对残差最大 {d["residual_max"]:.3e}，完整主任务保持误差最大 {d["task_lock_relative_max"]:.3e}。']
    a=groups[f'builtin_scale{scale}_2s'];b=groups[f'cuda_scale{scale}_2s']
    lines += [f'GPU总耗时为CPU的 {b["feedback_mean_ms"]/a["feedback_mean_ms"]:.2f} 倍；位置RMSE相对变化 {100*(b["position_rmse_mm"]/a["position_rmse_mm"]-1):+.5f}%，姿态RMSE变化 {100*(b["rotation_rmse_deg"]/a["rotation_rmse_deg"]-1):+.5f}%。','']
lines += ['所有400个周期通过共同的2e-7验收，四足持续接触，无力矩裁剪。跟踪结果是在未注入计算延迟的仿真中得到，不能解释为这些计算时延下的实时跟踪性能。',
    '', '短闭环仅用于确认接入、精度和当前规模的计算开销；没有完成GPU的12秒完整轨迹验证，也不构成统计鲁棒性实验。普通/双幅参考的后段困难状态另由回放覆盖，但回放不能替代闭环。',
    '', '## 回放与必要修复','',
    '回放使用普通轨迹0、1.6、4、9.8秒及双幅轨迹4秒的五个状态。CPU和GPU第一层H/g/A/l/u逐位相同；最终构建十次求解全部通过原问题KKT和主任务保持检查。',
    f'CPU与GPU首步力矩最大差异 {max(r.get("first_torque_difference_Nm",0) for r in audit["records"]):.6g} N·m。回放每次从新工作区开始，其耗时不能当作闭环平均时延。','',
    '使用官方预编译包时发现其采用FP32，因此为对齐CPU构建了FP64。即使使用FP64，官方内层CG精度下限1e-7仍使部分运动状态的原问题驻点残差停在约1e-6；保留默认设置的回放记录于verification_fp64_stock_cg.json。将下限收紧为1e-12后保持外部2e-7门槛不变。',
    '', '闭环又暴露多工作区生命周期问题：任一工作区清理会释放仍被另一层使用的全局CUDA句柄。增加串行工作区引用计数后，矩阵重建、更新和连续闭环均通过。补丁见cuda_fp64_precision.patch，崩溃栈见crash_before_lifetime_fix.log。故最终结果对应明确记录的本地修补版本，不能称为未经修改的官方预编译包。','',
    '## 为什么没有加速','',
    '本项目两层分别只有180、120个变量，却每周期更新数值矩阵。CUDA使用内层共轭梯度及外层ADMM，CPU builtin使用直接线性求解。实测GPU同时承受更多迭代与更高的API执行开销；小矩阵、GPU内核启动/同步、传输和缩放/预条件都可能影响结果。现有计时不足以将差额全部归因于单一因素，GPU同步和传输已计入完整时间。',
    '', '普通参考下，GPU两层solve API合计约2.10秒，占完整反馈时间约97.6%；准备/更新阶段两层合计约18.9ms。此次主要耗时集中在迭代求解API内部，单独优化输入上传不足以改变结论。API内部尚未进一步区分每个CUDA内核与同步操作。',
    '', '因此当前B2+Z1层级GMPC没有获得CUDA OSQP加速，继续以CPU为默认更合理。更大规模、批量QP或专门设计的GPU求解流程需要另外测试，不能据此结论排除全部GPU求解器。','',
    '## 复现','', '从/home/ganhe/project/b2z1_mpc执行：','', '```bash',
    'source relu_hgmpc/env_cuda_osqp.sh',
    'python relu_hgmpc/run_hierarchical_gmpc.py --algebra cuda --duration 2 --scale 1 --results-dir cuda_osqp --name gpu_demo',
    'python relu_hgmpc/run_hierarchical_gmpc.py --algebra builtin --duration 2 --scale 1 --results-dir cuda_osqp --name cpu_demo',
    '```','',
    '完整时间包含模型/矩阵构造、两层求解、数据传输、显式同步、乘子恢复及原坐标验收，初始化成本摊销；不计MuJoCo推进。30秒单次求解超时限制对两后端相同，失败时明确停止，不切回CPU。',
    '', '![短闭环对照](cuda_comparison.png)','',
    '[OSQP官方后端说明](https://osqp.org/docs/backends/index.html) · [实际使用的核心源码版本](https://github.com/osqp/osqp/tree/v1.0.0)']
(out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
fig,axes=plt.subplots(2,3,figsize=(12,6),layout='constrained')
for i,scale in enumerate((1,2)):
    for alg,label,color,style in [('builtin','CPU','#2464a5','-'),('cuda','GPU','#b65035','--')]:
        r=traces[f'{alg}_scale{scale}_2s'];t=[x['time'] for x in r]
        for j,(key,title,unit) in enumerate([('position_error_mm','Position error','mm'),('rotation_error_deg','Orientation error','deg'),('feedback_ms','Total feedback (log)','ms')]):
            ax=axes[i,j];ax.plot(t,[x[key] for x in r],style,color=color,label=label,linewidth=1.5)
            ax.set_title(f'Scale {scale}: {title}');ax.set_xlabel('Simulation time (s)');ax.set_ylabel(unit);ax.grid(alpha=.2)
            if j==2:ax.set_yscale('log')
            ax.legend()
fig.savefig(out/'cuda_comparison.png',dpi=170);plt.close(fig)
if len(sys.argv)>1:
    dst=pathlib.Path(sys.argv[1]);dst.mkdir(parents=True,exist_ok=True)
    for f in ('RESULTS.md','comparison.json','verification.json','verification_fp64_stock_cg.json','cuda_comparison.png','crash_before_lifetime_fix.log'):
        shutil.copy2(out/f,dst/f)
    shutil.copy2(root/'relu_hgmpc/cuda_fp64_precision.patch',dst/'cuda_fp64_precision.patch')
print(json.dumps(groups,indent=2))
