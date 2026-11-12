"""Write a candid experiment report; never infer speedup from aborted trajectories."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'results/paper_repro'
s=json.loads((D/'summary.json').read_text());c=s['central'];d=s['distributed']
def read(name): return np.atleast_1d(np.genfromtxt(D/(name+'.csv'),delimiter=',',names=True))
one=read('distributed_trial1');serial=read('distributed_serial_trial1')
fields=['error_m','interface_l2','full_defect_l2']+[f'q{i}' for i in range(25)]
diff=max(float(np.max(np.abs(one[f]-serial[f]))) for f in fields) if len(one)==len(serial) else float('inf')
times=[x['time'] for x in d['abort_events']]
report=f'''## Material Passport

- Artifact: experiment result / reproducibility validation
- Verification status: **NOT VERIFIED — distributed closed loop failed**
- Project: `/home/ganhe/project/b2z1_mpc/paper_repro`
- Source: [paper](https://arxiv.org/html/2403.11742v3), [author repository](https://github.com/iit-DLSLab/DWMPC)
- Data: local MuJoCo simulation, no hardware experiment
- Environment: WSL Ubuntu 22.04, Intel Core i9-13980HX; dependency commits in `environment.json`

## 本轮结论

已完成 C++ acados/SQP-RTI 实现与共同实验口径，但**尚未复现论文中的分布式闭环性能或加速结论**。集中式完成全部三次实验，分布式重复失败，不能将失败轨迹的短耗时当作加速结果。这也不能证明论文方法本身无效：当前是 B2+Z1、Pinocchio 和重新构建的接口消元实现，仍可能有离散化、局部 OCP 构造及参数选择方面的差异。

## 已实现与对齐

- Pinocchio 完整非线性刚体动力学，CasADi 自动微分与 C 代码生成；在线控制器、三个独立求解器 capsule 及 MuJoCo 均由 C++ 调用。
- 两者均为 50 个预测区间、10 ms 步长、0.5 s 预测时域；仿真每 20 ms 更新一次控制，每次执行一个 SQP-RTI 步。热启动前移两个节点。
- 每节点维度（不含软约束松弛变量）：集中式 49 状态 + 30 输入；前、后腿分别 25 + 12；机械臂 25 + 6。没有接口力决策变量。
- 相同完整机器人、接触几何、任务权重、静力前馈控制参考、关节/力矩/摩擦约束、物理引擎设置及低层伺服。分布式额外包含一致性及近端项。
- HPIPM、50 个保留区间、50 次 QP 迭代上限、1e-6 QP 容差；acados 内部单线程，局部求解器在三个线程并行运行。
- 分别记录 acados 内部求解时间、并行区域墙钟时间，以及参数更新、通信/合并在内的控制器总时间。诊断、日志和物理仿真不计入两者控制器计时。
- 接口残差是三个无序子系统对、51 个状态节点的速度差整体 L2 范数，同时记录单节点最大值；另用同一完整动力学函数核查两者的预测缺陷。原文未完整说明残差跨节点的聚合方式，不能声称数值阈值完全相同。

## 有效结果：集中式

三次各 6 s，共 {c['samples']} 个控制周期：

- 控制器总耗时均值 **{c['controller_ms']['mean']:.3f} ms**，P95 **{c['controller_ms']['p95']:.3f} ms**。
- 求解器内部耗时均值 {c['max_local_solve_ms']['mean']:.3f} ms；其中线性化约 {c['critical_lin_ms']['mean']:.3f} ms，QP 约 {c['critical_qp_ms']['mean']:.3f} ms。
- 末端位置 RMSE **{c['rmse_mm']:.3f} mm**，最大误差 {c['max_error_mm']:.3f} mm。
- 全部控制采样点均为四足接触；最大足端位移 {c['max_foot_drift_mm']:.3f} mm，最大机身倾角 {c['max_tilt_deg']:.3f}°。
- 求解失败 {c['failed_solves']} 次，力矩截断 {c['torque_clips']} 次。
- 完整动力学预测缺陷整体 L2 最大 {c['full_defect_l2']['max']:.6g}，单节点最大 {c['full_defect_max']['max']:.6g}。

**这不是已达到 50 Hz 实时控制的证明。** {100*c['deadline_miss_fraction']:.1f}% 的周期耗时超过 20 ms。仿真时间独立推进，没有把计算延迟注入机器人；真实实时性能尚未通过。

## 分布式失败结果

三次退出码为 {d['exit_codes']}，首次求解失败时刻分别为 {times} s。失败命令未施加到仿真。失败前已经出现接口残差和完整动力学缺陷增长，因此不能使用其部分轨迹均值与完整集中式轨迹计算加速比。

串行执行同一组三个局部求解器也出现相同失败；状态、跟踪误差和两类残差的最大记录差值为 {diff:.3g}。这说明所观察到的问题并非仅由并行调度引起。根因仍未确认，不能把它直接归因于某个罚因子或据此否定论文结论。

`summary.json` 中 `speedup.valid=false`。原始 CSV、运行日志、退出码及失败事件均保留，不能只选择失败前较快的时段报告性能。

## 已通过的模型验证与未通过的验证

12 组非零姿态、速度下：分解后完整预测与完整模型最大绝对差约 6e-14；独立 Pinocchio 数值动力学对照约 7e-15；任务代价分解误差约 2e-13；自动微分方向导数与中心差分约 2.3e-8。详见 `model_verification.log`。

这些检查只验证接口一致处的模型/代价与导数，**不验证非一致迭代中的局部 NLP 等价性、ADMM 收敛性或闭环稳定性**。分布式控制器未通过闭环验证。

## 与原始实验的差异和复现边界

论文采用 ADAM 和其机器人模型；这里依用户要求采用 Pinocchio、B2+Z1，以及可配置的仿真机背安装位置。任务是四足站立时的机械臂末端跟踪，没有完成原文步态、外力扰动及全部轨迹套件。

作者公开 main 分支当前生成脚本是 15 节点、非均匀步长、约化全身惯量块及辅助接触力输入，不能当作论文 50 节点机械臂实验的完整源码。develop 分支也未提供可直接运行的原始三子系统机械臂生成脚本。当前物理子系统/接口 Schur 消元属于根据公式重建的实现。

共享躯干惯量均分为三份，合计仍等于原模型；具体原始惯量分配和全部权重未获得。默认 rho=200、sigma=100 为当前记录的实现参数，运行节点代价乘 0.01，终端只保留近端项，不加一致性项。静力前馈参考、接触软约束、末端约束范围等细节见 `paper_repro/README.md`。

输入近端、更新松弛、加速度消元、完整动力学热启动重滚动、足端高度容差及多次协调迭代等开发诊断均不是默认正式配置，也没有被认定为有效修复。旧日志仅作排错记录，不能混入正式统计。

## 复现命令

```bash
cd /home/ganhe/project/b2z1_mpc
source paper_repro/env.sh
python paper_repro/verify.py
bash paper_repro/benchmark.sh
python paper_repro/audit_report.py
```

生成/构建使用 `bash paper_repro/build.sh`。`benchmark.sh` 即使个别控制器失败也会完成结果收集，必须检查每次 `.exit` 和 `summary.json` 的 `completed/tracking_valid`，不能只看脚本是否执行结束。

尚待完成的是分布式局部问题及其非一致迭代行为的验证，之后才能做同等可行性和跟踪精度下的有效性能比较。当前不报告“复现成功”或“加速若干倍”。
'''
(D/'REPORT.md').write_text(report)
manifest={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((ROOT/'paper_repro').glob('*')) if f.is_file()}
(D/'source_sha256.json').write_text(json.dumps(manifest,indent=2))
print('Report written; distributed reproduction remains NOT VERIFIED.')
