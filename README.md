# B2 与 Z1 层级 GMPC 加几何 WBC

当前维护入口为 `bash run_mpc_wbc.sh --name my_run --case normal`，构建命令为 `bash relu_hgmpc/build.sh`。执行框架与验证口径见 [MPC_WBC_RUNTIME.md](relu_hgmpc/MPC_WBC_RUNTIME.md)。当前可验证的是四足支撑下的机械臂与本体协同；行走阶段将复用 `/home/ganhe/project/legged_control_ros2` 的完整行走链路。

2026 年 11 月 12 日已清理一次性源码改写脚本、散落备份、旧 Python 缓冲运行入口和已放弃的爬行原型，原生核心恢复到此前验证过的站立版本。清理清单和恢复入口见 [CLEANUP.md](CLEANUP.md)。论文基线、ADMM/ReLU-QP/OSQP 实验与原始结果保留，下面为历史实验说明，不能当作当前默认入口。

## acados/SQP-RTI reproduction status

The nonlinear acados implementation is in [paper_repro](paper_repro/README.md). The damped distributed profile now passes three 12-second runs, a serial comparison, and a 60-second closed-loop simulation. It uses ten coordination rounds, primal step 0.2, and dual step 0.05. The single-full-step paper reproduction still fails; no reproduction of the paper's speedup is claimed. Read [the current interface/ADMM audit and results](results/interface_audit/RESULTS.md). The earlier [failure report](results/paper_repro/REPORT.md) and the OSQP prototype below are retained separately.

# B2 + Z1：热启动、有限迭代 MPC 的 MuJoCo 闭环

主工程在 **`/home/ganhe/project/b2z1_mpc`**，运行于 Ubuntu-22.04 WSL2。Windows 中的上一版工程保留为备份，不再作为编译运行目录。

当前已跑通：浮动基座 B2 四足站立，Z1 末端跟踪平滑三维参考；对照集中式 OSQP 与前腿/后腿/机械臂三个物理子系统的 consensus ADMM。动力学和控制器均为 C++，MuJoCo 物理循环也由 C++ 直接调用。

## 运行

```bash
cd /home/ganhe/project/b2z1_mpc
bash scripts/build.sh

# 默认使用热启动。每个控制周期 40 ms，预测 8 段。
bash scripts/run_closed_loop.sh --mode central --duration 6 --central 500 --name central_run
bash scripts/run_closed_loop.sh --mode admm --duration 6 --outer 10 --inner 100 --name admm_run

# 四种配置各重复三次，顺序执行以免互相干扰计时。
bash scripts/benchmark_closed_loop.sh

# 从已执行的闭环状态生成视频，不是 MPC 预测轨迹回放。
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate b2z1_mpc
python scripts/render_closed_loop.py
```

首次配置新机器可运行 `bash scripts/setup.sh`。环境名为 `b2z1_mpc`。`environment.yml` 是环境说明，精确安装清单见 `environment-linux-64.lock` 和 `pip-freeze.txt`。

`--warm 0` 可关闭在线热启动作消融，`--threads 1` 可测试单线程 ADMM；不保证这些组合在低预算下仍通过质量门槛。启动时仍单独做一次初始收敛求解，时间写入 `bootstrap_ms`，不计入在线周期统计。ADMM 启动不使用集中式最优解。

## 已测结果（2026-11-09）

同一初始状态和参考，每种配置 3 次、每次 6 秒，共 450 个控制周期。模拟时间步 1 ms；每 40 个物理步重规划一次。测试为固定仿真步长的离线闭环，没有把计算超时注入机器人运动。

- 集中式，OSQP 最多 500 次：平均 **21.61 ms**，P95 **43.01 ms**，末端 RMS **0.404 mm**。
- ADMM，外层最多 5 次、每个局部 OSQP 最多 100 次：平均 **16.84 ms**，P95 **18.88 ms**，RMS **0.528 mm**。
- ADMM，外层最多 10 次：平均 **24.20 ms**，P95 **27.60 ms**，RMS **0.463 mm**。
- ADMM，外层最多 30 次：平均 **49.86 ms**，P95 **61.65 ms**，RMS **0.390 mm**。

所有配置在记录的控制采样时刻四足接触比例为 100%，没有发生执行器力矩截断。最大足端中心位移约 0.17 mm，机身最大倾斜小于 0.26 度。

**不能把上述结果称为等精度加速。**完整 QP 的缩放约束残差 P95，集中式约 `3.32e-4`，ADMM 5/10/30 次分别约 `0.236 / 0.123 / 0.0503`。ADMM 绝大多数周期到预算上限时还没有达到严格收敛判据；轨迹跟踪良好并不等于优化解已完全可行。当前展示的是特定站立小运动算例下的计算与精度取舍。

计时包含模型更新、预测轨迹平移与重定基、QP 组装/更新和求解；不包含初始收敛、诊断计算、日志写入、物理积分及两种控制器共用的 1 kHz 执行层。集中式约 6.2% 周期超过 40 ms，ADMM 30 次约 78% 超过；ADMM 5/10 次在本次采样中没有超时。这里没有证明硬实时能力。

原始数据和可重复生成的统计：

- `results/closed_loop/bench_*.csv`：真实闭环状态、参考、误差、耗时、约束残差、迭代数。
- `results/closed_loop/bench_*.json`：单次测试结果。
- `results/closed_loop/summary.json`：全部重复试验统计和实验口径。
- `results/closed_loop/comparison.png / comparison.pdf`：误差、耗时分布、QP 残差、足端位移。
- `results/closed_loop/tracking.png`：三维末端位置跟踪。
- `results/closed_loop/closed_loop.mp4`：集中式与 ADMM 10 次的实际执行状态回放，采用碰撞几何显示。
- `results/closed_loop/tests.log`：4 项 CTest 验收结果。

## 状态、物理模型和执行层

反馈直接来自 MuJoCo `qpos/qvel`，没有状态估计器。四元数顺序和自由基座线速度的世界/本体系约定均显式转换。控制中不读取 MuJoCo 接触反力或加速度来代替优化；足端位置和接触点雅可比额外与 MuJoCo 真值校验。

B2 12 个关节 + Z1 6 个关节 + 浮动基座，`nq=25, nv=24`。官方模型固定为 Unitree 仓库提交 `725b59b6de39a26c425632120c89f23c49a36f1e`。无夹爪。安装位置仍是用户同意采用的机背假设 `[0,0,0.12] m`，可在 `config/model.json` 修改。

平地接触位于半径 0.032 m 球形脚的底部。接触点雅可比同时用于接触力映射与无滑移速度约束。MuJoCo 采用椭圆摩擦锥、摩擦系数 0.6、`noslip_iterations=10`、`solref=[0.005,1]`，接触阻抗前三项为 `[0.99,0.999,0.0001]`。这些设置对两种控制器完全相同，用于抑制默认软摩擦下的站立爬移。参见 [MuJoCo 接触建模说明](https://mujoco.readthedocs.io/en/stable/modeling.html#preventing-slip)。

官方 Z1 URDF 带有 1–2 Nm 库仑摩擦和关节阻尼。Pinocchio 的 `nonLinearEffects/RNEA` 不包含这些被动力，仿真保留它们。两种控制器使用相同执行层：MPC 力矩前馈 + 预测关节位置/速度的轻量 PD + 已知参数的阻尼与平滑摩擦补偿：

\[
\tau_{cmd}=\tau_{MPC}+K_p(q_{plan}-q)+K_d(v_{plan}-v)
+d\,v+f_c\tanh(v_{plan}/0.005).
\]

腿部 `Kp=40, Kd=2`，机械臂 `Kp=20, Kd=1`。计划状态在每个 40 ms 周期内由当前状态与首步计划加速度插值。最终命令按官方力矩限幅，所有截断都会计数；验收要求截断数为零。此补偿使用模型参数和状态/计划速度，不是摩擦力真值补偿。

## 几何坐标与闭环预测

每周期基于当前配置建立切空间：

\[
\delta q_k=\operatorname{difference}(\bar q,q_k),\quad
q_k=\operatorname{integrate}(\bar q,\delta q_k).
\]

Pinocchio 更新完整 `M(q)`、`h(q,v)`、末端/接触点雅可比与接触 `Jdot*v`。同一预测时域内仍采用冻结的局部近似，并使用未来各节点的末端参考：

\[
\delta q_{k+1}=\delta q_k+dt\,v_k+\tfrac12dt^2 a_k,
\quad v_{k+1}=v_k+dt\,a_k,
\]
\[
Ma_k+h=S\tau_k+J_c^Tf_k,
\quad J_ca_k+\dot J_cv+K_dJ_cv_k+K_p(e_c+J_c\delta q_k)=0.
\]

接触稳定化增益为 `Kp=60, Kd=16`。目标含机身姿态/位置、末端位置、关节姿态及速度/力矩/力等正则项；约束包含关节限位、速度/力矩限制、局部信赖域及内接摩擦棱锥。

**这仍是局部几何坐标的 MPC 原型，不是完整非线性几何 SQP。**预测时域内没有沿预测轨迹反复线性化所有动力学，也没有非线性有限迭代稳定性证明。

## 热启动与有限迭代

- 上一周期的整段预测向前移一段，末端以最后加速度外推。
- 配置通过 `integrate/difference` 转换到新切空间，速度热启动使用解析链式导数 `dDifference * dIntegrate`。在线过程不做数值有限差分。
- 集中式保留 OSQP 工作区，更新矩阵与边界，平移原始变量和数值对偶初值；稀疏结构改变时重建工作区并重新注入初值，不会丢掉新增非零项。
- ADMM 保留各子问题原始/对偶迭代、接口变量与外层乘子，做时域平移。对偶及接口力的平移是数值初值复用，不声称它是流形上的精确协变解搬运，仅在当前小姿态变化场景中验证。
- 达到迭代上限时可以返回有限的未收敛迭代，但状态明确标记为未收敛；不可行/非有限求解仍中止。闭环日志保留残差，不把提前退出伪装为收敛。
- 物理子系统是前腿、后腿、机械臂；共享基座状态一致，接口力之和为零。模型质量/惯量只计一次，并验证分解重构与完整模型相等。

分解参考 [Amatucci 等的 DWMPC 论文](https://arxiv.org/html/2403.11742v3#S3)，本实现采用显式接口力和局部凸 QP，未直接复现论文的非线性接口力消元/单次 Newton 策略。

## 验收与代码入口

- `src/closed_loop.cpp`：MuJoCo 真值反馈、控制循环、执行层及日志。
- `src/robot.cpp`：Pinocchio 动力学、接触雅可比、解析流形导数、物理子系统缓存。
- `src/warm_start.cpp`：时域平移、末端外推、切空间重定基。
- `src/qp.cpp`：可更新 OSQP 工作区、热启动、迭代预算与状态检查。
- `src/subsystem.cpp`：物理子系统 QP 和有记忆的并行 ADMM。
- `src/test_online.cpp`：解析导数对独立有限差分验证、非零速度的 `Jdot*v`、动力学重构、预测连续性、截断后继续求解。

4 项 CTest 均通过：原有模型/求解一致性；在线导数/热启动/预算；集中式 MuJoCo 闭环；ADMM MuJoCo 闭环。闭环验收要求最大末端误差 <5 mm、最大足端中心漂移 <5 mm、最大机身倾斜 <0.05 rad、控制采样时刻四足接触比例 ≥99%、零力矩截断。

下一步应针对残差和控制性能指定共同精度预算，再优化稀疏 QP 组装、内外层精度分配和接口变量缩放。暂未实现接触切换、碰撞约束、硬件通信或真实延迟闭环。


## ReLU-QP MPC 与优先级 GMPC 对照

新增独立目录 [relu_hgmpc](relu_hgmpc/README.md)：固定站立点全身 MPC + ReLU-QP，与 SE(3) 两层优先级 GMPC + ReLU-QP。C++ Pinocchio/MuJoCo 共用物理接口，PyTorch 执行 CUDA QP 迭代。原 acados/分布式实现保留。

运行 `bash relu_hgmpc/build.sh` 和 `bash relu_hgmpc/benchmark.sh`；结果见 [results/relu_hgmpc/RESULTS.md](results/relu_hgmpc/RESULTS.md)。


## Xie 任务优先级与 WBC 对照

第三种方法的原文 3D 任务及 6D 扩展已加入；见 [实现说明](relu_hgmpc/PRIORITIZED.md) 和 [三种方法结果](results/prioritized/RESULTS.md)。运行 `bash relu_hgmpc/benchmark_prioritized.sh`。
