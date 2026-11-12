# 四足本体—机械臂接口共识 GMPC

本实现是现有 `SharedUpper` 的离线物理分解实验。统一几何 WBC、低层关节伺服和 MuJoCo 闭环入口保持原样。是否接入闭环由独立结果报告决定；求解器返回一个力矩序列不等于满足接入条件。

## 固定的预测问题

- B2 的 12 个腿关节与 Z1 的 6 个臂关节，浮动基位姿位于 SE(3)。
- 预测为 10 个节点、每节点 0.04 s，总时域 0.4 s。
- 沿用当前冻结仿射动力学：质量矩阵、接触雅可比及 Coriolis 矩阵在同一个测量状态计算；保留 `h0 + C0 (v-v0)` 和原接触恢复项。这里没有换成另一种非线性预测器。
- 沿用同一已提交参考的初始 q/v/a、首步 0.02 s 力矩变化约束、其余节点 0.04 s、全部 1340 条物理约束及原权重。
- 默认 `cooperative` 模式由本体和机械臂共同完成末端第一优先级任务；底盘名义位姿与机械臂奇异性偏好共同为第二优先级；原姿态/控制平滑目标为第三优先级。底盘位姿不预先锁定，从而允许冗余运动参与末端跟踪。
- `joint` 对照模式将底盘与末端的任务行合为第一优先级，姿态及控制平滑作为第二优先级。`strict` 对照模式保留原来的底盘、末端、平滑三级。每种模式均与相同模式的集中式问题比较。

三种布局均保持接触、摩擦、关节与力矩边界为硬约束。底盘和机械臂是共同完成末端任务的执行机构，不按执行机构设置主从优先级；优先级作用于末端、底盘参考、奇异性偏好和平滑等任务。既有位置与旋转误差权重保持不变。

`body_arm_tasks.py` 为两种求解器构造同一任务布局。奇异性目标采用经过长度尺度归一化的机械臂局部雅可比，计算正则化对数体积及其六个臂关节方向的一阶梯度。它是第二级软偏好，默认权重 1、正则化参数 0.05、长度尺度 0.5 m；并不构成远离奇异点的硬保证，在梯度为零的配置处可能不产生动作。集中式奇异性行直接从原生约束矩阵中取关节局部位移映射，分布式行从独立机械臂预测节点构造，然后逐行验证两者等价。

具体地，令机械臂 LOCAL 雅可比的线速度三行除以长度尺度，得到 \(J_s\)。定义

\[
m(q_a)=\tfrac12\log\det(J_sJ_s^T+\epsilon^2I),\qquad
r_{sv,k}=\sqrt{w_{sv}}\left(1-\tfrac12\nabla m(q_{a,0})^Tp_{a,k+1}\right).
\]

每个周期用 6 个臂关节方向的中心差分求一次梯度，随后整个预测时域与两种求解器复用。该残差是一阶局部偏好，原有关节位移信赖域保持不变。

## 按物理接口分解

`body_arm_interface.py` 在固定安装关节 `z1_mount` 处分配不重叠的 link 惯性。保持同一 Pinocchio 关节树，再分别选择本体坐标和臂坐标：

\[
v_B=(v_b,\dot q_\ell),\qquad v_A=(v_b,\dot q_a).
\]

两套模型中的浮动基运动是同一物理量的副本，惯性不是副本。将局部矩阵嵌回全身坐标后，必须恢复原来的质量、偏置和 Coriolis 矩阵。

接口 wrench 在 `z1_link00` 的局部坐标系表达，参考点是安装接口原点，排列为线力、力矩。共同变量的符号定义为“机械臂作用于本体”的力：本体动力学使用正号，机械臂动力学使用负号。该物理力与 ADMM 对偶乘子分开存储。

局部动力学写成

\[
M_Ba_B+h_B+C_B(v_B-v_{B0})
=B_B\tau_\ell+J_c^Tf_c+J_{IB}^TF_I,
\]

\[
M_Aa_A+h_A+C_A(v_A-v_{A0})
=B_A\tau_a-J_{IA}^TF_I.
\]

本体保留原接触恢复方程；机械臂没有独立的地面接触力。局部消元后，每节点本体变量为 12 个腿部力矩与 6 个接口力分量，机械臂变量为 6 个臂部力矩与 6 个接口力分量。因此首层局部维数分别为 180、120，原集中式凝聚问题为 180 维。此次分解并不自动减小较大子问题的维数。

## 接口共识与同一几何坐标

每个节点交换安装接口的冻结加速度映射与六维 wrench：

\[
E_Bx_B+e_B=E_Ax_A+e_A.
\]

双方使用相同的名义位姿、初始局部位移和速度，以及相同积分公式。接口加速度一致因而推出后续速度和局部位移一致；恢复阶段仍独立检查三者，不用加速度残差替代全部运动检查。

位姿采用 SE(3) 局部坐标，与当前 GMPC 的表示一致。没有对旋转矩阵作元素平均，也没有把不同切空间中的坐标直接相减。

## 数值算法

`body_arm_admm_native` 使用两个可并行的局部二次更新，加上一个公共投影步骤。公共步骤投影各自的物理边界，并对安装接口形成共识。算法上的第一块是两个局部变量的整体，第二块是约束及接口变量；它不是任意同步更新的两块 Jacobi 迭代。

- 每个控制状态只构造一批冻结局部矩阵。
- 在同一层 ADMM 迭代内复用两个数值分解；改变惩罚参数时重分解，并计入耗时。
- 近端项作用于相邻迭代差，不能成为原目标额外的永久正则项。
- `body_arm_admm_solver.py` 可做可逆坐标预条件化和约束生成。预条件对角移位仅用于选坐标，原二次目标不额外增加正则；共同目标倍率同时作用于两块，返回乘子时恢复原目标尺度。
- 最终层级驱动传入最小二乘因子 \(F,b\)，满足 \(H=F^TF\)、\(g=F^Tb\)。预条件器对 \([F;\sqrt{\delta}I;\sqrt{\gamma}WE]\) 做 QR，取上三角因子 \(U\) 并使用 \(x=U^{-1}y\)；\(WE\) 项可选。原目标仍由变换后的最小二乘因子表达，接口项只用于坐标度量。这样避免先形成半正定 Hessian 再白化放大零空间舍入误差；没有裁剪特征值来改变原目标。
- 约束生成可以先求较少边界行，但随后必须检查全部原始边界并加入被违反行；被省略行的乘子为零。不能把筛出的约束从物理问题永久删除。
- 局部物理边界不保证在中间迭代全部满足，只有通过原始模型证书的结果才可接受。
- 每一级求解后，使用候选自己求得的任务输出和活跃约束构造下一级锁。集中式解、任务锁与活跃集不作为候选输入。
- 任何一级的接口、物理边界、原始 KKT 或层级锁检查失败，就停止该次层级求解，不生成可执行 Plan。

## 等价性与验收

`test_body_arm_interface.py` 检查惯性分割、接口力符号、全身映射重组和单边接口扰动。

`test_body_arm_problem.py` 将独立构造的局部任务和约束回拼，与未修改的原生集中式矩阵逐行比较。比较全部 1340 条约束、三组原任务的 1140 行、无穷界模式、仿射偏置及力矩往返映射。即使采用 `joint` 模式，原三组任务行仍保留以便独立核查。

`cooperative` 的独立任务审计还覆盖新增 10 行奇异性偏好，实际三级任务合计 1150 行；全局与局部行 ID 必须完整且没有重复。

离线基准同时检查：

1. 原全身坐标下的 primal、dual 与 normal-cone 残差及物理越界。
2. 各层自身的任务锁与全局投影误差。
3. 安装接口加速度、速度、局部位移与 wrench 的不一致量。
4. 相同模式集中式解与分布式解的任务输出、首步与全时域力矩差。
5. 原始与局部矩阵的代数等价性。

原始 KKT 和接口阈值默认 `2e-7`。另设公开的任务输出差阈值 `2e-7`、物理力矩差阈值 `1e-4 N m`。这些指标单独记录；病态目标下仅 KKT 小并不保证任务输出接近。

## 离线运行

在 WSL 中：

```bash
cd /home/ganhe/project/b2z1_mpc
source relu_hgmpc/env.sh
python relu_hgmpc/test_body_arm_interface.py
python relu_hgmpc/test_body_arm_problem.py
python relu_hgmpc/test_body_arm_tasks.py
python relu_hgmpc/test_body_arm_admm_kernel.py
python relu_hgmpc/benchmark_body_arm_admm.py \
  --priority cooperative --cases normal --limit 2 --repeat 1 \
  --output results/body_arm_admm/calibration
```

默认案例来自已有 `normal`、`double`、`push` 的历史真值轨迹。每个案例使用 12 个时点，其中 0、0.5 s 只用于校准；其余 10 个时点为验证集。完整验证应覆盖 30 个不同验证状态，并重复测量。采样、参数、代码哈希、原始解与失败记录均写入输出目录。

数值选项与物理任务参数分开：

- `--kernel-tol` 和 `--kernel-interface-tol` 可以让内层迭代采用比公共验收更严格的停止阈值。
- `--objective-multiplier` 对两个局部目标乘同一个正数，不改变原任务权重比。
- `--preconditioner-shift` 控制坐标预条件的移位；`--preconditioner-interface-weight` 可在可逆坐标度量中考虑接口矩阵，不向物理目标加入接口惩罚。`--screen-bounds` 开启带全边界复核的约束生成。
- `--rho`、`--sigma`、`--alpha`、`--adaptive-interval`、`--max-iter` 控制 ADMM 迭代，必须记录重分解和迭代成本。
- `--singularity-weight`、`--singularity-eps`、`--length-scale` 是显式任务参数，两种求解器必须相同；改变它们属于改变实验问题，不能用于偷偷减轻候选负担。

调参时只使用校准集，例如以下候选设置；它们不是已经验证有效的默认配置：

```bash
python relu_hgmpc/benchmark_body_arm_admm.py \
  --priority cooperative --cases normal --limit 1 --repeat 1 \
  --objective-multiplier 1e6 --kernel-tol 1e-10 \
  --kernel-interface-tol 1e-9 --preconditioner-shift 1e-6 \
  --screen-bounds --max-iter 20000 \
  --output results/body_arm_admm/calibration_hierarchical_v1_first
```

早期 `calibration_cooperative_first` 和 `calibration_cooperative_scale1e6` 是尚未引入奇异性第二级的旧两层调试结果，不代表当前三级方法；不要将它们混入正式统计。当前三级实验的元数据应包含 `expected_solver_levels=3`，任务顺序为 `end_effector`、`base_and_singularity_preference`、`smoothness`。

本轮末端共同主任务的冻结配置完整命令如下。该配置的正式验收未通过，这些命令用于复现失败结果，不是已经验证可上线的推荐配置：

```bash
python relu_hgmpc/benchmark_body_arm_admm.py \
  --priority cooperative --split calibration --repeat 1 \
  --objective-multiplier 1e6 --kernel-tol 1e-10 \
  --kernel-interface-tol 1e-9 --preconditioner-shift 1e-6 \
  --preconditioner-interface-weight .1 --screen-bounds --max-iter 20000 \
  --output results/body_arm_admm/calibration_hierarchical_v2
python relu_hgmpc/benchmark_body_arm_admm.py \
  --priority cooperative --split heldout --repeat 3 \
  --objective-multiplier 1e6 --kernel-tol 1e-10 \
  --kernel-interface-tol 1e-9 --preconditioner-shift 1e-6 \
  --preconditioner-interface-weight .1 --screen-bounds --max-iter 20000 \
  --output results/body_arm_admm/heldout_hierarchical_v2
python relu_hgmpc/report_body_arm_admm.py \
  results/body_arm_admm/heldout_hierarchical_v2/results.json \
  --output results/body_arm_admm/heldout_hierarchical_v2/final_report
```

正式命令还须带上在校准阶段冻结的数值选项；以输出 `arguments`、`options` 和源码哈希为准。不能在验证集上边观察结果边更换参数，然后把同一批样本称为未参与调参的验证集。

## 计时和结果解释

QP 端点总时间包含同一次状态评估、各自的物理线性化及矩阵构造、局部凝聚或原集中式构造、奇异性梯度、各自任务行、层级处理、数值分解、ADMM 迭代、数据转换和求解器所需的原坐标证书。集中式装配在未修改的 `SharedUpper` 求解器入口截获，随后使用原 `HierarchyQP` 求解新的共同任务布局。共享梯度和公共布局开销在两种独立控制器的假设计时中各计一次；中央不免费依赖局部凝聚矩阵。当前集中式方法是现有的层级混合求解器，不能用一个未优化的 OSQP 调用替代它以制造加速。

双方均排除目标数据加载、磁盘写入、MuJoCo 前进、Plan 构建和非线性预测缺陷诊断。预测轨迹恢复另计；完整代数审计及与参考最优解的比较属于离线验证，也单独报告。不能把这些离线结果直接称为 50 Hz 闭环结果。

报告保留所有失败尝试。仅共同通过的样本用于精度相同时的加速比，全部尝试的耗时与失败数同时公开。少量校准样本永远不触发闭环接入；正式完整验证集全部满足精度，并且完整求解均值与 P95 均改善后，才具备开展下一步闭环实验的依据。

本轮最终结果位于 `results/body_arm_admm/heldout_hierarchical_v2/FINAL_ANALYSIS.md` 与 `final_report/`：集中式 90/90 通过，分布式 0/90 通过，同解精度及速度门槛均未通过。全部原始边界仍被最终检查，未收敛的中间解没有交给 WBC。记录中的 `completed_levels` 包含失败层，应理解为尝试层数，不能将到达末层解释为成功完成优化。
