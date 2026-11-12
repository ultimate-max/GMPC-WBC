# 50 Hz MPC／200 Hz 几何 WBC：双线程与 OSQP

当前入口为 `run_threaded_online_mpc.py`，默认筛选 OSQP 与实际规划工作线程。WBC 运行于主线程，规划线程持有自己的 Pinocchio 数据、原生 Plant 和 OSQP 工作区。两线程传递独立的状态／参考快照，不共享可变动力学缓存；各自绑定不同 CPU 核，BLAS 单线程。

MPC 保持 10 个 40 ms 预测节点、0.4 s 时域，每 20 ms 根据新状态重新规划。WBC 每 5 ms 跟踪连续参考。目标交接延迟为 20 ms；如果规划实际未完成，WBC 继续原参考，完成后的计划再连续交接。错过的请求槽、迟到完成及 WBC 时限均记录。预热在控制时钟启动前单独计时。

## 等价计算加速

冻结动力学／接触系数和 SE(3) 一阶任务误差与原在线模型一致。C++ 构造相同凝聚 QP，期间释放 Python 执行锁；独立测试核对 H、g、G、l、u 和目标值。五次 SE(3) 参考和接触运动学也转到原生运算，使用同一数值公式。RK4 中只需位姿／速度时不求加速度；只需接触雅可比时不求质量矩阵和偏置力。

每次规划的原问题为

\[
\min_z \tfrac12 z^T H z+g^Tz,\qquad l\le Gz\le u.
\]

使用当前 H 的 Cholesky 分解 H=LLᵀ，令 z₀=−H⁻¹g、z=z₀+L⁻ᵀy，则目标变为 ½‖y‖² 加常数，约束变为 l−Gz₀≤GL⁻ᵀy≤u−Gz₀。这是变量变换，模型、约束和代价没有更改。每次系数变化都重新进行必要的数值分解。

1. 活跃面预测：取前次非零乘子对应的上下界，以 SVD 求其等式约束最小范数解。只有所有原物理边界误差≤1e−9、原问题归一化 KKT≤1e−10 才直接接受。无活跃约束时同样检查无约束解。快路径失败不产生控制输出。
2. OSQP 约束生成：以前次活跃约束及无约束解违反的行初始化。求解后检查全部 1340 行，补入遗漏的违反约束，直到完整原问题通过共同的 2e−7 验收门槛。未纳入行的乘子设为零并参与完整驻点、法锥检查。当前循环内约束集合只增加。
3. 结构复用：OSQP 使用固定数量的稠密约束槽，容量不足时扩容。更新矩阵数值时仍重新分解，复用的是符号结构。上一步原始变量解按重规划间隔／预测步长移动，本设置是半节点插值；对偶热启动重置，避免槽映射变化时错误沿用乘子。

在严格凸 QP 的精确算术下，完整 KKT 条件是全局最优性证书；这里使用数值容差，因此报告实际残差。约束生成和活跃面预测本身不提供固定最坏执行时间保证。快路径直接解出时没有执行 ADMM，报告明确区分这两类调用。

## 对照与计时

`benchmark_threaded_online_mpc.py` 比较串行 quadprog、双线程 quadprog、双线程活跃面＋OSQP。三者使用同一原生矩阵和参考实现、约束、权重、预测时域和 WBC。三个工况、各两次 12 s；第二轮反转策略顺序。

计时包含在线矩阵与参考构造、CPU 数据传递、全部求解及原问题／节点复核。并行总工作量用两线程 CPU 时间之和，不把重叠的墙钟时间简单相加。规划从控制周期起点到实际完成的延迟，以及 WBC 从计划唤醒时刻到控制计算完成的响应时间分别记录。目标预计算、预热、日志和 MuJoCo 积分单独说明。

这是墙钟配速的 WSL 仿真；出现超时会真实推迟计划交接并记录跳槽。物理引擎仍按每个控制循环推进 5 个 1 ms 步长，尚未加入实机丢帧保持力矩模型，也不是硬实时认证。最终数据见 `../results/online_wbc_mpc_threads/RESULTS.md`。

```bash
cd /home/ganhe/project/b2z1_mpc
source relu_hgmpc/env.sh
python relu_hgmpc/test_native_online_mpc.py
python relu_hgmpc/test_fast_online_reference.py
python relu_hgmpc/benchmark_threaded_online_mpc.py
python relu_hgmpc/report_threaded_online_mpc.py
# 单次 12 s 双幅工况，默认实际双线程和 OSQP 组合求解：
python relu_hgmpc/run_threaded_online_mpc.py --case double --name demo
```
