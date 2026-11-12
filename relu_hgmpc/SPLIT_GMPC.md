# 底盘＋机械臂分布式 GMPC

这是从速度分配原型扩展的有限时域动力学控制器。它每个反馈周期求解未来 0.3 s 的力矩、地面反力和接口力，直接执行首步力矩；不再调用速度层分配器或其最终 WBC QP。

代码为 `split_gmpc.py`、`run_split_gmpc.py`、`test_split_gmpc.py`。C++ Pinocchio/MuJoCo 执行及测量接口为 `native.cpp`。局部 QP 当前使用 OSQP，原有 ReLU-QP 基线与两层 GMPC 另行保留。

## 两个预测子系统

- 底盘块：浮动机身 6 自由度、腿关节 12 自由度；每节点控制为 12 个关节力矩、12 个地面力分量、6 个接口力旋量分量。
- 机械臂块：机身副本 6 自由度、臂关节 6 自由度；每节点控制为 6 个关节力矩和 6 个接口力旋量分量。

两块使用同一个测量位姿和机身切空间。为获得两个正定的虚拟子系统惯量，机身公共根惯量各分一半，腿与臂的关节分支仅计入一次。这是计算分解；接口力不能直接当作真实安装法兰传感器读数。合并模型的惯量、偏置及接口 KKT 加速度均与完整 Pinocchio 模型独立核对。

各块满足 `M_i a_i+h_i=S_i tau_i+Jci^T f_i+E_i^T w_i`。臂块没有地面力。接口满足 `v_base,b=v_base,m` 与 `w_b+w_m=0`。两块从相同初始状态出发并使用相同积分方式，逐节点速度一致使机身位姿增量一致；另记录位姿增量差。

## 几何预测与物理约束

10 节点、30 ms 预测步长、20 ms 反馈周期。机身和末端采用 SE(3) 对数误差，右扰动导数 `Jlog6(Td^-1*T)*Jbody` 已经有限差分验证。加速度按半隐式方法积分；序列按 20/30 时间比例移位热启动，机身局部坐标变化时旋转接口热启动量与乘子。

每周期更新动力学与雅可比，在本次预测域内冻结系数；目标位姿和几何误差按节点更新。这是重线性化几何 MPC，不是沿整条预测轨迹反复求解到收敛的非线性 SQP。另在预测节点重算完整非线性 Pinocchio 动力学，分开记录基座力、基座力矩和关节力矩残差，量化冻结模型近似误差。

底盘块保持 `Jc*a+Jdot*v+100*e_foot+20*Jc*v=0`，并包含摩擦棱锥与 10–800 N 法向力界。两块包含关节位置/速度、85% 额定力矩界和机身运动范围。前馈力矩变化界为腿 1000 Nm/s、臂 100 Nm/s，首步相对上次执行的前馈力矩。共同伺服还叠加 PD 与摩擦补偿，因此这个界不自动约束最终执行力矩，后者单独测量。

末端位姿/速度、机身姿态、关节姿态及输入平滑组成当前加权代价。物理约束是硬约束，机身/臂偏好是连续代价，没有 0/1 启停选择。本版本不声称严格词典序任务优先级；原有两层主任务保持 GMPC 仍在 `controller.py`。

## ADMM 与集中式对照

状态经动力学消元后，底盘块 300 变量、臂块 120 变量。共识量为每节点机身速度与有符号接口力；线速度、角速度、力、力矩分别以 0.2 m/s、0.5 rad/s、200 N、50 Nm 缩放。

两局部 QP 同时更新，再更新共同共识和缩放乘子；不以新旧混合远端状态消元接口力。`--parallel` 使用两个独立 OSQP 实例和两个工作线程。全部协调轮次、模型构造、求解器设置与结果提取计入反馈耗时。rho=30；原始残差 <1e-5 且对偶残差 <1e-4 才标为已收敛，耗尽预算的周期不能当作已收敛。

集中式直接堆叠相同代价和局部约束，把相同接口等式作为硬约束；比较的是同一个冻结模型优化问题。它不是旧的 ReLU-QP 固定站立点基线，不能混用旧数字计算加速比。

没有集中式回退。执行前检查局部约束、接口残差和首步完整接触动力学加速度偏差，超过门槛停止仿真并记录失败。接触良好不等于优化完全收敛；原始/对偶残差、接口净力、完整模型缺陷均保存。

## 运行

```bash
cd /home/ganhe/project/b2z1_mpc
source relu_hgmpc/env.sh
python relu_hgmpc/test_split_gmpc.py
python relu_hgmpc/run_split_gmpc.py --mode central --duration 12 --name central_12s
python relu_hgmpc/run_split_gmpc.py --mode distributed --parallel --iterations 30 --duration 12 --name distributed30_12s
python relu_hgmpc/run_split_gmpc.py --mode distributed --parallel --iterations 30 --duration 60 --name distributed30_60s
```

输出在 `results/split_gmpc/`。源码哈希保存在 JSON；NPZ 包含 1 kHz 接触、接触点速度、最终力矩和实测关节速度。速度层抖振消融在 `results/chatter_1khz/`。

MuJoCo 时间独立于计算耗时推进，未注入延迟。当前 Python/稠密凝聚原型没有达到 20 ms 实时预算，不能宣称 ADMM 已带来计算加速，或几何表示单独提高了精度。

## 集中式 ReLU-QP 后端

集中式模式增加 `--backend reluqp`；默认仍为 `osqp`，分布式模式仍使用 OSQP。接入保持 `build_part`、模型、物理约束、权重、10×30 ms 预测和控制序列移位不变。新后端 `central_relu.py` 用精确等式消元与 Hessian 白化，将相同的 420 变量、1280 行约束 QP 转成 180 变量等价问题，再调用 `relu_solver.py` 的 GPU ReLU-QP 递推。没有改变物理目标或约束，没有 CPU 求解器回退；固定内部 rho=0.1，不声称完整复现上游自适应罚参数实现。

原始可行性、驻点、法锥投影三项统一相对残差均不超过 `--qp-tol 2e-7` 才执行控制。OSQP 内部停止不满足共同标准时继续收紧精度。所有预处理、构造、数据传输、迭代和结果校核进入反馈计时，首次 CUDA 初始化也计入。求解器的预处理不同，等式消元带来的高等式精度不能归为 GPU 的固有优势。

复现：

```bash
source /home/ganhe/project/b2z1_mpc/relu_hgmpc/env.sh
python relu_hgmpc/test_central_relu.py
bash relu_hgmpc/benchmark_central_relu.sh
python relu_hgmpc/report_central_relu.py
```

2026-11-10 的 ABBA 顺序各两次 12 秒对照：OSQP / ReLU-QP 位置 RMSE 为 0.32596 / 0.32969 mm，总耗时均值为 54.01 / 81.99 ms，所有周期残差达标且四足接触保持。当前 ReLU-QP 的预处理/设置/上传均值约 50.55 ms，整体尚未加速。完整结果在 `results/central_relu/RESULTS.md`，原始 CSV/NPZ 在 `results/split_gmpc/central_{osqp,reluqp}_12s_r{1,2}*`。
