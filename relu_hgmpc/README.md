# B2+Z1: ReLU-QP MPC 与优先级 GMPC

## 当前入口

运行项目根目录的 `run_mpc_wbc.sh`，进入 `run_mrt_wbc.py` 的 MPC 生产者和原生 C++ WBC 消费者。公共实验支持函数位于 `runtime_support.py`。详见 [MPC_WBC_RUNTIME.md](MPC_WBC_RUNTIME.md) 和 [清理记录](../CLEANUP.md)。以下各节保留历史实验的条件与结论。

## 共用几何 WBC 的严格对照（2026-11-11）

新增统一接口对照见 [SHARED_GEOMETRIC_WBC.md](SHARED_GEOMETRIC_WBC.md)，结果在 `../results/shared_geometric_wbc/RESULTS.md`。运动分配适配基线与严格层级 GMPC 共用 SE(3) WBC、完整任务目标、优先级、物理边界、参考拼接与关节伺服。基线是当前目标的优先级微分逆运动学适配，不是原论文 SNS 算法原样复现；原实现仍保留。

固定逻辑 50 Hz 的普通／双幅结果显示跟踪更准、机械臂实际速度高频量下降，但力矩与腿部指标不是全面改善，总 CPU 更高。实际多线程尚未稳定达到 50 Hz；扰动仍有失败，原始记录、残差、计时和失败诊断均保留。因此本轮不宣称已同时实现精度、抖振与计算效率的全面优势。


## 历史 Python 双线程 MPC＋几何 WBC（2026-11-11）

该历史实验入口 `run_threaded_online_mpc.py` 默认使用实际 MPC 工作线程、独立动力学数据和活跃面预测＋筛选 OSQP；主线程执行 200 Hz 几何 WBC。目标规划频率 50 Hz，实际迟到与跳槽显式统计。C++ 负责等价矩阵构造和几何参考，计算时释放 Python 执行锁。实现与复现见 [THREADED_MPC.md](THREADED_MPC.md)，完整对照见 `../results/online_wbc_mpc_threads/RESULTS.md`。

最终 18 次完整闭环均保持四足接触、无力矩裁剪。加速组合相对双线程 quadprog 总控制 CPU 时间减少约 19%，末端位置 RMSE 变化小于 0.04%。加速版 3597 次规划中 6 次超过 20 ms，并跳过 3 个请求槽；WBC 超过 5 ms 为 55/14400 次，尚未满足所有周期均准时的实时验收。实际关节高频指标没有整体改善。

与优先级运动分配＋WBC 的共同末端任务对照见 `../results/allocation_vs_online_mpc/RESULTS.md`。基线保留滞环，并分别检查旧接触位置稳定项与滚动接触修正。当前 MPC 的高频运动和接触保持较好，但普通工况 CPU 比修正基线高约 63%、末端姿态误差也较大；双方底盘目标、层级和约束结构不同，不能归因为单独的 MPC／李群优势。双幅基线因底盘侧向越过本站立移植的 0.15 m 界后速度区间为空而终止，独立诊断已保存，不把该移植边界失败解释为原论文方法必然失败。

## 在线 MPC＋几何 WBC：最低 50 Hz（2026-11-11）

该历史在线规划入口 `run_online_wbc_mpc.py` 默认采用 50 Hz MPC、200 Hz WBC、20 ms 逻辑交接延迟。预测节点仍为 40 ms、10 个节点、0.4 s 时域。MPC 是更新实测模型的单次线性化参考生成器，几何 WBC 使用严格任务层级；不是下方旧层级 GMPC 原封不动提高频率。

普通、双幅和 120 N 外力扰动各重复两次，6 次 50 Hz 闭环及 6 次相同延迟的历史 10 Hz 对照全部完成，每次 12 s。50 Hz 共 3600 次规划、每次包含矩阵与参考构造的平均耗时约 10.2–10.5 ms、最慢 18.27 ms；全程保持四足接触、无力矩裁剪。双幅工况末端位置 RMSE 从 10 Hz 的 2.23 mm 降至 1.65 mm，总在线计算增加约 86%。

同核一次 MPC 加四次 WBC 等工作的 20 ms 计算窗口仍有 46/3600 个超预算；这是串行仿真的逻辑交接，不是实际异步线程的硬实时验证。旧 5–20 Hz 反馈触发方案不满足当前最低频率要求，不作为合格加速方案。完整残差、实际关节抖振和计时见 `../results/online_wbc_mpc_50hz/RESULTS.md`。

```bash
cd /home/ganhe/project/b2z1_mpc
source relu_hgmpc/env.sh
python relu_hgmpc/test_online_wbc_mpc_50hz.py
python relu_hgmpc/benchmark_online_wbc_mpc_50hz.py
python relu_hgmpc/report_online_wbc_mpc_50hz.py
```

## CUDA OSQP 实测（2026-11-10）

已在RTX 4060 Laptop接入双精度CUDA OSQP，维持与CPU相同的2e-7原问题验收门槛。两档幅值各2秒、两后端共400个控制周期均通过精度检查、保持四足接触、无力矩裁剪；另有五个真实MPC状态的回放核对。

GPU完整反馈均值约2151/2022 ms，CPU同条件约51.6/54.0 ms，GPU耗时约为CPU的41.7/37.4倍，末端RMSE相对变化不到0.03%。这是未注入计算延迟的短闭环，不能用于宣称实时控制，也不等于12秒全轨迹验证。当前仍以CPU为默认。

实现需要FP64构建、内层CG精度下限收紧，以及多工作区全局CUDA句柄的引用计数修复。说明与复现见 [CUDA_OSQP.md](CUDA_OSQP.md)，详细结果见 `../results/cuda_osqp/RESULTS.md`。


## 当前一阶层级 GMPC（2026-11-10 v0.6）

当前验证入口为 `run_hierarchical_gmpc.py`，实现文件为 `hierarchical_gmpc.py`。在补全一阶任务动力学的 `PaperGMPC` 上，第一层跟踪末端 SE(3) 位姿与误差变化率，第二层严格保持预测主任务输出，再调节机身/关节构型和输入平滑。物理硬约束两层共用。接触参数化后第一层180变量；任务零空间参数化后第二层120变量。

运行方式、数学关系及精度范围见 [HIERARCHICAL_GMPC.md](HIERARCHICAL_GMPC.md)。当前版本使用集中式 CPU OSQP；下文早期 ReLU-QP 两层原型及加权分布式原型为历史独立实现，不能把它们的求解器或实验结论混用为此版本。

普通/双幅参考各两次12秒闭环均通过两层残差和主任务保持检查，四足接触持续、无力矩裁剪。位置RMSE约0.109/0.138 mm，姿态RMSE约0.0694/0.1091度；平均总耗时约47.8/171.5 ms。相比已优化加权版本，位置误差降低，姿态误差增加、计算更慢，实际关节平滑性没有整体改善。没有注入计算延迟，不能称为50Hz实时实现。

```bash
source /home/ganhe/project/b2z1_mpc/relu_hgmpc/env.sh
python relu_hgmpc/test_hierarchical_gmpc.py
python relu_hgmpc/run_hierarchical_gmpc.py --duration 12 --scale 1 --name hierarchical_demo
python relu_hgmpc/run_hierarchical_gmpc.py --controller weighted --duration 12 --scale 1 --name weighted_demo
```

完整匹配对照见 `../results/hierarchical_gmpc/RESULTS.md`。旧加权一阶GMPC与其加速数据保留，模型矩阵逐位一致测试通过。

## 历史实现与对照

集中式 GMPC 新增 `--backend reluqp`，与默认 `--backend osqp` 求解相同物理 QP。运行 `bash relu_hgmpc/benchmark_central_relu.sh`；对照报告在 `results/central_relu/RESULTS.md`。2026-11-10 各两次 12 秒闭环均通过共同残差与接触检查，总耗时均值 OSQP 54.01 ms、ReLU-QP 81.99 ms；当前预处理/矩阵重建成本尚未带来整体加速。等价预处理与计时边界见 [SPLIT_GMPC.md](SPLIT_GMPC.md)。

2026-11-10 新增：底盘＋机械臂两块动力学分布式 GMPC，包含显式接口力共识、SE(3) 误差、有限预测时域及并行局部 QP。实现范围、与旧两层 GMPC 的区别及运行方式见 [SPLIT_GMPC.md](SPLIT_GMPC.md)。实际 1 kHz 抖振审计在 `results/chatter_1khz/`，完整预测闭环在 `results/split_gmpc/`。

本目录实现用户指定的两组对照。它是 **B2+Z1 站立操作的研究原型**，不把机器人型号、约束消元、参数和低层控制不同的实验称作原论文逐项复现。

新增第三种方法：Xie 等（2025）的任务优先级运动分配＋WBC，原文 3D 位置任务及 6D 位姿扩展。见 [PRIORITIZED.md](PRIORITIZED.md)；运行 `bash relu_hgmpc/benchmark_prioritized.sh`。第三种方法的小型 WBC QP 使用 CPU quadprog，下面关于 ReLU-QP/OSQP 的说明仅针对原两种 MPC。

## 两种控制器

- `baseline`：按 ReLU-QP 论文的四足加机械臂实验思路，在站立平衡点对完整刚体动力学线性化，固定模型；跟踪全身位形/速度参考。使用 LQR 预条件化与稠密状态消元，固定 QP 矩阵离线生成 ReLU 层，在线只更新线性项与约束边界。末端轨迹通过固定机身的离线 IK 转成关节参考。
- `hgmpc`：每周期更新接触一致的动力学和末端体雅可比；以 SE(3) 对数误差构建任务空间预测。第一层跟踪末端位姿/速度，第二层保持第一层预测及任务有效输入不变，调节机身/关节姿态、内部运动和控制平滑性。两个 QP 均使用 ReLU-QP。

第二层的机身姿态权重大于机械臂姿态权重，用来体现“优先让臂完成操作、必要时机身参与”的运动分配。这里没有直接复制 Xie 等人的秩切换/SNS 算法，也未实现其全部第三任务或局部避障；它是本文选定的两层 GMPC 扩展，不是那篇非 MPC 方法的原样复现。

## 运行

工程目录：`/home/ganhe/project/b2z1_mpc/relu_hgmpc`。

```bash
bash /home/ganhe/project/b2z1_mpc/relu_hgmpc/build.sh
source /home/ganhe/project/b2z1_mpc/relu_hgmpc/env.sh
python relu_hgmpc/test_solver.py
python relu_hgmpc/test_model.py
bash relu_hgmpc/benchmark.sh
python relu_hgmpc/run.py --mode baseline --duration 12 --iterations 100
python relu_hgmpc/run.py --mode hgmpc --duration 12 --iterations 100
python relu_hgmpc/analyze.py
```

`native.cpp` 使用 C++ Pinocchio 和 MuJoCo，管理真值、刚体动力学、SE(3) 雅可比、1 kHz 物理执行与共同的关节伺服。QP 构造和实验编排目前使用 Python；CUDA ReLU-QP 使用 PyTorch。因而这还不是纯 C++/LibTorch 控制器。

复用 `b2z1_mpc` conda 环境；在其 Python 搜索路径末尾追加现有 `frictionest` 的 PyTorch 安装，不修改两个环境的软件包。`RELU_TORCH_SITE` 可以指定其他兼容安装。当前验证：Python 3.11、PyTorch 2.5.1+cu121、RTX 4060 Laptop 8 GB、FP64。没有 CUDA 时明确报错，不偷偷退回 CPU。

## 公共实验条件

- 原项目 B2+Z1 URDF/MJCF，可配置背部安装；浮动基座 `nq=25, nv=24`，18 个关节力矩。
- 四足持续站立，地面库仑系数 0.6，内接摩擦棱锥使用 `0.6/sqrt(2)`；法向力预测限制 10–800 N。
- 同一 6D 末端轨迹、同一初始状态、同一 MuJoCo 真值；位移约厘米量级、姿态数度。基线 IK 轨迹离线生成，生成时间单独记录。
- 预测域 0.3 s：10 个节点，每节点 0.03 s；控制周期 0.02 s；MuJoCo 步长 0.001 s。
- 相同关节位置、速度、力矩、基座偏移及接触力约束。MPC 力矩限为 URDF 额定值的 85%，为共同低层伺服留余量；物理端仍限制额定力矩并记录截断。
- 所有状态/控制/力约束都是 QP 约束。没有隐藏 OSQP 控制回退。OSQP 仅用于离线验证。
- 相同固定迭代预算分别比较 15 和 100 次。GMPC 有两个层级，故每周期有两个预算相同的 QP，不把二层总迭代数误称为一层。
- 模拟按固定仿真时间推进，未注入计算延迟；`controller_ms > 20` 表示不能按当前实现实时执行，不能以仿真稳定代替实时闭环稳定。

## 计时与有效性

`qp_ms` 包含 GPU 迭代、同步、结果传输和残差计算。`setup_ms` 包含缩放、稠密求解层生成及上传，GMPC 每周期两层都计入；基线该成本仅在初始化支付。`controller_ms` 包含参考查询、模型/代价/约束/消元、求解层构造、两层求解和验证。CUDA 预热、初始线性化/LQR 和预规划轨迹生成不计入在线时延。

总控制器计时从已读取的 C++ 真值/模型快照开始；公共快照获取时间在 `snapshot_ms` 中单独记录，`feedback_ms` 为两者之和。MuJoCo 积分和 1 kHz 伺服属于执行层，不计入优化时延。

记录物理单位约束最大违反量、缩放后原始/对偶残差、第二层主任务保持误差、末端误差、足端漂移及截断次数。**固定迭代次数不保证 QP 已收敛**；尤其 15 次运行必须同时看对偶残差。任何本次单轨迹差异都不足以证明 GMPC 一般优于基线，也不能把差异单独归因于 Lie 群，因为模型更新、代价与优先级也同时改变。

## 文件与出处

- `model.py`：接触动力学消元、站立点线性化、LQR 稠密消元、公共参考与 IK。
- `controller.py`：固定全身 MPC 和两层优先级 GMPC。
- `relu_solver.py`：依据公开 ReLU-QP 递推独立实现；包含变量/约束缩放、真实热启动缓冲及矩阵重建。不是未经修改的上游 Python 包。暂用固定 rho；没有原实现的自适应 rho 候选库。
- `test_solver.py`：活跃约束 QP 与 OSQP 对照、更新与热启动检查。
- `test_model.py`：动力学 KKT、体雅可比、几何线性化、非零次任务修正、主任务代价保持及实际 GMPC QP 的独立核对。
- `FORMULATION.md`：公式、适用条件及与论文不同之处。
- `../results/relu_hgmpc/`：逐周期 CSV、摘要 JSON、检查结果及对比图。

ReLU-QP 原论文：[Bishop et al., arXiv:2311.18056](https://arxiv.org/abs/2311.18056)；公开实现：[RoboticExplorationLab/ReLUQP-py](https://github.com/RoboticExplorationLab/ReLUQP-py)。

优先级运动分配参考：[Xie et al., 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12087763/)。几何层级设计依据用户 Zotero 中 **Hierarchical Geometric Model Predictive Tracking Control for Redundant Robot Manipulators in Task Space**（条目 JRM7WCEK，附件 X3PT7JII）的式 (16)、(28)–(30)。本工程没有对该本地稿件的发表状态作推断。

## 一阶 GMPC 的 OSQP 求解加速（2026-11-10）

`paper_gmpc.py` 补全了任务空间一阶误差/速度动力学，公式核对见 `PAPER_FIRST_ORDER.md`。在该相同模型上，`osqp_acceleration.py` 提供 OSQP 工作区持久化及接触/接口等式精确消元，变量420→180，全部1040行不等式保留；验收仍使用原420维QP的共同相对残差门槛2e-7。

四组对照（原版、仅复用、仅消元、组合）各在两档参考下运行两次12 s。普通幅度总耗时55.53→25.74 ms，双幅95.41→30.89 ms；跟踪RMSE变化不到1%。总耗时包含矩阵构造、消元和恢复/检查；尚未注入实际计算延迟，未达到20 ms实时周期。

```
source /home/ganhe/project/b2z1_mpc/relu_hgmpc/env.sh
python relu_hgmpc/run_osqp_acceleration.py --optimization combined --duration 12 --name combined_demo
```

完整复现使用 `test_osqp_acceleration.py`、`benchmark_osqp_acceleration.sh`、`audit_osqp_accuracy.py`、`report_osqp_acceleration.py`。推导见 `OSQP_ACCELERATION.md`，结果与逐周期数据见 `../results/osqp_acceleration/RESULTS.md`。
