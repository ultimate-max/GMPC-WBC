# MPC 生产策略、WBC 独立执行

2026-11-12，工程：`/home/ganhe/project/b2z1_mpc`。

本次参考 `/home/ganhe/project/legged_control_ros2` 的控制器结构重构执行层，没有替换几何控制问题。入口为 `run_mpc_wbc.sh`，原 Python 双线程实验保留在 `relu_hgmpc/run_buffered_wbc.py`，供诊断复现。

## 运行

```bash
cd /home/ganhe/project/b2z1_mpc
bash run_mpc_wbc.sh --name my_normal --case normal --duration 12
bash run_mpc_wbc.sh --name my_unified --wbc-mode unified --duration 12
```

`--case` 还支持 `double`、`hold`、`push`。其中 `hold` 是末端位姿固定而本体目标继续运动，不是整机静止。默认 WBC 400 Hz、MPC 目标频率 50 Hz，MuJoCo/关节执行器 2 kHz。结果存入 `results/buffered_wbc/`，包含原始状态、实际关节速度、执行器力矩、时序、残差和求解日志。

## 对照参考工程后做出的修改

参考依据是 `src/legged_controllers/src/LeggedController.cpp`：`on_activate()` 等待首条策略，`update()` 更新观测、切换策略并按观测时间执行 `evaluatePolicy()`，`setupMrt()` 用独立线程运行 `advanceMpc()`。参考控制器传入 WBC 的是名义状态、输入和接触模式；其质心模型和任务定义与本工程不同，未直接替换为该算法。当前参考目录内没有 OCS2 MRT 底层实现文件，因此这里只依据可见的调用链重构，不声称复制了其内部锁实现。

现在的数据通路是：

```text
最新实测状态 ──→ MPC 生产者（50 Hz 目标频率）
                      │ 完整、带时间戳的预测策略
                      ▼
                 最新策略缓冲区
                      │ 非阻塞取新策略
                      ▼
          C++ WBC 消费者（400 Hz）
        按当前时间采样 → 几何 WBC → 执行器
                      │
                      └──→ 最新观测缓冲区
```

- 两个缓冲区均只保留最新值，避免 FIFO 积压。策略发布前完成复制和检查；高频线程只交换不可变对象的指针，使用 `try_lock`，未取得锁时继续原策略。
- 启动控制前取得首条合格策略。新策略没到时继续采样旧策略，不等待 MPC。序号倒退、过时或维度错误的策略不能执行。
- MPC 从实测状态、对应时间戳和上一条 WBC 力矩出发，不把上一条“已提交参考”的未来状态当作物理初值。
- WBC 直接按策略的绝对时间在 SE(3) 图内采样，重建足端无滑动参考。去掉人为的 20 ms 延后释放和每次更新重新启动的 40 ms 过渡桥。
- WBC 的观测、插值、接触运动学、几何任务、动力学、层级求解和 MuJoCo 步进均在原生 C++ 线程中，不调用 Python。MPC 暂时保留 Python 编排与现有原生矩阵构造；两个线程分别拥有模型和工作区。
- 仍保留预测节点之间的五次插值；策略切换不额外保证 C2 连续。关节力矩变化限制和平滑目标仍在 WBC 内执行，不能把线程重构本身视为消除抖振的证明。

## 保留的控制问题与数值验证

上层仍是每周期一次线性化的冻结仿射预测模型、一阶几何误差、10 个 40 ms 节点；不是收敛到解的非线性 SQP。优先级仍为物理约束、末端位姿/速度、本体位姿/速度、冗余利用和平滑。下层双 WBC 仍通过安装点六维作用力和加速度一致性协调，保持原来的统一层级优化，并非独立计算两组力矩后直接拼接。关节速度阻尼、摩擦约束、力矩和力矩变化限制未放宽。

原生求解调用当前安装的 `quadprog 0.1.13` 的同一个 C 数值内核，固定版本并检查导出符号；没有换成更宽松的求解器。此内部接口需要版本锁定，升级依赖后须重跑一致性测试。原生实现没有原 Python 实现的 SLSQP 数值备用路径：内核或残差检查失败时停止该仿真，不能冒充硬件安全降级控制器。

120 组同输入回放（双 WBC、统一 WBC各60组）验证：最大力矩差低于 `5e-10 Nm`，最大接触力差约 `2.1e-9 N`，任务矩阵差低于 `3.5e-10`。双 WBC 内核平均约 `0.194 ms`，原 Python 实现约 `1.338 ms`。这仅是内核回放时间，闭环完整耗时另计。

专项测试覆盖不可变复制、错误包、序号倒退、生产者连续占用 Python 锁时消费者继续运行，以及策略过期停止。无新策略时约 0.3625 s 触发预测时域余量检查，没有无限外推旧指令。

## 实测结果

普通轨迹，12 s、双 WBC：

- 4800 个控制周期；实际约 `400.07 Hz`，策略安装约 `49.76 Hz`。
- MPC 总耗时平均 `8.39 ms`、P99 `16.01 ms`；包含观测读取、模型计算、矩阵构造、求解、检查和策略发布。
- WBC 执行链连同 MuJoCo 步进平均 `0.381 ms`、P99 `0.526 ms`。该项从周期开始计到物理步进完成，最后一条日志对象入库不在此分项内；实际频率覆盖整个循环。
- 末端位置 RMS `0.300 mm`，姿态 RMS `0.097°`；本体位置 RMS `0.048 mm`。
- 四足接触保持，执行器无力矩截断；这次运行没有完整周期截止时间违约。

统一 WBC 同条件：末端位置 RMS `0.299 mm`，完整执行链平均 `0.368 ms`，也达到约400 Hz。两者的精细差异包含异步策略到达时刻的变化，不能当作双 WBC 改进控制精度或加速求解的证据。

双幅和120 N短时外力扰动场景也完成12 s，保持四足接触。双幅场景 MPC 最大耗时 `64.1 ms`，但 WBC 未等待求解，完整周期没有违约。其平均策略安装频率约49 Hz，并不代表每条策略都满足20 ms截止时间。

固定末端、移动本体场景同样运行12 s，但有3个0.5 ms物理步出现前左足失接触，累计 `1.5 ms`；末端位置 RMS `1.77 mm`，本体位置 RMS `20.67 mm`。本体为低优先级任务，可能因末端和关节约束让步，但短暂失接触依然是需要保留的物理局限，不能由小 QP 残差排除。

以上是 WSL、单次确定轨迹、正常操作系统调度的有限时长仿真验证，不是硬实时保证。物理仿真仍归控制线程所有，控制超时不会自动模拟独立硬件持续运动；没有做独立执行器时钟/HIL验证。本次改善的主要证据指向执行框架与原生控制循环，不是新的 GMPC/ADMM 数学加速结论。

## 文件与检查

- `relu_hgmpc/run_mrt_wbc.py`：MPC 生产者与离线结果汇总。
- `relu_hgmpc/runtime_native.hpp`：策略/观测缓冲和 C++ 消费线程。
- `relu_hgmpc/wbc_native.cpp`：几何双/统一 WBC 内核。
- `relu_hgmpc/test_native_wbc.py`：原实现与 C++ 内核同输入验证。
- `relu_hgmpc/test_native_runtime.py`：生产消费、错误包及过期处理验证。
- `results/buffered_wbc/native_equivalence.json`、`native_runtime_tests.json`、`native_mrt400_*.json`：原始检查与结果。

构建和检查：

```bash
cd /home/ganhe/project/b2z1_mpc
bash relu_hgmpc/build.sh
source relu_hgmpc/env.sh
python relu_hgmpc/test_native_wbc.py
python relu_hgmpc/test_native_runtime.py
```
