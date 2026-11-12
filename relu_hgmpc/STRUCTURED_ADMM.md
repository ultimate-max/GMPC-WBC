# 固定 QP 结构化 ADMM 原型

本实验在现有 B2＋Z1 几何 MPC 的**同一批 QP**上比较求解算法。它没有修改 MPC＋WBC 闭环、预测模型、物理约束或任务层级，也不是 DWMPC 论文的完整复现。

## 实现范围

`structured_admm_native/solver.cpp` 实现 C++ 近端 ADMM：对冻结的凸 QP 做线性系统更新、区间投影和对偶更新，不在 ADMM 每一轮重新调用非线性动力学或重新线性化。没有 OSQP fallback、参考解注入或活跃面直接求解。

当前在线模型本来就每次规划只构造一次冻结线性化。接触等式已通过接触 KKT 精确消元，预测状态继续凝聚为力矩变量。默认 N=10、dt=0.04 s、预测时域0.4 s，完整问题180变量、1340双侧约束行。本原型利用的是固定 QP 的分解复用与预条件；目前仍为稠密凝聚求解，**没有实现支链分解或预测时域并行**。

坐标预条件使用可逆变换，同时变换 H、g、G，并将解与乘子恢复到原坐标。半正定层级任务在构造坐标变换的辅助矩阵时使用shift，但求解的原目标不增加该shift惩罚。ADMM的近端项同样属于迭代算法，不是增加原问题的姿态/力矩正则项。原加权QP既有的1e−9 Hessian正则保留不变。

同一个 QP 内，矩阵和rho固定时数值分解可复用；rho更新会重新分解。跨QP的模型、约束、层级坐标通常改变，因此重新执行必要预处理和数值分解，全部计时。不存在无条件跨周期复用旧因子的假设。

## 同批语料与公平性

`capture_structured_qps.py`读取历史 `online_wbc_mpc_50hz/{normal,double,push}_fixed50_r1_trace.npz`，不推进仿真。每工况12个状态，时刻0、0.5秒为calibration，其余10个为heldout。

每个状态保存一个完整加权QP和三个固定参考路径上的层级QP。总计144个：24个calibration、120个heldout。层级为基座任务、末端任务、平滑任务；前两级保持半正定原目标。记录器观察原 `HierarchyQP.solve` 的实际路径，不替换它的求解算法，不修改H来获得正定性。后续两种算法均解已固定的同一个约化问题，不能将这项验证表述为两种完整在线层级路径已经等价。

语料保存原矩阵、参考解与乘子、任务矩阵、约化坐标到原力矩坐标的映射、维数和SHA256。参考解仅用于离线误差比较，不参与被测初值、活跃集或矩阵分解。

正式v2被测策略：

- `osqp_preconditioned`：与ADMM相同的坐标预条件后，仍使用全部约束的OSQP；用于区分预条件收益和算法本身收益。
- `structured_admm`：新原生近端ADMM。
- `osqp_screened`：当前白化、活跃面证书和约束生成的OSQP路径，仅用于正定QP；不能单独代表完整三级层级。
- `existing_hierarchy`：现有最小二乘／quadprog精修＋末级OSQP混合算法；这是实际工程层级基线，并非纯OSQP。

原始全约束`osqp_full`仅保留于v1探索记录，不重跑为正式主基线。首次测试发现OSQP精确不可行证书受层级继承近零行（约1e−15舍入）影响，正式v2将`eps_prim_inf/eps_dual_inf=1e−30`，避免提前将容差内的继承问题判为精确不可行。原QP矩阵和边界保持不变，最终仍使用共同原问题残差验收。这属于基线配置诊断修复，不代表OSQP算法失败。

所有被测求解从零原始/对偶变量开始；OSQP可以按维数复用工作区、符号结构和适应后的rho，数值更新计时；ADMM每QP从配置rho开始。策略运行顺序轮换，并绑定同一CPU、使用单线程BLAS。结果中记录这些差异，不将它称为跨周期热启动实验。

共同验收使用原坐标 primal/dual/normal 残差均≤2e−7，同时原始物理不等式违反≤2e−7。内部 `solved` 状态不等于通过。参考目标差与任务输出差单独报告；半正定问题的解向量可以不唯一，但真正任务零空间中的解差应满足C差为零，不能用解不唯一解释非零任务输出差。当前KKT容差在病态尺度下不保证任务输出等价，闭环前还需独立任务输出验收。

## 计时口径

完整加权周期时间为：状态动力学评估＋公共矩阵构造＋完整solver wall。

完整层级周期时间为：状态评估＋公共构造一次＋各级H/约束约化与QR任务锁＋各级完整solver wall。`hierarchy_partition_ms`已包含在`snapshot_common_construction_ms`，不能再加一遍。不得直接累加每级QP的`total_ms`来充当完整层级周期；benchmark的`cycles`已执行正确归并。

solver wall包含数据转换、预条件、必要的矩阵分解/更新、迭代、rho变化导致的重分解、polish（若开启）及独立原问题证书。原生内核有单独factor/check/iteration计时；其中`factor_ms`包含`coordinate_factor_ms`。OSQP的setup/update不等于纯分解时间，solve也可能包含rho更新和额外分解。

公共构造从历史状态回放测量，重复取中位数并为双方加入同一公共项。这是固定QP回放的配对总成本估计，尚未包含MPC轨迹恢复、WBC、MuJoCo或真实在线调度延迟。预热、参考求解、文件读写和语料哈希单列排除。

报告同时展示全部尝试耗时、成功耗时、失败率、配对共成功子集速度比和完整原始失败记录。不能只取成功样本均值宣称加速。

## 验证与复现

`test_structured_admm.py`包含9项数值用例，本轮全部通过，覆盖正定/半正定目标、活动上下界、等式/零行、固定rho因子复用、热启动及精确坐标预条件等。它独立检查原QP目标、残差、乘子与输入不变性。另已只读审查原目标与乘子恢复及计时归并；这些检查不替代完整机器人QP的共同精度验收。

```bash
cd /home/ganhe/project/b2z1_mpc
source relu_hgmpc/env.sh
cmake -S relu_hgmpc/structured_admm_native \
      -B relu_hgmpc/structured_admm_native/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
      -DPython3_EXECUTABLE="$(which python)"
cmake --build relu_hgmpc/structured_admm_native/build -j2
python relu_hgmpc/test_structured_admm.py
python relu_hgmpc/capture_structured_qps.py --repeats 3
python relu_hgmpc/benchmark_structured_admm.py \
      --split heldout --repeat 3 \
      --solvers osqp_preconditioned,structured_admm,osqp_screened,existing_hierarchy \
      --rho .1 --adaptive 50 --preconditioner -2 \
      --output results/structured_admm/heldout_reproduction
python relu_hgmpc/report_structured_admm.py \
      --input results/structured_admm/heldout_reproduction/results.json \
      --output results/structured_admm/reproduction_report
```

正式参数为rho=0.1、adaptive_interval=50、preconditioner=-2：完整加权QP和层级末级shift=0，前两级半正定任务的坐标预条件shift=1e−6。参数只在calibration确定；但OSQP基线证书配置根据首次测试诊断后修复，因此v2是修复后的复测，不能称一次盲测。benchmark拒绝复用已有`records.jsonl`的输出目录，复现时选择新的目录。CMake默认使用环境内pybind11头文件、Eigen、BLAS，并启用当前CPU向量指令；跨机器复现需核对这些配置。

正式报告默认读取`results/structured_admm/heldout_v2/results.json`，输出`RESULTS.md`、`summary.json`和静态科学图`comparison.png`。源代码和语料哈希均保留以审查一致性。

## 闭环接入条件

先要求全部heldout QP和完整周期通过共同精度门槛，并在全覆盖配对集合上，完整周期均值与P95均优于已优化工程基线：加权问题对比`osqp_screened`，完整层级对比`existing_hierarchy`。每周期胜率另列。若只超过未预条件的全约束OSQP，不足以替换现有工程。

闭环接入还必须通过独立的**任务输出精度验收**：预先确定各层任务输出允许误差，再检查相对参考任务输出和层级锁定误差。当前报告的自动gate只计算共同KKT/物理约束与速度条件，尚未将任务输出阈值实现为自动判据，因此即便这些数值条件通过，也不能单独作为闭环接入许可。未制定或未通过任务输出验收时仍保持现有闭环。

满足门槛后仍需独立运行MPC＋WBC闭环，比较跟踪、实际关节抖振、接触和时限；本固定QP报告不推断这些闭环结论。
