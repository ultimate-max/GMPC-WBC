# 层级 GMPC 的 CUDA OSQP 接口

当前默认仍为 CPU。GPU 试验入口与 CPU 共用模型、两层目标、物理约束、N=10和dt=30ms；在 `run_hierarchical_gmpc.py` 中显式选择 `--algebra cuda`。原坐标原始/驻点/法锥残差及完整120行任务保持的门槛均为2e-7。没有 CPU 回退。

```bash
cd /home/ganhe/project/b2z1_mpc
source relu_hgmpc/env_cuda_osqp.sh
python relu_hgmpc/run_hierarchical_gmpc.py --algebra cuda --duration 2 --scale 1 --results-dir cuda_osqp --name gpu_demo
python relu_hgmpc/run_hierarchical_gmpc.py --algebra builtin --duration 2 --scale 1 --results-dir cuda_osqp --name cpu_demo
```

`env_cuda_osqp.sh` 仅在当前进程中加载工程私有扩展，复用已有frictionest环境提供的CUDA 12运行库，并将无显式选择时的OSQP后端固定为builtin。`env.sh` 与原Python环境的安装包保持不变。

## 实际使用的版本

- OSQP Python包装1.1.3，对应核心v1.0.0。
- CUDA 12.1，RTX 4060 Laptop，双精度FP64。
- 官方预编译osqp-cu12扩展为FP32，因此保留在`deps/osqp_cuda`，不用于本次与CPU双精度的正式对照。
- 官方源码FP64原始内层下限构建保留在`deps/osqp_cuda_fp64`；仅收紧内层下限的构建在`deps/osqp_cuda_fp64_tight`。
- 最终实验使用`deps/osqp_cuda_fp64_safe`。两个本地修补均记录在`cuda_fp64_precision.patch`，不能将结果描述为未经修改的官方预编译版本。

## 两个必要修补

1. `OSQP_CG_TOL_MIN`从1e-7收紧至1e-12。默认下限使部分状态在原坐标中的驻点残差停于约1e-6，即使外层继续收紧容差也无法通过2e-7验收。此修改增加内层可达到的精度，没有放宽外部门槛或改变QP目标。
2. 对全局CUDA库句柄增加串行工作区引用计数。原版会在任一工作区清理时释放全局句柄，另一个层级仍持有相关资源，导致后续矩阵更新崩溃。当前补丁适用于本试验单GPU、顺序执行的多个工作区；未验证多线程或多GPU共享。

源码位于`third_party/osqp_python_cuda113`（tag v1.1.3）和`third_party/osqp_core_cuda100`（tag v1.0.0）。工具链独立保存在`toolchains/cuda121`，包含CUDA 12.1的nvcc/cudart-dev/cudart-static及GCC 12，不替换原控制器环境的编译器。

在现有工程依赖齐全时，可用`bash relu_hgmpc/build_cuda_osqp_fp64.sh`复现构建。脚本先应用可重复的补丁，再以`OSQP_USE_FLOAT=OFF`构建，保留wheel。构建搜索路径只含Linux工具目录，以避免CMake扫描WSL映射的WindowsApps目录。

## 验证和计时

`test_cuda_osqp.py`核对解析边界QP及五个真实MPC状态；保存原始QP矩阵并比较CPU/GPU首步力矩。`report_cuda_osqp.py`汇总同条件短闭环。结果在`results/cuda_osqp/RESULTS.md`，包括原始FP64默认内层下限失败记录及生命周期问题的崩溃栈。

CUDA prepare/update、warm start和solve前后显式同步；控制计时包含矩阵构造、传输、两层求解、乘子恢复和原坐标验收，启动缓存和控制器初始化摊销。每次内部求解都有相同的12000次迭代与30秒上限，原问题不达标则停止。仿真推进不计入控制计算，未模拟实际计算延迟。

CUDA执行的是每个集中式QP内部的ADMM/共轭梯度运算；两层按优先级顺序求解。此接入不改变底盘与机械臂的运动分配结构。

本轮是两档幅值各2秒的接入检查和回放验证，未完成GPU的12秒全轨迹或延迟闭环验证。实测结果应与同一2秒任务的CPU结果比较，不能与以前12秒的均值直接比较。
