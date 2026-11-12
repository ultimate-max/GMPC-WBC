# B2 + Z1 层级 GMPC 与几何 WBC（仅源码）

保留 C++／Python 源码、构建脚本、测试、配置、URDF／MuJoCo 模型描述、文档与依赖版本。实验轨迹、统计输出、图表、视频、编译产物、第三方库副本及本地 CUDA 工具链已从工程移除。

## 当前入口

- 构建：`bash relu_hgmpc/build.sh`
- 运行：`bash run_mpc_wbc.sh --name my_run --case normal`
- 统一 WBC 对照：在运行命令中加 `--wbc-mode unified`。
- 默认框架：独立 MPC 线程与原生 WBC 线程，通过最新策略缓冲交换；MPC 目标 50 Hz，WBC 目标 400 Hz。
- 当前验证范围为四足站立时的末端跟踪；后续行走复用独立参考工程 `/home/ganhe/project/legged_control_ros2`。

上层使用末端任务优先、其次本体位姿的层级 GMPC；下层保留统一及双几何 WBC。具体构造见 [MPC_WBC_RUNTIME.md](relu_hgmpc/MPC_WBC_RUNTIME.md)，其余求解器与论文对照代码保留在 `relu_hgmpc/`、`paper_repro/` 和 `src/`。

## 新环境恢复

这是源码工程，不包含可直接执行的二进制。依赖版本见 `dependencies.lock.json`、`environment.yml`、`environment-linux-64.lock` 和 `pip-freeze.txt`。精简操作未删除工程目录外的 Conda 环境，也未更改参考行走工程。

1. 安装匹配的 Python 3.11、Pinocchio、MuJoCo、NumPy／SciPy、OSQP、quadprog 等依赖。原生 WBC 对接 quadprog 0.1.13 的 C 内核，换版本前需验证。
2. 按 `dependencies.lock.json` 中的提交恢复需要的第三方库。原型安装入口是 `scripts/setup.sh`；acados 的安装入口是 `paper_repro/setup.sh`。GPU 对照所需 CUDA 工具链按各构建脚本另装，不随源码携带。
3. 模型的外观网格从 Unitree 仓库恢复。`scripts/prepare_model.py` 可按 `config/model.json` 重建组合模型；模型描述和安装配置已保留。
4. 构建当前控制器，再运行上面的主入口。部分历史脚本仍含原 WSL 绝对路径；换用户名或目录时需检查 `env.sh`、构建脚本及模型资源路径。`relu_hgmpc/CMakeLists.txt` 的 `TORCH_HEADERS` 需指向包含 pybind11 头文件的有效位置。

第三方 OSQP 的本地修改完整保存在 `relu_hgmpc/cuda_fp64_precision.patch`，应用前按锁定提交检出相应仓库。精简时确认补丁与实际修改完全一致。

## 数据与版本管理

`results/` 是运行时输出，不再保存在源码库；历史报告、绘图与回放脚本需要重新运行实验生成输入。文档里的既往实验数字是历史说明，本次精简后没有重新编译或复现实验。

本地 Git 历史已过滤掉数据和依赖，仅保留源码历史，提交哈希会变化；远端地址保留，没有自动推送。详细变更见 [CLEANUP.md](CLEANUP.md)。
