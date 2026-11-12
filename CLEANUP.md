# 2026 年 11 月 12 日代码清理

从活动源码目录移除 24 个废弃文件，并把两个回归测试仍需要的历史实现移到 `relu_hgmpc/tests/fixtures/`。所有移除源码已逐文件校验归档至 `archive/retired_2026-11-12.tar.gz`，清单为 `archive/cleanup_2026-11-12.json`。

- 移除一次性源码生成、改写与注册脚本；已生成的有效实现保留。
- 移除散落备份与旧 Python 缓冲运行入口；抽出完全相同的 DenseTarget、physical_cores、stats 到 `runtime_support.py`，主程序和论文对照不再通过历史入口导入功能。
- 移除本轮已放弃的爬行原型。原生四文件恢复自 `results/loco_manipulation/standing_native_before_loco.tar.gz`；可变接触等未完成接口不再混入默认运行核心。
- 保留论文适配基线、结构化 ADMM、ReLU-QP、OSQP 及其复现实验。`paper_gmpc_weighted_legacy.py` 是层级方法测试依赖的对照实现，继续保留。
- 保留全部原始实验结果与失败记录，不删除第三方库和模型。清除 57 个 Python 缓存文件。

当前主入口：`run_mpc_wbc.sh`。后续移动操作以用户提供的 `legged_control_ros2` 为基础，不继续该临时爬行原型。

验证结果在 `results/cleanup_2026-11-12/`，清理不会自动提交或推送 Git。

## 清理后验证

- 重新编译原生控制内核通过；158 个 Python 文件语法检查通过，无已删除模块的残留导入。
- 抽出的三个公共函数与旧实现 AST 一致；论文基线、锚点任务与历史 WBC 回归通过。
- 统一与双 WBC 各 60 组输入的原生／Python 力矩差均小于 5×10⁻¹⁰ Nm。
- 策略缓冲、线程独立运行、乱序拒收与策略到期检查通过。
- 统一与双 WBC 各完成 12 秒站立末端跟踪：均保持 4 足接触、无求解失败、无力矩截断。此项是清理回归，不代表已验证行走或抓取。
- 本次机器可读摘要：`results/cleanup_2026-11-12/validation.json`；历史验证结果文件保持原样。
