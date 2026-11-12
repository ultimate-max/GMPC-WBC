# 单线程末端锚定 GMPC＋几何 WBC：实现与实测

本报告对应 B2＋Z1 的 6D 站立仿真；使用 MuJoCo 真值、Pinocchio 动力学。原论文是运动分配＋WBC，不是 MPC；本文的 paper 结果是对其方程的 B2＋Z1、6D 适配，不能代替作者代码复现，也不能据此认定原论文方法本身失败。

论文来源：[Xie 等，2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12087763/)。

## 已实现的联合任务

安装关系为 Y=T_WB T_BA，机械臂相对运动为 F=T_AE(q_a)，目标为 D=T_WE^d。反推安装基座目标 Y_req=D F^-1，并使用同一个闭合条件 Y=Y_req；不会给底盘和机械臂各自指定一个互不相关的终点。

原始接口误差 e_I=Log(Y_req^-1 Y)=Ad_F Log(D^-1 YF)。F 随机械臂构型变化，导数必须包含这一项。本实现使用拉回到统一末端度量的误差 r=Ad_F^-1 e_I，因此和直接的 SE(3) 末端误差严格等价。换对接位置本身不构成性能优势。

MPC 层级：硬物理约束 > 末端位姿/误差速度 > 连续运动分配＋柔性底盘姿态＋局部避奇异偏好 > 姿态/加速度/力矩平滑。WBC 层级：硬物理约束 > 规划末端 > 规划底盘 > 关节与力矩平滑。WBC 跟踪一致的规划参考，但全部结果对外部原始末端目标统计。

分配使用归一化底盘增量与机械臂增量的代价。机械臂权重由 Jacobian 最小奇异值平滑调整，范围 0.2–0.8；关节限位通过硬约束限制。末端误差足够小后，底盘回正权重平滑降至零；仍保留运动抑制与必要的避奇异偏好。这里不是“谁先到就硬切掉谁”的逻辑。

可操作度使用正则化 log-volume 的局部一阶偏好，不是严格的远离奇异点保证。所有动态权重、门函数与尺度均在 anchored_mpc.py 中明示。

## 单线程调度

主循环先运行 WBC，再运行到期的 MPC。MPC 只能在整数个 WBC tick 上触发。默认 WBC 200 Hz、MPC 50 Hz（4 分频），WBC 100 Hz 对应 2 分频。保留用户要求的 MPC 最低 50 Hz；预算模式用耗时 P95 决定分频并使用滞回，但不会自动降到 50 Hz 以下。

预测规模固定为 10×40 ms；改变执行频率不会改变预测时域。约束、模型、求解精度保持一致。串行总耗时包含状态读取、参考生成、QP 矩阵构造、分解/求解、认证和轨迹恢复。目标缓存与模型初始化单列在计时范围外。

**这些闭环使用受控仿真时钟，求解时仿真会等待。降低 MPC 调用频率不能消除单次求解对 WBC 的阻塞；以下超时统计不等于实际机器人延迟实验。当前版本不能宣称达到 200 Hz WBC 的硬实时要求。**

## 完整运行结果（目标均为 12 秒）

- single200_normal：12.000 s，完成。末端位置 RMS=0.398 mm，姿态 RMS=0.056°。失败运行的误差只覆盖已完成片段，不能与完整运行直接排名。
- single200_paper_shared：10.290 s，停止：Level 0 residual gate: {'qp_primal_raw': 0.046989753542344914, 'qp_primal_relative': 0.0003314908555836493, 'qp_dual_raw': 5.374851935168922e-09, 'qp_dual_relative': 5.374851935168922e-09, 'qp_normal_relative': 0.0003314908555836493, 'qp_objective': -4.3428935985580956e-08}。末端位置 RMS=4.262 mm，姿态 RMS=0.366°。失败运行的误差只覆盖已完成片段，不能与完整运行直接排名。
- single200_paper：10.645 s，停止：Measured physical state gate。末端位置 RMS=1.781 mm，姿态 RMS=0.044°。失败运行的误差只覆盖已完成片段，不能与完整运行直接排名。
- single100_normal：12.000 s，完成。末端位置 RMS=0.434 mm，姿态 RMS=0.062°。失败运行的误差只覆盖已完成片段，不能与完整运行直接排名。
- single200_hold：12.000 s，完成。末端位置 RMS=0.026 mm，姿态 RMS=0.003°。失败运行的误差只覆盖已完成片段，不能与完整运行直接排名。
- single200_push：12.000 s，完成。末端位置 RMS=0.497 mm，姿态 RMS=0.077°。失败运行的误差只覆盖已完成片段，不能与完整运行直接排名。
- single200_double：0.760 s，停止：constraints are inconsistent, no solution。末端位置 RMS=0.441 mm，姿态 RMS=0.025°。失败运行的误差只覆盖已完成片段，不能与完整运行直接排名。
- single200_repeat：12.000 s，完成。末端位置 RMS=0.398 mm，姿态 RMS=0.056°。失败运行的误差只覆盖已完成片段，不能与完整运行直接排名。

## 相同时间段对照

正常工况统一取 0–8 s；实际关节信号的频带指标统一取 1–8 s，避免把失败后的缺失数据当成零误差。这是用于定位差异的共同前缀，不能掩盖后续失败。

- Anchored GMPC + geometric WBC：末端位置 RMS 0.428 mm，姿态 RMS 0.047°；总控制 CPU 7.055 s；每周期总耗时均值/P95 4.422/16.446 ms。
  实际机械臂速度/力矩的 5–100 Hz RMS：0.031183 rad/s / 0.258870 N·m。实际腿速度/力矩：0.053573 rad/s / 2.447309 N·m。
- Paper allocation + shared geometric WBC：末端位置 RMS 1.091 mm，姿态 RMS 0.189°；总控制 CPU 2.929 s；每周期总耗时均值/P95 1.831/3.856 ms。
  实际机械臂速度/力矩的 5–100 Hz RMS：0.004675 rad/s / 0.139743 N·m。实际腿速度/力矩：0.007804 rad/s / 1.356338 N·m。
  该区间记录到底盘启停切换 3 次；这不是完整选择矩阵所有元素的切换计数。
- Paper allocation + paper WBC (6D adaptation)：末端位置 RMS 0.205 mm，姿态 RMS 0.030°；总控制 CPU 2.106 s；每周期总耗时均值/P95 1.319/2.219 ms。
  实际机械臂速度/力矩的 5–100 Hz RMS：0.015406 rad/s / 0.217598 N·m。实际腿速度/力矩：0.014993 rad/s / 4.875819 N·m。
  该区间记录到底盘启停切换 5 次；这不是完整选择矩阵所有元素的切换计数。

![跟踪与耗时](singlethread_tracking.png)

![实际关节信号](singlethread_actuators.png)

## 基线适配与结论边界

- paper：保留论文的机械臂优先、基座激活、饱和重分配、零空间可操作度和原 WBC 方程。分配 50 Hz，WBC 200 Hz；已验证抽取 allocate() 前后原控制输出完全一致。原 WBC 的基座加速度修正可能改变此前接触任务结果，相关接触残差有单独记录。
- paper_shared：上述运动分配接入同一个末端优先几何 WBC，增加发布时刻参考补偿和有界加速度的速度指令转换，属于明确标注的适配实验。它和原 WBC 基线共同报告，不能只选较差的一组比较。
- MPC 和瞬时运动分配优化的是不同问题，耗时比较是完整控制栈成本，不是同一个 QP 的求解器加速比。
- 原论文基线没有被要求跟踪这里的名义底盘轨迹。图中的底盘偏差用于展示运动分配结果，不能直接作为该基线跟踪失败的证据。耗时曲线显示每四个周期抽取的重规划周期；均值统计包含全部 WBC 周期。
- 在共同前 8 秒，MPC 的末端误差优于共用几何 WBC 的运动分配适配版，但劣于原 WBC 的论文 6D 适配版；MPC 的实际速度和力矩频带分量也没有普遍减小。因此本次试验没有证明全面性能优势。
- 共同前缀中的频带 RMS 不是纯粹的“选择矩阵抖振”。控制增益、预测、插值、物理约束和低层控制都可能影响它；还需选择矩阵开关的单因素消融。
- 候选 MPC 在双幅工况仍有层级 QP 数值失败。已证明其中一个被报告不可行的低层子问题存在可行零增量；1e-10 原单位的舍入处理可恢复该点，但后续点仍会失败。始终使用原始目标/约束的 2e-7 KKT 与任务锁标准验收，未发布不合格解。
- 历史双线程测试失败和迟到丢弃测试都保留；按照最新要求，推荐入口已经改为单线程。
- 当前结果支持继续研究连续冗余分配与预测控制，不能据此宣称 GMPC 已全面优于论文、严格避奇异、必然抑制全部关节抖振，或满足硬实时。

## 运行入口

工程路径：/home/ganhe/project/b2z1_mpc。

```bash
cd /home/ganhe/project/b2z1_mpc
source relu_hgmpc/env.sh
python relu_hgmpc/run_singlethread_anchor.py --method anchor --case normal --wbc-hz 200 --schedule budget --duration 12 --name my_anchor
python relu_hgmpc/run_singlethread_anchor.py --method paper_shared --case normal --wbc-hz 200 --schedule ratio --duration 12 --name my_paper_shared
python relu_hgmpc/run_singlethread_anchor.py --method paper --case normal --wbc-hz 200 --schedule ratio --duration 12 --name my_paper
```

原始日志、CSV、1 kHz 实际速度/施加力矩、源文件哈希、失败时刻均在 results/anchored_mpc；comparison_summary.json 提供可机器读取的汇总。