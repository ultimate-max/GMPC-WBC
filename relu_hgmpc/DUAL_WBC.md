# 分层 GMPC＋安装接口协调双 WBC：实现与验证

本轮已实现上层严格分层 GMPC、下层本体与机械臂两个局部逆动力学 WBC、安装接口协调以及力矩平滑。所有控制计算串行执行。常规轨迹、双倍幅度、末端固定、120 N 短时外推力四种工况中，双 WBC 和统一 WBC 均完成 12 秒仿真。

**结论：接口分解正确，但当前单线程实现没有带来加速，也没有证据表明拆分本身改善跟踪或抖振。论文基线在常规轨迹前 8 秒的末端误差和计算成本更低；它在后段失效，不能忽略。**

## 1. 实际架构

上层使用统一的全机器人预测变量，按硬约束 → 末端位姿及速度 → base 位姿及速度 → 可操作度/关节姿态/平滑项逐层求解。每层锁定已取得的任务输出，不以大权重代替优先级。N=10，节点间隔 40 ms。每周期更新测量状态与模型，在当前状态处构造局部仿射动力学及 SE(3) 误差；这不是完整非线性多重射击求解器。

末端误差为 $e_E=\operatorname{Log}(T_{E,d}^{-1}T_E)^\vee$，一阶模型为 $e_E\simeq\bar e_E+J_{\log}(\bar E_E)J_E\delta q$。base 同样使用 SE(3) 误差。下层保留安装锚点误差的伴随运输及其时间导数。

下层有两个局部模型、任务与约束模块，协调器通过安装处加速度一致和内力平衡，求解一个结构化分层 QP。当前不是两个互不相干的 QP，也没有 ADMM 迭代。两个局部模型共享测得的安装位姿/速度；安装处加速度通过求解一致性方程协调。

局部动力学采用 $M_Ba_B+h_B=S_B^T\tau_B+J_c^Tf+J_{IB}^Tw$、$M_Aa_A+h_A=S_A^T\tau_A-J_{IA}^Tw$，其中 $w$ 是机械臂施加在本体上的安装处六维力，表达在安装坐标系，排列为 [力；力矩]。实体惯量严格分区，未重复计入。

局部消元得到 $a_B=T_B\tau_B+W_Bw+b_B$、$a_A=T_A\tau_A+W_Aw+b_A$。利用 $J_{IB}a_B=J_{IA}a_A$，解六维系统 $(J_{IB}W_B-J_{IA}W_A)w=J_{IA}T_A\tau_A-J_{IB}T_B\tau_B+J_{IA}b_A-J_{IB}b_B$。最后在 18 维关节力矩变量上执行 EE → base → 平滑的全局层级。

本体接触消元是 30 维线性系统，机械臂惯性系统为 12 维，接口为 6 维；统一 WBC 使用 36 维接触动力学系统。当前 Pinocchio 局部模型仍保留零惯量的非所属运动学分支，再提取局部矩阵，且控制装配在 Python 中，因此分块尺寸较小并不直接等于运行更快。底层继续使用现有 C++/Pinocchio/MuJoCo 内核。

## 2. 平滑与限位

最低层包含归一化力矩增量项 $\|D_\tau^{-1}(\tau_k-\tau_{k-1})\|^2$，腿/臂力矩尺度分别为 100/10 Nm。同时限制命令力矩变化率：腿 1000 Nm/s、臂 100 Nm/s。实际执行还有相同的关节 PD 和摩擦补偿，因此命令增量受限不等于最终施加力矩必然满足相同斜率。报告单独统计 MuJoCo 中的实际关节速度与最终施加力矩。

最初仅使用一步位置约束时，两种架构都在末端固定工况 2.71 秒失效。独立线性规划证实：机械臂第 5 个关节的位置上界与加速度下界冲突；不是接口力符号、ADMM 或求解容差问题。

修正是在上下层同时加入 $-\eta(q^+-q_{\min})\leq v^+\leq\eta(q_{\max}-q^+)$，$\eta=5\,s^{-1}$，关节限位保留 0.02 rad 裕度。下层用 $q^+=q+hv+\frac12h^2a$、$v^+=v+ha$；上层逐节点约束预测 q/v。两种架构完全一致，且在重建后的物理坐标中独立验算。该约束是提前减速，不是放宽原限位；它不构成非线性闭环的全局递归可行性证明。

实现过程中修正过上层减速约束的下界符号；最终结果均已重跑。修正前数据另存 results/dual_wbc_before_damper_correction，不纳入正式结果。

## 3. 等价性和闭环

32 个扰动状态的模型分区/矩阵/任务检查通过；四种闭环各抽取 48 个测量状态，共 192 个同输入重放点，使用相同参考、上一时刻力矩、约束和容差，交替两种求解顺序。最大力矩差 1.68e-09 Nm，最大加速度差 1.46e-09，安装加速度残差 4.68e-13。单个输入下的解在数值精度内等价。

同输入重放的完整 WBC 平均耗时：双 WBC 1.410 ms，统一 WBC 1.210 ms。闭环长期出现的小差别应归因于数值扰动经参考重规划、活动约束和仿真反馈传播，不能当作双 WBC 优势。

- normal：双/统一 WBC 均 12.000/12.000 s；末端位置 RMS 0.5627/0.5608 mm，姿态 RMS 0.5180/0.5149°。论文适配版完成 10.645 s，停止原因：Measured physical state gate。
- double：双/统一 WBC 均 12.000/12.000 s；末端位置 RMS 1.6048/1.6052 mm，姿态 RMS 0.2636/0.2656°。论文适配版完成 7.205 s，停止原因：Measured physical state gate。
- hold：双/统一 WBC 均 12.000/12.000 s；末端位置 RMS 1.8122/1.8122 mm，姿态 RMS 0.5301/0.5301°。论文适配版完成 12.000 s。
- push：双/统一 WBC 均 12.000/12.000 s；末端位置 RMS 0.6257/0.6239 mm，姿态 RMS 0.5206/0.5175°。论文适配版完成 10.645 s，停止原因：Measured physical state gate。

## 4. 相同常规轨迹前 8 秒比较

跟踪误差和累计控制 CPU 使用 [0,8) s；实际高频量统计 [1,8) s。高频 RMS 为各关节实际信号经四阶 5–100 Hz Butterworth 零相位滤波后，对时间和关节共同求 RMS。它包含运动瞬态，不能单独证明选择矩阵切换抖振。

- **分层 GMPC＋双 WBC**：末端位置 RMS 0.6808 mm，姿态 RMS 0.6338°；base 位置 RMS 0.1036 mm；累计控制 CPU 6.603 s。
  实际腿/臂速度高频 RMS 0.00261/0.02470 rad/s；实际腿/臂力矩高频 RMS 0.3957/0.1390 Nm。
- **相同 GMPC＋统一 WBC**：末端位置 RMS 0.6784 mm，姿态 RMS 0.6300°；base 位置 RMS 0.1028 mm；累计控制 CPU 6.336 s。
  实际腿/臂速度高频 RMS 0.00255/0.02234 rad/s；实际腿/臂力矩高频 RMS 0.3947/0.1332 Nm。
- **论文运动分配＋原 WBC 的 6D 适配版**：末端位置 RMS 0.2050 mm，姿态 RMS 0.0301°；base 位置 RMS 43.1844 mm；累计控制 CPU 1.962 s。
  实际腿/臂速度高频 RMS 0.01499/0.01529 rad/s；实际腿/臂力矩高频 RMS 4.8758/0.2175 Nm。
- **论文运动分配＋共同几何 WBC 适配版**：末端位置 RMS 1.0907 mm，姿态 RMS 0.1887°；base 位置 RMS 43.4006 mm；累计控制 CPU 2.814 s。
  实际腿/臂速度高频 RMS 0.00780/0.00467 rad/s；实际腿/臂力矩高频 RMS 1.3563/0.1397 Nm。

论文适配版不独立跟踪预设 base 位姿，所以它的 base 偏差不能作为公平的劣势结论。该版本在 normal/push 后段和 double 工况触发物理状态门槛；共同 WBC 适配版还存在末级 QP 数值残差越界，不应把这一实现问题归因于论文算法。所有失败均保留状态、日志及可获取的失败 QP。

本轮对照不能证明“GMPC 在末端精度、全部关节抖振和计算效率上都优于论文方法”。腿部高频力矩等指标改善，机械臂速度高频量和末端误差并非全面改善。

## 5. 平滑消融与计算口径

- 软力矩增量权重 1：全 12 秒末端位置 RMS 0.5627 mm；腿/臂实际速度高频 RMS 0.00208/0.01987 rad/s；腿/臂实际力矩高频 RMS 0.3159/0.1228 Nm。
- 软力矩增量权重 0：全 12 秒末端位置 RMS 0.5608 mm；腿/臂实际速度高频 RMS 0.00204/0.01807 rad/s；腿/臂实际力矩高频 RMS 0.3151/0.1181 Nm。

该消融只移除最低层的软力矩增量代价，保留硬力矩斜率、限位减速和参考连续化；不等于关闭全部平滑。软项只能利用上级任务剩余自由度，不能牺牲末端/base 最优任务输出来任意降低力矩波动。

在线总耗时含模型/状态读取、参考插值、矩阵构造、分解、层级求解与残差校验；不含启动、文件写出、MuJoCo 积分。WBC 本身与 WBC＋参考生成耗时分别记录。上层 solve_ms 包含额外几何任务构造，因此不能再与 task_construction_ms 直接相加。

调度设置为 GMPC 50 Hz、WBC 200 Hz、执行器 1 kHz，保持单线程。仿真会等待计算完成，因此这只是受控闭环验证，不能声明硬实时达标。常规轨迹约每四个 WBC 周期中的一个承担 MPC，超过 5 ms 周期预算；即使平均 GMPC 小于 20 ms，也会阻塞 WBC。要达到真实频率仍需调度/求解优化或不同执行架构。

## 6. 复现与文件

工程：/home/ganhe/project/b2z1_mpc。新入口 run_dual_wbc_singlethread.sh 默认运行带限位减速的双 WBC normal 工况。

```bash
cd /home/ganhe/project/b2z1_mpc
bash run_dual_wbc_singlethread.sh gmpc_dual normal my_dual
bash run_dual_wbc_singlethread.sh gmpc_unified normal my_unified
bash run_dual_wbc_singlethread.sh paper normal my_paper
source relu_hgmpc/env.sh
python relu_hgmpc/run_damped_suite.py --rerun
python relu_hgmpc/replay_dual_wbc.py
python relu_hgmpc/analyze_dual_wbc.py
```

- relu_hgmpc/strict_gmpc.py：严格 EE → base → 其余任务层级与预测限位减速。
- relu_hgmpc/dual_wbc.py：本体/臂局部 WBC、接口消元、平滑、约束和物理验算。
- relu_hgmpc/run_dual_wbc_comparison.py：同一调度、执行器、目标与数据记录。
- relu_hgmpc/test_dual_wbc.py、replay_dual_wbc.py：模型等价与同输入闭环重放。
- results/dual_wbc/comparison.json：完整数值、环境、残差及失败状态。
- results/dual_wbc/*_trace.npz：1 kHz 实际执行信号和 200 Hz 控制/参考状态。
- results/dual_wbc/*.csv、*.json、*.log：逐周期计时、计划残差和原始日志。

论文参考基线为 Prioritized Multi-task Motion Coordination of Physically Constrained Quadruped Manipulators 的本工程 B2+Z1 六维适配；并非作者原始代码或其原实验的逐项复现。当前没有据此做新颖性或全局稳定性声明。
