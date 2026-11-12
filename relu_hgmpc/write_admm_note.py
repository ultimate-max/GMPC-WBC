import json,pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/dual_wbc'
probe=json.loads((out/'admm_probe.json').read_text());rows=probe['results'];ok=[r for r in rows if r['converged']]
last=[r['history'][-1] for r in ok]
summary=dict(tested=len(rows),converged=len(ok),iterations=[min(r['iterations'] for r in ok),max(r['iterations'] for r in ok)],
    total_ms=[min(r['total_ms'] for r in ok),max(r['total_ms'] for r in ok)],
    worst_torque_difference=max(r['torque_difference'] for r in last),
    worst_hierarchy_lock=max(r['hierarchy_lock'] for r in last),
    scope=probe['scope'])
(out/'admm_probe_summary.json').write_text(json.dumps(summary,indent=2))
text=r'''# 两块 WBC 是否适合 ADMM：推导与离线探查

**可以构造，而且在当前固定测量状态的凸 QP 上，比直接对非线性 MPC 使用 ADMM 更容易给出优化层面的收敛条件。但当前单线程、小规模 WBC 没有显示出计算优势，暂不替换已经跑通闭环的精确接口消元。**

这里核验的是本工程的可分结构与有限迭代风险，不是全面文献综述或新颖性证明。外部理论来源和本工程推导、实验分别列出。

## 1. 两块如何分

保留本体与机械臂局部动力学消元，只不消去接口内力。定义

\[
x_B=[\tau_B;w_B],\qquad x_A=[\tau_A;w_A].
\]

实际代码对力矩与内力进行量纲缩放。本体有 12 个关节力矩与 6 个接口力变量；机械臂有 6 个关节力矩与 6 个接口力变量。两个 w 都表示“施加在本体上的力”；机械臂动力学中已使用负号，不能再把协调条件误写成 w_B+w_A=0。

局部加速度由各自动力学得到。协调条件是

\[
J_{IB}a_B(x_B)-J_{IA}a_A(x_A)=0,\qquad w_B-w_A=0.
\]

因此每层问题写成

\[
\min_{x_B\in\mathcal C_B,\,x_A\in\mathcal C_A}
f_B(x_B)+f_A(x_A),\qquad E_Bx_B+E_Ax_A=c.
\]

安装处 6 维运动和 6 维内力，一共 12 个协调残差。接触动力学、摩擦锥、力矩/变化率、关节位置/速度/加速度、限位减速分别留在局部约束集。接口必须使用同一坐标系与同一点，作用反作用关系在动力学中明确处理。

Lie 群仍用于位姿误差、雅可比和坐标运输。每周期在测量状态处形成 QP 后，ADMM 操作的是欧氏向量和线性接口条件，不直接平均 SE(3) 变换矩阵。

## 2. 标准两块更新与适用边界

使用缩放对偶变量 u，可写为

\[
x_B^{j+1}=\arg\min_{\mathcal C_B}f_B(x_B)+\frac\rho2\|E_Bx_B+E_Ax_A^j-c+u^j\|^2,
\]
\[
x_A^{j+1}=\arg\min_{\mathcal C_A}f_A(x_A)+\frac\rho2\|E_Bx_B^{j+1}+E_Ax_A-c+u^j\|^2,
\]
\[
u^{j+1}=u^j+E_Bx_B^{j+1}+E_Ax_A^{j+1}-c.
\]

这是标准两块 ADMM。固定状态下，任务平方范数为凸二次目标，局部硬约束为线性多面体；若可行、存在鞍点，并按相应条件求解局部子问题，可使用标准凸 ADMM 的残差与目标收敛结论。该结论不等于固定 5 次迭代可行，更不等于机器人非线性闭环稳定。[Boyd 等，§3.1–3.3](https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf)

## 3. 严格优先级不能改成加权和

完整方案必须按以下顺序运行：

1. 全部硬约束下求末端任务的最优输出。
2. 将末端最优输出变成硬任务锁，再求 base 任务。
3. 锁定前两层输出，再优化力矩平滑等低层目标。

每层都可以采用上述两块 ADMM；进入下一层前必须检查接口、局部约束和最优性残差。只是在每块本地固定任务输出并不够：若安装运动尚未一致，两块的局部解拼成真实机器人后，高优先级任务仍可能被改变。

本次探查为了单独检验接口迭代，只测试第 3 层。第 1、2 层的精确任务输出由已验证的集中式控制器提供。因此这是带已知高层任务锁的离线实验，**不是完整分布式分层 WBC，也没有接入 MuJoCo 闭环**。

## 4. 实测

选取 normal 的 1 s 与 8 s、hold 的 2.6 s、double 的 6 s，共 4 个真实闭环状态；每个状态测试 rho=0.1、1、10。两块分别使用 OSQP，并在迭代间复用矩阵分解，只更新线性项。原始力矩、任务、约束和检查阈值保持一致。初值来自上一控制周期的命令力矩，对偶初值为零，没有使用集中式最优决策热启动。

直接将严格任务等式交给局部迭代求解器时出现数值困难。随后将局部任务锁精确消元，再运行 ADMM。两阶段记录分别在 admm_probe_before_lock_elimination.json 和 admm_probe.json。
'''
text+=f"\n- 最终 {len(ok)}/{len(rows)} 个样本/参数组合通过残差门槛。\n- 迭代次数 {summary['iterations'][0]}–{summary['iterations'][1]}；包括局部构造、设置/分解、迭代与校验，耗时 {summary['total_ms'][0]:.2f}–{summary['total_ms'][1]:.2f} ms。\n- 最终相对集中式力矩的最大绝对差 {summary['worst_torque_difference']:.3g} Nm；完整机器人上的任务锁残差最大 {summary['worst_hierarchy_lock']:.3g}。\n"
text+=r'''
安装加速度和六维内力残差按各分量原单位检查至 1e-7；归一化最优性残差至 2e-7；任务锁至 2e-7；硬约束至 2e-5。5/10/20/50 次迭代的残差均有保存，不能只报告最终收敛曲线。

例如 normal 1 s、rho=0.1：第 5 次迭代安装加速度最大分量残差约 1.48e-3，真实机器人任务锁残差约 1.35e-2；第 10 次任务锁残差约 4.43e-3，仍远超当前 2e-7 门槛。局部任务等式已经精确满足，但接口未收敛导致拼接后的任务输出不一致。

耗时未计生成高层任务锁的集中式控制器，因此不能当作完整层级 ADMM 耗时，也不能据此宣布 ADMM 的普遍性能。当前整个统一 WBC 同输入重放约 1.21 ms，双 WBC 精确协调约 1.41 ms；本次 ADMM 仅最低层就已更慢。这支持“暂不直接替换”的工程判断，不能证明进一步预条件、完整对偶热启动或更优局部求解器也无效。

## 5. 为什么与分布式 MPC 论文的收益不同

该论文的目标是预测时域上的非线性 OCP，利用子系统并行执行减少每个 OCP 的规模；它还明确说明该非凸算法没有收敛保证。当前是单个时刻、已将接触和接口消元到 18 个控制变量的小 QP，而且要求单线程。其计算优势不能直接转移到这个 WBC。[Amatucci 等，v5，§III-B–IV](https://arxiv.org/html/2403.11742v5)

单线程 ADMM 的代价大致为构造时间加各层 K 次“本体更新＋机械臂更新＋协调”；当前精确接口协调只需解一个 6 维线性系统。减少每个子问题的维数，可能不足以补偿反复迭代。

## 6. 可继续研究的具体方案

- 保留这次已跑通的精确接口协调器，作为真值对照和当前周期的恢复求解器。恢复时重新求当前状态问题，不能直接重复旧力矩并假定其仍满足新约束。
- 对运动/内力分别做单位一致的缩放，研究分块正定惩罚矩阵，并保存、运输每层对偶变量；rho 改变时正确重缩放 u。
- 分析接口误差如何影响完整机器人的任务输出、接触约束和力矩，以物理残差预算决定何时可接受。不能仅凭固定迭代次数或目标值变化停止。
- 若增加恢复投影，必须同时检查局部硬约束与高优先级任务锁；只投影安装加速度可能破坏力矩/摩擦约束。恢复耗时必须计入总耗时。
- 首先做完整三层的同输入离线对照，再讨论有限迭代闭环；目前只有最低层离线可行性证据。若目标仍是单线程加速，优先保留小接口精确求解，并把分解研究放在更大规模的上层 GMPC，也是一条值得比较的路线。

这些是后续待检验设计，不是已经实现的控制效果或新颖性结论。

## 来源核验范围

核验了 Boyd 作者网站上的原文 §3.1–3.3：标准两块更新、凸性/鞍点条件、原始与对偶残差。核验了 Amatucci 等 arXiv v5 的 §III-B–IV：并行子系统 OCP、非凸收敛保证的限制与计时场景。没有将其结论外推为本工程闭环证明，没有运行跨模型审稿或系统性文献查全。

代码：relu_hgmpc/probe_wbc_admm.py。数据：results/dual_wbc/admm_probe.json。运行方式：激活 relu_hgmpc/env.sh 后，执行 python relu_hgmpc/probe_wbc_admm.py。
'''
(out/'ADMM_RESEARCH.md').write_text(text);(root/'relu_hgmpc/ADMM_WBC_RESEARCH.md').write_text(text)
row=next(r for r in rows if r['case']=='normal' and r['time']==1. and r['rho']==.1)
h=row['history'];it=[x['iteration'] for x in h]
fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
for key in ('acc_consensus','force_consensus'):
    axes[0].semilogy(it,[x[key] for x in h],'-o',ms=3,label=key)
axes[0].axhline(1e-7,color='k',ls='--',label='per-component threshold')
axes[0].set_title('Interface residuals (native component units)');axes[0].legend(fontsize=8)
axes[1].semilogy(it,[x['hierarchy_lock'] for x in h],'-o',ms=3,label='physical task-output lock error')
axes[1].axhline(2e-7,color='k',ls='--',label='same hierarchy threshold')
axes[1].set_title('Local locks alone do not lock the full robot');axes[1].legend(fontsize=8)
for ax in axes:ax.set_xlabel('ADMM iterations');ax.grid(alpha=.2)
fig.suptitle('Offline final-priority probe only: normal at 1 s, rho=0.1')
fig.savefig(out/'admm_residuals.png',dpi=170);plt.close(fig)
print(json.dumps(summary,indent=2))
