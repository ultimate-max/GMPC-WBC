"""Scientific figures and factual result report; no smoothing of control traces."""
import json,pathlib,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=pathlib.Path(__file__).resolve().parents[1];out=root/'results'/'relu_hgmpc'
results={};traces={}
for it in (15,100):
    for mode in ('baseline','hgmpc'):
        name=f'{mode}_i{it}_final_12s'
        results[(mode,it)]=json.loads((out/f'{name}.json').read_text())
        traces[(mode,it)]=np.genfromtxt(out/f'{name}.csv',delimiter=',',names=True)

fig,ax=plt.subplots(2,2,figsize=(12,7.5),layout='constrained')
colors={'baseline':'#2464a5','hgmpc':'#d26628'}
for col,it in enumerate((15,100)):
    for mode in ('baseline','hgmpc'):
        x=traces[(mode,it)]
        for row,key in enumerate(('position_error_mm','rotation_error_deg')):
            ax[row,col].plot(x['time'],x[key],color=colors[mode],label='ReLU-QP MPC' if mode=='baseline' else 'Prioritized GMPC + ReLU-QP',lw=1.4)
    ax[0,col].set_title(f'{it} iterations per QP')
    ax[0,col].set_ylabel('End-effector position error (mm)');ax[1,col].set_ylabel('End-effector rotation error (deg)')
    for a in ax[:,col]:a.set_xlabel('Simulation time (s)');a.grid(alpha=.2);a.legend(fontsize=8)
fig.suptitle('B2+Z1 standing manipulation | 0.3 s horizon | 20 ms control period\nSame 6D target; simulated time, computation delay not injected',fontsize=12)
fig.savefig(out/'tracking.png',dpi=180);fig.savefig(out/'tracking.pdf');plt.close(fig)

fig,ax=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
for i,it in enumerate((15,100)):
    labels=['ReLU-QP MPC','Prioritized GMPC']
    setup=[results[(m,it)]['setup_ms']['mean'] for m in ('baseline','hgmpc')]
    qp=[results[(m,it)]['qp_ms']['mean'] for m in ('baseline','hgmpc')]
    total=[results[(m,it)]['feedback_ms']['mean'] for m in ('baseline','hgmpc')]
    rest=np.array(total)-np.array(setup)-np.array(qp)
    ax[i].bar(labels,setup,label='Build / transfer solver matrices',color='#c1c7cf')
    ax[i].bar(labels,qp,bottom=setup,label='QP iterations / sync / residuals',color='#2464a5')
    ax[i].bar(labels,rest,bottom=np.array(setup)+qp,label='Model / assembly / other',color='#d26628')
    ax[i].axhline(20,color='black',ls='--',lw=1,label='20 ms control period')
    ax[i].set_ylabel('Mean feedback computation time (ms)');ax[i].set_title(f'{it} iterations per QP; GMPC solves two QPs')
    for j,value in enumerate(total):ax[i].text(j,value+1,f'{value:.2f} ms',ha='center')
    ax[i].set_ylim(0,max(total)*1.42);ax[i].legend(fontsize=7,loc='upper left');ax[i].grid(axis='y',alpha=.15)
fig.suptitle('RTX 4060 Laptop GPU / FP64 | CPU construction costs included',fontsize=12)
fig.savefig(out/'timing.png',dpi=180);fig.savefig(out/'timing.pdf');plt.close(fig)

# Verify deterministic physics against the preceding independent run. Runtime
# columns are deliberately excluded: elapsed wall-clock time is not deterministic.
repro={}
for key,x in traces.items():
    mode,it=key;prev=np.genfromtxt(out/f'{mode}_i{it}_v1_12s.csv',delimiter=',',names=True)
    errors={k:float(np.max(np.abs(x[k]-prev[k]))) for k in ('position_error_mm','rotation_error_deg','base_z','tilt_rad','foot_drift_mm')}
    repro[f'{mode}_{it}']=errors
    if any(v>1e-9 for v in errors.values()):raise AssertionError(('Non-reproducible physical trace',key,errors))
(out/'reproducibility.json').write_text(json.dumps(repro,indent=2))

lines=['# B2+Z1 ReLU-QP MPC 与优先级 GMPC：实测结果','',
       '日期：2026-11-09。验证状态：VERIFIED（同一确定性轨迹重复运行核对）。',
       '范围：12 秒 MuJoCo 四足站立、6D 末端跟踪；每个运行 600 个控制周期。',
       '硬件：RTX 4060 Laptop；FP64；预测域 0.3 秒（10 节点），控制周期 20 ms。',
       '', '两种控制器均调用 ReLU-QP。第一种是原论文固定站立点线性化 MPC 在 B2+Z1 上的移植原型；第二种是带接触一致模型与主任务保持等式的两层 GMPC。不是原论文硬件和全部实现细节的逐项复现。','']
for it in (100,15):
    lines += [f'## 每层 {it} 次迭代','']
    for mode,label in (('baseline','固定全身 MPC'),('hgmpc','优先级 GMPC')):
        r=results[(mode,it)];x=traces[(mode,it)]
        lines += [f'**{label}**：位置 RMSE {r["position_rmse_mm"]:.3f} mm；姿态 RMSE {r["rotation_rmse_deg"]:.3f}°。',
                  f'反馈计算平均 {r["feedback_ms"]["mean"]:.3f} ms，P95 {r["feedback_ms"]["p95"]:.3f} ms；其中矩阵构造/上传 {r["setup_ms"]["mean"]:.3f} ms，QP 求解 {r["qp_ms"]["mean"]:.3f} ms。',
                  f'对偶残差最大值：主层 {r["primary_dual"]["max"]:.3g}，次层 {r["secondary_dual"]["max"]:.3g}；物理约束最大违反 {r["primal"]["max"]:.3g}；力矩截断 {r["torque_clips"]} 次。',
                  f'达到/超过 20 ms 计算时间的周期占比 {100*np.mean(x["feedback_ms"]>=20):.1f}%；最少接触足数 {int(np.min(x["contacts"]))}；最大足端漂移 {r["foot_drift_mm"]["max"]:.3f} mm。','']
lines += ['## 如何解释','',
          '两层 GMPC 的主任务保持不靠大权重，而是对次层输入使用第一层任务有效动力学映射的零空间。保留的是本次局部模型和当前有限迭代主层解。',
          f'100 次运行的预测任务最大改变量为 {results[("hgmpc",100)]["hierarchy"]["max"]:.3g}。',
          '本版本 GMPC 每周期都重新构造两个 ReLU-QP 稠密层，固定 MPC 可复用。这个额外成本已进入总时间；不能从 QP 迭代快推导出整体控制器更快。',
          '较少迭代带来的速度提升必须与残差和跟踪误差一起评价。两个控制器的状态/代价/参考生成和模型更新也不同，因此目前不是只改变“欧氏空间/李群空间”的消融。',
          '仿真未注入计算延迟，超过 20 ms 的配置还不能在 50 Hz 下实时执行。所有控制周期的仿真成功不等于实时系统稳定性证明。',
          '', '## 检查与复现','',
          '接触动力学 KKT、体雅可比有限差分、几何误差线性化、非零次层修正后的完整主任务代价保持均通过。活跃约束随机 QP 和实际半正定 GMPC QP 均与 OSQP 核对。',
          '四组确定性物理轨迹独立运行两次，误差与漂移记录在 1e-9 以内一致。运行时间受系统调度影响，不宣称统计显著性。',
          '代码及逐文件 SHA-256 记录在每个摘要 JSON；原始逐周期 CSV、结果 JSON、verification.json、solver_verification.json、reproducibility.json 同目录。',
          '', '![末端跟踪](tracking.png)','', '![计时分解](timing.png)','',
          '完整方法与差异说明见工程 relu_hgmpc/README.md 和 FORMULATION.md。']
(out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines[:32]))
