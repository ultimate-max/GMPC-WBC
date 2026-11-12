"""Matched closed-loop summary; no substitution of older prototype metrics."""
import pathlib,json,sys,shutil,csv,platform,subprocess
import numpy as np
from scipy.signal import welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/hierarchical_gmpc'
groups={};records={}
for scale in (1,2):
    for mode,prefix in [('weighted','weighted'),('hierarchical','hierarchy_null')]:
        names=[f'{prefix}_scale{scale}_r{r}' for r in (1,2)]
        data=[json.loads((out/f'{n}.json').read_text()) for n in names]
        for d in data:
            assert d['failed'] is None and d['simulated_cycles']==600
            assert d['min_contacts']==4 and d['torque_clips']==0
            assert max(d[k]['max'] for k in ('qp_primal_relative','qp_dual_relative','qp_normal_relative'))<=2e-7
        rows=[]
        for name in names:
            with (out/f'{name}.csv').open() as f:
                rows.extend([{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)])
        result=dict(names=names,position_rmse_mm=float(np.sqrt(np.mean([r['position_error_mm']**2 for r in rows]))),
            rotation_rmse_deg=float(np.sqrt(np.mean([r['rotation_error_deg']**2 for r in rows]))),
            feedback_ms=float(np.mean([d['amortized_feedback_ms'] for d in data])),
            p95_ms=float(np.quantile([r['feedback_ms'] for r in rows],.95)),
            p99_ms=float(np.quantile([r['feedback_ms'] for r in rows],.99)),
            max_ms=max(r['feedback_ms'] for r in rows),
            min_contacts=min(d['min_contacts'] for d in data),torque_clips=sum(d['torque_clips'] for d in data),
            foot_center_max_mm=max(d['foot_center_mm']['max'] for d in data),
            residual_max=max(d[k]['max'] for d in data for k in ('qp_primal_relative','qp_dual_relative','qp_normal_relative')),
            run_means_ms=[d['amortized_feedback_ms'] for d in data])
        if mode=='hierarchical':
            result.update(task_lock_relative_max=max(d['task_lock_relative']['max'] for d in data),
                task_lock_weighted_max=max(d['task_lock_weighted_max']['max'] for d in data),
                primary_objective_change_max=max(max(abs(r['primary_objective_change']) for r in rows),0.),
                secondary_cost_worst_increase=max(r['secondary_objective_after']-r['secondary_objective_before'] for r in rows),
                primary_iterations_mean=float(np.mean([r['primary_iterations'] for r in rows])),
                secondary_iterations_mean=float(np.mean([r['secondary_iterations'] for r in rows])),
                task_lock_rank_ratio_min=min(r['task_lock_rank_ratio'] for r in rows))
            assert result['task_lock_relative_max']<=2e-7
        actual=[]
        for name in names:
            z=np.load(out/f'{name}_trace.npz')['trace'];assert z.shape==(12000,129),z.shape
            v=z[:,111:129];tau=z[:,57:75];m={}
            for group,sl in [('legs',slice(0,12)),('arm',slice(12,18))]:
                def band(x):
                    f,P=welch(x,fs=1000,nperseg=2048,noverlap=1024,axis=0)
                    ix=(f>=10)&(f<=100)
                    return float(np.sqrt(np.trapezoid(P[ix],f[ix],axis=0).mean()))
                m[group]=dict(velocity_increment_rms=float(np.sqrt(np.mean(np.diff(v[:,sl],axis=0)**2))),
                    torque_increment_rms=float(np.sqrt(np.mean(np.diff(tau[:,sl],axis=0)**2))),
                    velocity_band_10_100_hz=band(v[500:,sl]),torque_band_10_100_hz=band(tau[500:,sl]))
            actual.append(m)
        result['actual']={g:{k:float(np.sqrt(np.mean([m[g][k]**2 for m in actual]))) for k in actual[0][g]} for g in ('legs','arm')}
        groups[f'{mode}_scale{scale}']=result;records.update({n:d for n,d in zip(names,data)})

hardware=subprocess.run(['lscpu'],capture_output=True,text=True).stdout
summary=dict(groups=groups,cycles=4800,low_level_samples=96000,verification=json.loads((out/'verification.json').read_text()),
    machine=platform.platform(),cpu=hardware,notes='Two deterministic repeats per case; serialized runs. Not independent robustness trials. Both-level original-coordinate tolerance 2e-7. No simulated compute delay.')
(out/'comparison.json').write_text(json.dumps(summary,indent=2))
lines=['# 层级 GMPC 实现与闭环核对','',
    '物理约束保持不变；第一层优化末端 SE(3) 位姿与误差变化率，第二层在保持全部预测跟踪输出的条件下优化构型和输入平滑。通过接触等式消元与任务零空间参数化，两层分别为180、120个变量。',
    '', '同一12秒任务、N=10、30ms预测步长、20ms反馈、1kHz物理记录；普通/双幅各两次重复，合计4800个反馈周期、96000个低层样本。求解使用集中式CPU OSQP，模型、约束、低层与验收精度一致；目标层级是本轮主动改变的因素。','']
for scale in (1,2):
    lines += [f'## {"普通" if scale==1 else "双幅"}参考','']
    for mode,label in [('weighted','加权GMPC'),('hierarchical','层级GMPC')]:
        d=groups[f'{mode}_scale{scale}']
        lines += [f'- {label}：位置RMSE {d["position_rmse_mm"]:.6f} mm；姿态RMSE {d["rotation_rmse_deg"]:.6f}°；总耗时均值 {d["feedback_ms"]:.3f} ms，P95 {d["p95_ms"]:.3f} ms，P99 {d["p99_ms"]:.3f} ms。']
        for g,labelg in [('legs','腿部'),('arm','臂部')]:
            m=d['actual'][g]
            lines += [f'  - {labelg}：相邻1ms实际速度增量RMS {m["velocity_increment_rms"]:.8g} rad/s，施加力矩增量RMS {m["torque_increment_rms"]:.8g} N·m；10–100Hz速度/力矩RMS分别 {m["velocity_band_10_100_hz"]:.8g}、{m["torque_band_10_100_hz"]:.8g}。']
    a=groups[f'weighted_scale{scale}'];b=groups[f'hierarchical_scale{scale}']
    lines += [f'位置RMSE变化 {100*(b["position_rmse_mm"]/a["position_rmse_mm"]-1):+.2f}%，姿态RMSE变化 {100*(b["rotation_rmse_deg"]/a["rotation_rmse_deg"]-1):+.2f}%。层级版本耗时为加权版本的 {b["feedback_ms"]/a["feedback_ms"]:.2f} 倍。','']
lines += ['## 验收与解释','']
for scale in (1,2):
    d=groups[f'hierarchical_scale{scale}']
    lines += [f'- 幅度{scale}：两层原问题相对残差最大 {d["residual_max"]:.3e}；完整120行任务保持相对误差最大 {d["task_lock_relative_max"]:.3e}；第一层残差平方和改变最大 {d["primary_objective_change_max"]:.3e}。']
lines += ['', '全部运行四足持续接触、无力矩裁剪。位姿误差与变化率在第一层内部仍按原权重折中，不能宣称每个指标都改善；次层平滑不能牺牲第一层预测输出，可能使实际输入变化更大。抖振指标必须直接看本轮1kHz数据，不能沿用早期速度层改善倍数。',
    '', '本轮没有观察到执行平滑性的整体改善：层级版本腿部力矩增量更大，臂部力矩增量仅微小降低，实际速度增量有所增加。',
    '', '总耗时包含状态读取、模型和矩阵构造、两次消元/投影、两个QP、乘子恢复、KKT与任务保持检查，含启动摊销；不计MuJoCo推进。未注入计算延迟，尚不能据此宣称50Hz实时控制或硬件稳定性。严格层级性质针对固定的局部预测模型及已验收第一层解。',
    '', '旧加权加速实验的2.16–3.09倍加速保持为独立历史基线，不属于此次层级版本的加速倍率。',
    '', '![同条件跟踪与耗时](hierarchy_comparison.png)', '', '精确数据、残差及硬件信息见 comparison.json；原始CSV与NPZ保存在WSL results/hierarchical_gmpc。']
(out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

fig,axes=plt.subplots(2,3,figsize=(12,6),layout='constrained')
for row,scale in enumerate((1,2)):
    for col,(key,title,unit) in enumerate([('position_rmse_mm','Position RMSE','mm'),('rotation_rmse_deg','Orientation RMSE','deg'),('feedback_ms','Total feedback mean','ms')]):
        ax=axes[row,col];values=[groups[f'{m}_scale{scale}'][key] for m in ('weighted','hierarchical')]
        ax.bar(['Weighted','Hierarchical'],values,color=['#9c9c9c','#2878a0'])
        for i,val in enumerate(values):ax.text(i,val,f'{val:.3g}',ha='center',va='bottom',fontsize=10)
        ax.set_ylim(0,max(values)*1.25);ax.set_ylabel(unit);ax.set_title(f'Scale {scale}: {title}');ax.grid(axis='y',alpha=.2)
fig.savefig(out/'hierarchy_comparison.png',dpi=170);plt.close(fig)
if len(sys.argv)>1:
    dst=pathlib.Path(sys.argv[1]);dst.mkdir(parents=True,exist_ok=True)
    for f in ('RESULTS.md','comparison.json','hierarchy_comparison.png','verification.json'):shutil.copy2(out/f,dst/f)
print(json.dumps(groups,indent=2))
