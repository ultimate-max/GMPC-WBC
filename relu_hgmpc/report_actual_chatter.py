import pathlib,json,sys,shutil
import numpy as np
from scipy.signal import welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/chatter_1khz'
names=['threshold','hysteresis','admm_se3'];labels=['Hard switch','Hysteresis','Continuous ADMM + SE(3)']
traces={n:np.load(out/f'{n}_trace.npz')['trace'] for n in names}
result={}
for name,z in traces.items():
    assert z.shape[1]==129 and len(z)==12000,(name,z.shape)
    v=z[:,111:129];tau=z[:,57:75];metrics={}
    for label,sl in [('legs',slice(0,12)),('arm',slice(12,18))]:
        dv=np.diff(v[:,sl],axis=0);du=np.diff(tau[:,sl],axis=0)
        def band(x):
            f,P=welch(x,fs=1000,nperseg=2048,noverlap=1024,axis=0)
            ix=(f>=10)&(f<=100)
            return float(np.sqrt(np.trapezoid(P[ix],f[ix],axis=0).mean()))
        metrics[label]=dict(velocity_increment_rms=float(np.sqrt(np.mean(dv**2))),
            velocity_increment_max=float(np.max(np.abs(dv))),torque_increment_rms=float(np.sqrt(np.mean(du**2))),
            torque_increment_max=float(np.max(np.abs(du))),velocity_band_10_100_hz=band(v[500:,sl]),
            torque_band_10_100_hz=band(tau[500:,sl]))
    result[name]=metrics
(out/'actual_metrics.json').write_text(json.dumps(result,indent=2))
colors=['#b65035','#b78b18','#2464a5']
fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for ax,(group,key,unit) in zip(axes.ravel(),[
    ('legs','velocity_increment_rms','rad/s'),('arm','velocity_increment_rms','rad/s'),
    ('legs','torque_increment_rms','Nm'),('arm','torque_increment_rms','Nm')]):
    values=[result[n][group][key] for n in names]
    ax.bar(np.arange(3),values,color=colors);ax.set_yscale('log')
    ax.set_xticks(np.arange(3),['Hard switch','Hysteresis','Continuous\nADMM + SE(3)'],fontsize=9)
    ax.set_title(group.capitalize()+' measured '+('velocity' if 'velocity' in key else 'applied torque'))
    ax.set_ylabel('RMS 1 ms increment ('+unit+')');ax.grid(axis='y',alpha=.2)
    for i,v in enumerate(values):ax.text(i,v*1.15,f'{v:.4g}',ha='center',fontsize=9)
    ax.set_ylim(min(values)/2,max(values)*3)
fig.suptitle('Actual motion and applied torque: 1 kHz MuJoCo truth, same 12 s task')
fig.savefig(out/'actual_chatter.png',dpi=170);plt.close(fig)
lines=['# 实际关节速度与执行力矩抖振核对','',
 '同一 12 s 任务、20 ms 控制周期，三种方法均记录 12000 个 1 ms 步长。速度来自 MuJoCo 真值；力矩是前馈＋关节 PD＋摩擦补偿后的实际施加指令，三组均无力矩截断。',
 '增量 RMS 是相邻 1 ms 样本的差，在各组关节及时间上取均方根，不能把它称作加速度 RMS 或整段速度 RMS。另报告去掉前 0.5 s 后的 10–100 Hz Welch 频带 RMS，用于区分缓慢任务运动与高频变化。','']
for name,label in zip(names,labels):
    lines += [f'## {label}','']
    for group in ['legs','arm']:
        m=result[name][group]
        lines += [f'- {group}：速度增量 RMS {m["velocity_increment_rms"]:.8g} rad/s；力矩增量 RMS {m["torque_increment_rms"]:.8g} Nm；速度 10–100 Hz RMS {m["velocity_band_10_100_hz"]:.8g} rad/s；力矩同频段 RMS {m["torque_band_10_100_hz"]:.8g} Nm。']
    lines+=['']
lines+=['连续 ADMM 版本降低实际速度和执行力矩变化；集中式连续 QP 的此前对照得到近乎相同的结果，因此因果改善来自连续分配与时间正则化，不能归为 ADMM 独有优势。',
 '速度层原型仍使用原 WBC；它不是完整动力学分布式 GMPC。完整预测层另见 results/split_gmpc。','',
 '![实际抖振](actual_chatter.png)']
(out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if len(sys.argv)>1:
    dst=pathlib.Path(sys.argv[1]);dst.mkdir(parents=True,exist_ok=True)
    for file in ['RESULTS.md','actual_metrics.json','actual_chatter.png']:shutil.copy2(out/file,dst/file)
print(json.dumps(result,indent=2))
