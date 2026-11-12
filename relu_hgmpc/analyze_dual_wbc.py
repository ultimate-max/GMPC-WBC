import csv,json,pathlib,hashlib,platform,sys
import numpy as np
from scipy.signal import butter,sosfiltfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/dual_wbc'
methods={'dual':'Split WBC','unified':'Unified WBC','paper':'Paper 6D adaptation','paper_shared':'Paper allocator + common WBC'}
colors={'dual':'#1775b8','unified':'#e99a25','paper':'#47915a','paper_shared':'#a469b5'}
def load(name):
    meta=json.loads((out/f'{name}.json').read_text())
    with (out/f'{name}.csv').open() as f:rows=list(csv.DictReader(f))
    data={k:np.array([float(r[k]) for r in rows]) for k in rows[0]}
    return meta,data
def metrics(name,end):
    meta,d=load(name);mask=d['time']<end
    trace=np.load(out/f'{name}_trace.npz')['trace'];sel=(trace[:,0]>=1.)&(trace[:,0]<end)
    result={'interval_seconds':[0,end],'jitter_interval_seconds':[1,end],
        'controller_cpu_seconds':float(d['cpu_ms'][mask].sum()/1000),
        'controller_wall_seconds':float(d['total_ms'][mask].sum()/1000),
        'cycle_overrun_fraction':float(d['cycle_overrun'][mask].mean())}
    for k in ('ee_position_error_mm','ee_rotation_error_deg','base_position_error_mm','base_rotation_error_deg'):
        x=d[k][mask];result[k]={'rms':float(np.sqrt(np.mean(x*x))),'max':float(x.max())}
    for k in ('total_ms','wbc_reference_ms','controller_ms','wbc_construction_ms','wbc_solve_ms','wbc_validation_ms'):
        if k in d:
            x=d[k][mask];result[k]={'mean':float(x.mean()),'p95':float(np.quantile(x,.95))}
    for name2,sl in [('leg_velocity',slice(111,123)),('arm_velocity',slice(123,129)),('leg_torque',slice(57,69)),('arm_torque',slice(69,75))]:
        x=trace[:,sl];r={'step_rms':float(np.sqrt(np.mean(np.diff(x[sel],axis=0)**2)))}
        for low in (5,10):
            filtered=sosfiltfilt(butter(4,[low,100],btype='bandpass',fs=1000,output='sos'),x,axis=0)
            r[f'hf_{low}_100_rms']=float(np.sqrt(np.mean(filtered[sel]**2)))
        result[name2]=r
    plans=[p for p in meta['plans'] if p['time']<end]
    for k in ('wall_ms','construction_ms','task_construction_ms','solve_ms'):
        x=np.array([p[k] for p in plans if k in p])
        if len(x):result['upper_'+k]={'mean':float(x.mean()),'p95':float(np.quantile(x,.95)),'max':float(x.max())}
    result['reference_hash']=meta['target_hash']
    return result

normal_names={'dual':'dual_normal_damped','unified':'unified_normal_damped','paper':'paper_normal','paper_shared':'paper_shared_normal_damped'}
data={};report=dict(normal_common_8s={},cases={},ablation={},repeats={})
for method,name in normal_names.items():
    report['normal_common_8s'][method]=metrics(name,8.)
    data[method]=load(name)[1]
hashes={m['reference_hash'] for m in report['normal_common_8s'].values()};assert len(hashes)==1
for case in ('normal','double','hold','push'):
    report['cases'][case]={}
    for method in ('dual','unified','paper'):
        name=f'{method}_{case}'+('_damped' if method!='paper' else '')
        meta,_=load(name)
        keys=['completed_seconds','failure','ee_position_error_mm','ee_rotation_error_deg','base_position_error_mm',
              'base_rotation_error_deg','upper_total_ms','wbc_reference_ms','total_controller_cpu_ms',
              'constraint_violation','contact_residual','dynamics_residual','task_lock_max','qp_kkt',
              'minimum_contacts','contact_loss_fraction','torque_clips','minimum_sigma','singlethread_timing']
        report['cases'][case][method]={k:meta[k] for k in keys}
for mode in ('dual','unified'):
    report['repeats'][mode]=[metrics(f'{mode}_normal_damped'+suffix,12.) for suffix in ('','_repeat')]
for key,name in [('weight_1','dual_normal_damped'),('weight_0','dual_normal_damped_no_soft_smoothing')]:
    report['ablation'][key]=metrics(name,12.)
report['ablation']['note']='Only the soft torque-increment cost is removed; hard torque-rate bounds and joint-limit velocity damping remain. This does not isolate all smoothing mechanisms.'
report['definitions']={'timing':'Online CPU and wall time include state/model evaluation, reference/interpolation, matrix construction, factorization, hierarchical solving, residual verification; exclude one-time startup, disk output and MuJoCo stepping. No worker thread/GPU.',
    'rates':'50 Hz GMPC, 200 Hz WBC, 1000 Hz actuator servo; controlled simulation waits during solves, not a certified real-time run.',
    'jitter':'RMS across time and joints of measured velocity or actually applied torque, fourth-order 5-100 Hz Butterworth, zero-phase filtering. Includes motion transients; not unique evidence of switching chatter.',
    'paper':'B2+Z1 6D adaptation of prioritized motion distribution + original WBC. Not authors code or an exact experimental reproduction. Prescribed base pose is not its independent objective.',
    'model':'10 nodes, 40 ms spacing; measured model frozen per cycle with affine contact/velocity corrections. Not full nonlinear multiple-shooting MPC.',
    'hierarchy':'Hard feasibility > EE pose/rate > base pose/rate > manipulability/posture/smoothing. Upper and lower pose tasks use SE(3).',
    'damping':'Matched upper/lower linear joint-limit velocity damper eta=5/s; hard torque slew legs1000 and arm100 Nm/s. Kept identical in split and unified comparisons.'}
report['environment']=dict(python=sys.version,platform=platform.platform())
try:
    from threadpoolctl import threadpool_info
    report['environment']['threadpools']=threadpool_info()
except ImportError:pass
for key in ('equivalence_test','matched_replay','failure_diagnosis'):
    p=out/f'{key}.json'
    if p.exists():report[key]=json.loads(p.read_text())
(out/'comparison.json').write_text(json.dumps(report,indent=2))

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(3,1,figsize=(10,8),sharex=True,constrained_layout=True)
for method,d in data.items():
    for ax,key in zip(axs,('ee_position_error_mm','ee_rotation_error_deg','base_position_error_mm')):
        ax.plot(d['time'],d[key],label=methods[method],color=colors[method],lw=1.2,alpha=.9)
        ax.axvspan(8,12,color='0.8',alpha=.08)
for ax,label in zip(axs,('EE position error (mm)','EE orientation error (deg)','Base position error (mm)')):
    ax.set_ylabel(label);ax.grid(alpha=.2);ax.set_xlim(0,12)
axs[0].legend(ncol=2,fontsize=9);axs[-1].set_xlabel('Simulation time (s)')
fig.suptitle('Same external task; failed runs stop at failure (no extrapolation)')
fig.savefig(out/'normal_tracking.png',dpi=170);plt.close(fig)

fig,axs=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
keys=['ee_position_error_mm','ee_rotation_error_deg','controller_cpu_seconds','leg_velocity','arm_velocity','arm_torque']
titles=['EE position RMS (mm)','EE orientation RMS (deg)','Controller CPU for 8 s (s)','Actual leg velocity 5-100 Hz (rad/s)','Actual arm velocity 5-100 Hz (rad/s)','Applied arm torque 5-100 Hz (Nm)']
labels=['Split','Unified','Paper','Paper+WBC']
for ax,key,title in zip(axs.flat,keys,titles):
    values=[]
    for mode in methods:
        value=report['normal_common_8s'][mode][key]
        values.append(value if isinstance(value,float) else value['rms' if 'error' in key else 'hf_5_100_rms'])
    ax.bar(labels,values,color=[colors[m] for m in methods]);ax.set_title(title,fontsize=10);ax.grid(axis='y',alpha=.2)
    for i,v in enumerate(values):ax.text(i,v,f'{v:.3g}',ha='center',va='bottom',fontsize=9)
    ax.set_ylim(0,max(values)*1.2)
fig.suptitle('Common prefix: tracking/cost 0-8 s; measured high-frequency content 1-8 s')
fig.savefig(out/'common_comparison.png',dpi=170);plt.close(fig)

fig,axs=plt.subplots(2,1,figsize=(10,6),sharex=True,constrained_layout=True)
for mode in ('dual','unified'):
    name=f'{mode}_hold_damped';_,d=load(name)
    axs[0].plot(d['time'],d['ee_position_error_mm'],label=methods[mode],color=colors[mode])
    tr=np.load(out/f'{name}_trace.npz');axs[1].plot(np.arange(len(tr['q']))*.005,tr['q'][:,23],color=colors[mode])
axs[0].set_ylabel('Fixed EE error (mm)');axs[0].legend();axs[1].set_ylabel('Arm joint 5 position (rad)');axs[1].set_xlabel('Simulation time (s)')
for ax in axs:ax.axvline(2.71,color='r',ls='--',label='Failure without limit damping');ax.grid(alpha=.2)
fig.suptitle('Joint-limit damping prevents the original 2.71 s constraint conflict')
fig.savefig(out/'hold_limit_damping.png',dpi=170);plt.close(fig)
print(json.dumps({k:report[k] for k in ('normal_common_8s','ablation')},indent=2))
