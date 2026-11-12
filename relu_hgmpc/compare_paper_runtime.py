"""Common-window, same-grid descriptive comparison; preserve failure outcomes."""
import csv,json,pathlib,hashlib
import numpy as np
import pinocchio as pin
from scipy.signal import butter,sosfiltfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import native
from runtime_support import DenseTarget
from geometric_wbc import pose

root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/paper_comparison';native_dir=root/'results/buffered_wbc'
paper_meta=json.loads((out/'paper400_normal.json').read_text());our_meta=json.loads((native_dir/'native_mrt400_normal.json').read_text())
assert paper_meta['target_hash']==our_meta['target_hash'],'Different external target'
plant=native.Plant(str(root));target=DenseTarget(plant,root,'normal')
def loadrows(path):
    with path.open() as f:rows=list(csv.DictReader(f))
    return {k:np.array([float(r[k]) for r in rows]) for k in rows[0]}
def loadarrays(path):
    with np.load(path) as source:return {k:source[k] for k in source.files}
entries={
    'paper':(paper_meta,loadrows(out/'paper400_normal.csv'),loadarrays(out/'paper400_normal_trace.npz')),
    'ours':(our_meta,loadrows(native_dir/'native_mrt400_normal.csv'),loadarrays(native_dir/'native_mrt400_normal_trace.npz'))}
end=8.;metrics={};curves={}
for method,(meta,rows,data) in entries.items():
    assert meta['completed_seconds']>=end
    t=rows['time'];mask=t<end;trace=data['trace'];assert np.max(np.abs(np.diff(t)-.0025))<1e-10
    pe=[];ae=[];be=[]
    for i,tt in enumerate(t):
        E=data['ee_pose'][i] if method=='paper' else data['ee_pose_columns'][i].reshape((4,4),order='F')
        goal=target.at(tt);pe.append(1000*np.linalg.norm(E[:3,3]-goal['ee_pose'][:3,3]));ae.append(180/np.pi*np.linalg.norm(pin.log3(goal['ee_pose'][:3,:3].T@E[:3,:3])));be.append(1000*np.linalg.norm(data['q'][i,:3]-goal['base_pose'][:3,3]))
    curves[method]=dict(t=t,position=np.array(pe),orientation=np.array(ae),base=np.array(be))
    r=dict(ee_position_rms_mm=float(np.sqrt(np.mean(np.square(np.array(pe)[mask])))),ee_orientation_rms_deg=float(np.sqrt(np.mean(np.square(np.array(ae)[mask])))),
        base_target_deviation_rms_mm=float(np.sqrt(np.mean(np.square(np.array(be)[mask])))),
        completed_seconds=meta['completed_seconds'],failure=meta['failure'],minimum_contacts=int(rows['contacts'][mask].min()),
        control_mean_ms=float(rows['control_ms'][mask].mean()),control_p99_ms=float(np.quantile(rows['control_ms'][mask],.99)),
        execution_mean_ms=float(rows['full_loop_ms'][mask].mean()),execution_p99_ms=float(np.quantile(rows['full_loop_ms'][mask],.99)),
        common_deadline_miss_fraction=float(np.mean((rows['start_lateness_ms'][mask]+rows['full_loop_ms'][mask])>2.5)))
    trace_mask=(trace[:,0]>=1)&(trace[:,0]<end)
    for name,sl in [('leg_velocity',slice(111,123)),('arm_velocity',slice(123,129)),('leg_torque',slice(57,69)),('arm_torque',slice(69,75))]:
        # Filter identical 0-8 s physical-time prefixes, to avoid borrowing
        # different post-window lengths from failed and completed runs.
        sig=trace[trace[:,0]<end,sl]
        filtered=sosfiltfilt(butter(4,[5,100],btype='bandpass',fs=2000,output='sos'),sig,axis=0)
        sel=trace[trace[:,0]<end,0]>=1
        r[name+'_hf_rms']=float(np.sqrt(np.mean(filtered[sel]**2)))
    r['controller_components_wall_seconds']=float(rows['control_ms'][mask].sum()/1000)
    if method=='ours':
        ev=[e for e in meta['mpc_events'] if e['time']<end]
        r['mpc_total_mean_ms']=float(np.mean([e['total_ms'] for e in ev]));r['mpc_total_p99_ms']=float(np.quantile([e['total_ms'] for e in ev],.99))
        r['mpc_components_wall_seconds']=sum(e['total_ms'] for e in ev)/1000
        r['controller_components_wall_seconds']+=r['mpc_components_wall_seconds']
        r['policy_install_hz_common']=float(rows['installed'][mask].sum()/end)
    else:
        r['allocation_mean_ms']=float(rows['allocation_ms'][mask][rows['allocated'][mask]>0].mean())
    metrics[method]=r
rel={k:100*(metrics['ours'][k]/metrics['paper'][k]-1) for k in ['ee_position_rms_mm','ee_orientation_rms_deg','leg_velocity_hf_rms','arm_velocity_hf_rms','leg_torque_hf_rms','arm_torque_hf_rms','controller_components_wall_seconds']}
report=dict(status='ANALYZED',comparison='B2+Z1 6D engineering adaptation of Prioritized Multi-task Motion Coordination of Physically Constrained Quadruped Manipulators versus current hierarchical GMPC + native coordinated dual WBC',
    control_hz=400,upper_hz=50,servo_hz=2000,tracking_interval_seconds=[0,8],hf_interval_seconds=[1,8],hf_filter='Fourth-order Butterworth 5-100 Hz, zero phase, RMS over physical time and all joints; same 0-8 s prefix filtered for both',
    target_sha256=target.hash,metrics=metrics,ours_relative_change_percent=rel,
    limits=['One run per method; no statistical significance or global stability claims.',
            'Paper adaptation is not author code, original paper experiment or same task/constraint optimization.',
            'Base reference is an independent task only for GMPC; baseline base deviation is not a fair tracking penalty.',
            'Python paper controller versus native C++ WBC; timing cannot isolate algorithmic speedup.',
            'Controller-components wall sum includes MPC and WBC work but overlaps across threads; it is neither process CPU time nor elapsed closed-loop duration. MuJoCo excluded in this sum.',
            'Physical simulator remains control-owned; timing does not model an independently running actuator.',
            'High-frequency energy includes legitimate transients and cannot alone identify S-switching causality.'],
    fallacy_scan=dict(coverage='11/11 checked',notes={'Simpson':'Leg and arm metrics kept separate; no combined win claim.','Ecological':'No claims about each individual joint from pooled RMS.','Berkson':'Common window declared; full-run failures retained.','Collider':'No covariate adjustment.','Base rate':'No diagnostic-probability claim.','Regression to mean':'Single runs, no inference from repeated extreme selection.','Survivorship':'Baseline failure is explicitly retained alongside common-prefix metrics.','Look elsewhere':'All selected position/orientation, velocity/torque and cost metrics reported.','Forking paths':'Window and filter specified; single exploratory comparison.','Causation':'Frequency aligned but task/constraint/language differences remain; no Lie-group-only or acceleration claim.','Reverse causality':'No directional statistical inference.'}),
    sources={'paper_json':str(out/'paper400_normal.json'),'ours_json':str(native_dir/'native_mrt400_normal.json')})
(out/'comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors=['#777777','#1b678c'];labels=['Paper 6D adaptation','Hierarchical GMPC + dual WBC']
fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
for ax,key,title,unit in zip(axes.flat,
    ['ee_position_rms_mm','ee_orientation_rms_deg','controller_components_wall_seconds','leg_velocity_hf_rms','arm_velocity_hf_rms','leg_torque_hf_rms'],
    ['EE position RMS','EE orientation RMS','Controller component-time sum','Actual leg velocity, 5-100 Hz','Actual arm velocity, 5-100 Hz','Applied leg torque, 5-100 Hz'],
    ['mm','deg','s / 8 s simulation','rad/s','rad/s','Nm']):
    vals=[metrics[m][key] for m in ('paper','ours')];ax.bar(['Paper adaptation','GMPC + WBC'],vals,color=colors);ax.set_title(title,fontsize=11);ax.set_ylabel(unit);ax.set_xlabel('Controller');ax.set_ylim(0,max(vals)*1.23);ax.grid(axis='y',alpha=.18)
    for i,v in enumerate(vals):ax.text(i,v,f'{v:.4g}',ha='center',va='bottom')
fig.suptitle('B2+Z1: 400 Hz control / 50 Hz upper layer / 2 kHz servo\nTracking and cost: 0-8 s; high-frequency signals: 1-8 s. One run each.',fontsize=12)
fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
fig,axs=plt.subplots(2,1,figsize=(11,6),sharex=True,constrained_layout=True)
for method,color,label in zip(('paper','ours'),colors,labels):
    c=curves[method];axs[0].plot(c['t'],c['position'],label=label,color=color,lw=1.1);axs[1].plot(c['t'],c['orientation'],label=label,color=color,lw=1.1)
for ax in axs:ax.axvline(8,color='black',ls=':',lw=.8);ax.grid(alpha=.2);ax.set_xlim(0,12)
axs[0].set_ylabel('EE position error (mm)');axs[1].set_ylabel('EE orientation error (deg)');axs[1].set_xlabel('Simulation time (s)');axs[0].legend()
fig.suptitle('Identical external EE target; failed runs stop at failure (no extrapolation)')
fig.savefig(out/'tracking.png',dpi=160);plt.close(fig)
