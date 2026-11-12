from pathlib import Path
import json,sys,subprocess,platform
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'results/paper_repro'
groups={}
for group in ['central','distributed','distributed_serial']:
    files=sorted(D.glob(group+'_trial[123].csv'))
    if files: groups[group]=np.concatenate([np.atleast_1d(np.genfromtxt(f,delimiter=',',names=True)) for f in files])
summary={}
for name,a in groups.items():
    result={'samples':len(a),'rmse_mm':float(1000*np.sqrt(np.mean(a['error_m']**2))),'max_error_mm':float(1000*a['error_m'].max()),'four_contacts_fraction':float(np.mean(a['contacts']==4)),'max_tilt_deg':float(np.rad2deg(a['tilt_rad'].max())),'max_foot_drift_mm':float(1000*a['foot_drift_m'].max()),'failed_solves':int(a['solver_failures'].sum()),'torque_clips':int(a['torque_clips'].sum()),'deadline_miss_fraction':float(np.mean(a['controller_ms']>20))}
    exits=[int(f.read_text()) for f in sorted(D.glob(name+'_trial[123].exit'))]
    result['exit_codes']=exits
    failures=[json.loads(f.read_text()) for f in sorted(D.glob(name+'_trial[123]_failure.json')) if int(f.with_name(f.name.replace('_failure.json','.exit')).read_text())!=0]
    result['abort_events']=failures
    result['failed_solves']+=sum(f.get('solver_failures',0) for f in failures)
    result['completed']=bool(exits and all(x==0 for x in exits) and len(a)==300*len(exits))
    result['tracking_valid']=bool(result['completed'] and result['max_error_mm']<5 and result['max_foot_drift_mm']<5 and result['max_tilt_deg']<3 and result['four_contacts_fraction']>=.99 and result['torque_clips']==0 and result['failed_solves']==0)
    for key in ['update_ms','max_local_solve_ms','parallel_wall_ms','controller_ms','interface_l2','interface_max','full_defect_l2','full_defect_max','critical_lin_ms','critical_qp_ms']:
        result[key]={'mean':float(a[key].mean()),'p50':float(np.median(a[key])),'p95':float(np.percentile(a[key],95)),'max':float(a[key].max())}
    summary[name]=result
if 'central' in summary and 'distributed' in summary:
    if summary['central']['tracking_valid'] and summary['distributed']['tracking_valid']:
        summary['speedup']={k:summary['central'][k]['mean']/summary['distributed'][k]['mean'] for k in ['max_local_solve_ms','controller_ms']}
    else: summary['speedup']={'valid':False,'reason':'A controller failed closed-loop validation. Partial-run timing is diagnostic only, not a speedup comparison.'}
(D/'summary.json').write_text(json.dumps(summary,indent=2))
env={'platform':platform.platform(),'python':platform.python_version()}
for repo in ['acados','DWMPC','unitree_ros']:
    env[repo]=subprocess.check_output(['git','-C',str(ROOT/'third_party'/repo),'rev-parse','HEAD'],text=True).strip()
for mod in ['pinocchio','casadi','mujoco','numpy']:
    env[mod]=__import__(mod).__version__
env['cpu']=subprocess.check_output(['lscpu'],text=True)
(D/'environment.json').write_text(json.dumps(env,indent=2))
colors=['#2166ac','#d95f02','#666666'];fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for c,(name,a) in zip(colors,groups.items()):
    # Display the first full trial, with raw unsmoothed values.
    first=sorted(D.glob(name+'_trial[123].csv'))[0]
    run=np.atleast_1d(np.genfromtxt(first,delimiter=',',names=True))
    label=name.replace('_',' ')
    axes[0,0].plot(run['time'],run['controller_ms'],color=c,label=label,alpha=.85)
    axes[0,1].plot(run['time'],1000*run['error_m'],color=c,label=label)
    axes[1,0].semilogy(run['time'],np.maximum(run['full_defect_l2'],1e-15),color=c,label=label)
    if name!='central': axes[1,1].semilogy(run['time'],np.maximum(run['interface_l2'],1e-15),color=c,label=label)
axes[0,0].axhline(20,color='black',ls=':',label='50 Hz budget')
for ax,ylabel in zip(axes.flat,['Controller wall time (ms)','Tool position error (mm)','Full nonlinear dynamics defect, L2','Interface velocity residual, L2']):
    ax.set(xlabel='Simulation time (s)',ylabel=ylabel);ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('B2 + Z1: 50 nodes, dt = 10 ms, control = 20 ms\nDistributed runs aborted; no valid speedup comparison')
fig.savefig(D/'comparison.png',dpi=180);fig.savefig(D/'comparison.pdf');plt.close(fig)
print(json.dumps(summary,indent=2))
