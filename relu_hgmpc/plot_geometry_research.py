"""Summarize the independent offline experiment, without changing control."""
import json,pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/geometry_reuse_research'
d=json.loads((out/'diagnostics.json').read_text());rows=d['physical_reduced_reuse']['rows']
ages=[1,5,25,50];current=[r for r in rows if r['age_cycles']==1]
summary=[]
for age in ages:
    x=[r for r in rows if r['age_cycles']==age]
    summary.append(dict(age_ms=age*20,samples=len(x),
        cg_iterations_min=min(r['oracle_cg_iterations'] for r in x),
        cg_iterations_max=max(r['oracle_cg_iterations'] for r in x),
        cg_true_residual_pass=sum(r['oracle_cg_exact_residual_pass'] for r in x),
        cg_torque_error_max_Nm=max(r['oracle_cg_torque_error_Nm'] for r in x),
        preconditioned_condition_max=max(r['preconditioned_condition'] for r in x)))
stats=dict(age_summary=summary,
    age1_cg_ms_median=float(np.median([r['oracle_cg_solve_ms'] for r in current])),
    age1_fresh_factor_solve_ms_median=float(np.median([r['fresh_factor_solve_ms'] for r in current])),
    age1_reduction_ms_median=float(np.median([r['reduction_ms'] for r in current])),
    all_qr_difference_max=max(r['x_vs_independent_qr_max'] for r in d['physical_reduced_reuse']['independent_validation']),
    scope='Offline, five reference instants; CG threshold uses an expensive eigenvalue oracle; timings exclude full controller and oracle.')
(out/'summary.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats,indent=2))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(11.8,4.6),layout='constrained')
t=[r['time_s'] for r in current]
axes[0].semilogy(t,[r['shifted_warm_one_correction_torque_error_Nm'] for r in current],'-o',color='#b85b35',label='One correction after shifted warm start')
axes[0].semilogy(t,[r['oracle_cg_torque_error_Nm'] for r in current],'-o',color='#206b9b',label='Cached-factor CG + eigenvalue oracle')
axes[0].axhline(.001,color='.45',ls='--',label='Diagnostic target: 0.001 Nm')
axes[0].set(xlabel='Stored simulation time (s)',ylabel='Max first-step torque error (Nm)',title='Cache age 20 ms: correction accuracy')
axes[0].set_xticks(t);axes[0].legend(fontsize=8,loc='center left');axes[0].grid(alpha=.2)
x=np.array(ages)*20
ys=[[r['oracle_cg_iterations'] for r in rows if r['age_cycles']==a] for a in ages]
axes[1].fill_between(x,[min(y) for y in ys],[max(y) for y in ys],color='#206b9b',alpha=.15,label='Min–max over five stored instants')
axes[1].plot(x,[np.median(y) for y in ys],'-o',color='#206b9b',label='Median CG iterations')
axes[1].set(xlabel='Cache age (ms)',ylabel='CG iterations (count)',title='An aging cache needs more correction',xscale='log')
axes[1].set_xticks(x,[str(v) for v in x]);axes[1].legend(fontsize=8,loc='upper left');axes[1].grid(alpha=.2)
fig.suptitle('B2+Z1 offline reuse diagnostics — fixed contact; no closed-loop speedup demonstrated',fontsize=12)
fig.supxlabel('Source: central_osqp_12s_r1_trace.npz; snapshots at 2, 4, 6, 8, 10 s. Same 10-node GMPC QP.',fontsize=8)
fig.savefig(out/'reuse_diagnostics.png',dpi=170);plt.close(fig)
