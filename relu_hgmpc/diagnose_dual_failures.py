import json,pathlib,numpy as np
from scipy.optimize import linprog
root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/dual_wbc'
groups={'body_torque':(0,12),'body_slew':(12,24),'body_velocity':(24,36),
        'body_position':(36,48),'body_acceleration':(48,60),'arm_torque':(60,66),
        'arm_slew':(66,72),'arm_velocity':(72,78),'arm_position':(78,84),
        'arm_acceleration':(84,90),'friction_and_normal_force':(90,110)}
report={}
for name in ('dual_hold','unified_hold','paper_shared_normal'):
    p=out/f'{name}_failure_wbc.npz'
    if not p.exists():continue
    z=np.load(p);G,lo,hi=[z[k] for k in ('G','lo','hi')]
    il=np.flatnonzero(np.isfinite(lo));iu=np.flatnonzero(np.isfinite(hi))
    A=np.vstack((-G[il],G[iu]));b=np.r_[-lo[il],hi[iu]];ids=np.r_[il,iu]
    norm=np.maximum(np.linalg.norm(A,axis=1),1e-12);A/=norm[:,None];b/=norm
    feasibility=linprog(np.zeros(G.shape[1]),A_ub=A,b_ub=b,bounds=[(None,None)]*G.shape[1],method='highs')
    item={'feasible':bool(feasibility.success),'status':feasibility.message,'removing_group_feasible':{}}
    if not feasibility.success:
        for label,(a,c) in groups.items():
            sel=(ids<a)|(ids>=c)
            sol=linprog(np.zeros(G.shape[1]),A_ub=A[sel],b_ub=b[sel],bounds=[(None,None)]*G.shape[1],method='highs')
            item['removing_group_feasible'][label]=bool(sol.success)
        phase=linprog(np.r_[np.zeros(G.shape[1]),1.],A_ub=np.c_[A,-np.ones(len(A))],b_ub=b,
            bounds=[(None,None)]*G.shape[1]+[(0,None)],method='highs')
        item['normalized_min_slack']=float(phase.x[-1])
        act=np.flatnonzero(np.abs(phase.ineqlin.marginals)>1e-8)
        item['conflict_rows']=[dict(row=int(ids[j]),side='lower' if j<len(il) else 'upper',
              group=next(label for label,(a,c) in groups.items() if a<=ids[j]<c)) for j in act]
    report[name]=item
print(json.dumps(report,indent=2));(out/'failure_diagnosis.json').write_text(json.dumps(report,indent=2))
