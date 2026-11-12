import pathlib,json
import numpy as np
root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/contact_audit'
for name in ['threshold','always','arm','hysteresis']:
    s=json.loads((out/f'{name}.json').read_text());z=np.load(out/f'{name}_trace.npz');trace=z['trace']
    sig=np.genfromtxt(out/f'{name}.csv',delimiter=',',names=True)
    print(name,dict(switches=s['switch_count'],min_contacts=s['contacts_min'],no_four_feet_ms=s['missing_contact_steps'],
        drift=s['max_center_drift_mm'],tracking=s['position_rmse_mm'],
        slip_net_mm=np.linalg.norm(s['per_foot_final_slip_xy_mm'],axis=1).round(3).tolist(),
        rotation_net_mm=np.linalg.norm(s['per_foot_final_roll_xy_mm'],axis=1).round(3).tolist(),
        air_net_mm=np.linalg.norm(s['per_foot_final_air_xy_mm'],axis=1).round(3).tolist(),
        slip_path_mm=np.array(s['per_foot_slip_path_mm']).round(3).tolist(),
        base_linear_acceleration=float(np.max(np.abs(z['desired_acc'][:,:3]))),
        base_angular_acceleration=float(np.max(np.abs(z['desired_acc'][:,3:6]))),
        arm_acceleration=float(np.max(np.abs(z['desired_acc'][:,6:]))),
        post_qp_contact=s['max_contact_post']))
    if name=='threshold':
        print('first switches',s['switch_events'][:5])
        old=np.genfromtxt(root/'results/prioritized/xie_6d_final_r1.csv',delimiter=',',names=True)
        print('reproduction error mm',np.max(np.abs(old['position_error_mm']-sig['position_error_mm'])))
        # Freeze the failing snapshot: determine whether the exact kinematic
        # acceleration has a feasible contact-force distribution without delta_a.
        from scipy.optimize import linprog
        for t in [2.78,2.8,2.82,3.8,6.]:
            j=round(t/.02);A=z['A'][j];lo=z['low'][j];hi=z['upp'][j]
            C=A[6:,6:];haslo=np.isfinite(lo[6:]);hashi=np.isfinite(hi[6:])
            result=linprog(np.zeros(12),A_eq=A[:6,6:],b_eq=lo[:6],
                A_ub=np.vstack((-C[haslo],C[hashi])),b_ub=np.r_[-lo[6:][haslo],hi[6:][hashi]],
                bounds=[(None,None)]*12,method='highs')
            print('delta_a=0 force feasibility',t,result.success,'delta',np.linalg.norm(z['delta_a'][j]),'sigma',sig['sigma'][j])
