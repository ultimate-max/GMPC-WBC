import pathlib,json,numpy as np
root=pathlib.Path(__file__).resolve().parents[1]/'results/anchored_mpc'
for name in ('v3_normal_anchor','v3_normal_direct','v3_normal_paper_shared','v3_normal_paper'):
    d=json.loads((root/f'{name}.json').read_text());a=np.load(root/f'{name}_trace.npz')
    print(name,{k:d[k] for k in ('completed_seconds','ee_position_error_mm','ee_rotation_error_deg','base_position_error_mm','minimum_contacts','minimum_sigma','upper_total_ms')})
    for k in ('q','v','reference_q','reference_v','acc'):
        x=a[k];print(k,'base',np.round(np.max(abs(x[:,:6]),axis=0),3),'arm',np.round(np.max(abs(x[:,-6:]),axis=0),3))
    print('end base q',a['q'][-1,:7], 'end ref',a['reference_q'][-1,:7])
