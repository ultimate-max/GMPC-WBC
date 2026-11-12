import json,pathlib,numpy as np,native
root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/paper_comparison'
f=np.load(out/'paper400_normal_failure_state.npz');p=native.Plant(str(root));s=p.evaluate(f['q'],f['v'])
lo=s['q_min']-s['q'][7:];hi=s['q'][7:]-s['q_max'];vel=np.abs(s['v'][6:])-s['v_max']
r=dict(base_height=float(s['q'][2]),position_lower_excess_rad=float(lo.max()),position_upper_excess_rad=float(hi.max()),
    position_upper_joint_index_zero_based=int(hi.argmax()),velocity_excess_rad_s=float(vel.max()),
    joint_position=float(s['q'][7+hi.argmax()]),joint_upper_limit=float(s['q_max'][hi.argmax()]))
(out/'failure_detail.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
