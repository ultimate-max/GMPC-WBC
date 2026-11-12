import json,pathlib,numpy as np
root=pathlib.Path(__file__).resolve().parents[1]/'results/buffered_wbc'
d=np.load(root/'native_mrt400_hold_trace.npz');tr=d['trace'];bad=tr[:,1:5].sum(axis=1)<4
indices=np.flatnonzero(bad);ranges=[]
if len(indices):
    groups=np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)
    ranges=[dict(start=float(tr[g[0],0]),end=float(tr[g[-1],0]+.0005),samples=len(g),missing_feet=np.flatnonzero(tr[g,1:5].min(axis=0)<1).tolist()) for g in groups]
result=dict(total_loss_seconds=float(bad.sum()*.0005),intervals=ranges,
    interpretation='Hold means EE pose held fixed while base target continues moving. Lower-priority base tracking can yield to EE and physical joint constraints; contact loss is nevertheless a physical limitation, not certified by the QP residual.')
(root/'hold_contact_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
