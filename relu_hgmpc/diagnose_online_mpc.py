import pathlib,numpy as np,pinocchio as pin,native
from run_online_wbc_mpc import Target
from online_wbc_mpc import OnlineMPC,pose
root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/online_wbc_mpc';p=native.Plant(str(root));target=Target(p,'normal',4.,out)
tr=np.load(out/'smoke_fixed10_trace.npz');mpc=OnlineMPC(p,root,target)
for t in [.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1.]:
    i=round(t/.005);print('wrist',t,'q/qref/goal',tr['q'][i,-1],tr['reference_q'][i,-1],target.at(t)['q'][-1],
        'v/vref/goal',tr['v'][i,-1],tr['reference_v'][i,-1],target.at(t)['v'][-1],
        'a/aref/goal',tr['acc'][i,-1],tr['reference_a'][i,-1],target.at(t)['a'][-1],flush=True)
raise SystemExit
for t in [0.,.4,.8,1.,2.]:
    i=round(t/.005);s=p.evaluate(tr['q'][i],tr['v'][i]);plan=mpc.solve(s,t,tr['tau'][max(0,i-1)],tr['acc'][max(0,i-1)])
    print('time',t,'current base',s['q'][:3],'goal',target.at(t)['q'][:3],flush=True)
    for dt in [.04,.12,.4]:
        q,v,a=plan.state(t+dt);goal=target.at(t+dt)
        e=p.difference(q,goal['q'])
        print(' lookahead',dt,'base error',e[:6],'arm error',e[18:],'base acc',a[:6],'arm acc',a[18:],flush=True)
    zref=np.r_[*[target.at(t+k*.04)['tau']/mpc.scale for k in range(10)]];d=mpc.debug
    print('obj opt, nominal',.5*d['z']@d['H']@d['z']+d['g']@d['z'],.5*zref@d['H']@zref+d['g']@zref,
        'nominal violation',max(0,np.max(d['lo']-d['G']@zref),np.max(d['G']@zref-d['hi'])),flush=True)
