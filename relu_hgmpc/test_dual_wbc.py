import pathlib,json,time
import numpy as np
import native
from dual_wbc import CoordinatedWBC,inf
from geometric_wbc import rolling_contact
from anchored_geometry import AnchoredWBC
from model import equilibrium
from run_anchor_comparison import Goal

def main():
    root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/dual_wbc';out.mkdir(parents=True,exist_ok=True)
    p=native.Plant(str(root));target=Goal(p,root,'normal')
    du=CoordinatedWBC(p,root,'dual');un=CoordinatedWBC(p,root,'unified');old=AnchoredWBC(p,du.C)
    rng=np.random.default_rng(1112);worst={k:0. for k in ('mass','bias','maps','bounds','tasks','tau',
        'acceleration','force','legacy_tau','interface_acc','body_dynamics','arm_dynamics','wrench_power')}
    records=[]
    for i,t in enumerate(np.linspace(0.,11.,32)):
        t=round(t/.005)*.005;r=target.at(t);dq=rng.normal(size=24)*.0005
        q=p.integrate(r['q'],dq);v=r['v']+rng.normal(size=24)*.003
        s=rolling_contact(p.evaluate(q,v));prev=r['tau'].copy()
        du.prev_tau=prev.copy();un.prev_tau=prev.copy();old.prev_tau=prev.copy()
        pd=du.assemble(s,r);pu=un.assemble(s,r)
        mb=du.body.last;ma=du.arm.last
        fullM=np.zeros((24,24));fullh=np.zeros(24)
        fullM[np.ix_(du.body.idx,du.body.idx)]+=mb.M;fullM[np.ix_(du.arm.idx,du.arm.idx)]+=ma.M
        fullh[du.body.idx]+=mb.h;fullh[du.arm.idx]+=ma.h
        worst['mass']=max(worst['mass'],inf(fullM-s['M']));worst['bias']=max(worst['bias'],inf(fullh-s['h']))
        for key in ('T','a0','F','f0'):worst['maps']=max(worst['maps'],inf(pd['maps'][key]-pu['maps'][key]))
        for key in ('G','lo','hi'):
            mask=np.isfinite(pd[key]);assert np.array_equal(mask,np.isfinite(pu[key]))
            worst['bounds']=max(worst['bounds'],inf(pd[key][mask]-pu[key][mask]))
        for ld,lu in zip(pd['tasks'],pu['tasks']):
            worst['tasks']=max(worst['tasks'],*(inf(a-b) for a,b in zip(ld,lu)))
        td,ad=du.control(s,r);tu,au=un.control(s,r);to,ao=old.control(s,r)
        worst['tau']=max(worst['tau'],inf(td-tu));worst['acceleration']=max(worst['acceleration'],inf(ad-au))
        worst['force']=max(worst['force'],inf(du.debug['force']-un.debug['force']))
        worst['legacy_tau']=max(worst['legacy_tau'],inf(tu-to))
        worst['interface_acc']=max(worst['interface_acc'],du.info['interface_acc_residual'])
        worst['body_dynamics']=max(worst['body_dynamics'],du.info['body_dynamics_residual'])
        worst['arm_dynamics']=max(worst['arm_dynamics'],du.info['arm_dynamics_residual'])
        power=du.interface['wrench']@(mb.Ji@s['v'][du.body.idx]-ma.Ji@s['v'][du.arm.idx])
        worst['wrench_power']=max(worst['wrench_power'],abs(float(power)))
        records.append(dict(time=t,tau_difference=inf(td-tu),dual_ms=du.info['controller_ms'],unified_ms=un.info['controller_ms'],
            interface_acc_residual=du.info['interface_acc_residual']))
    # All frames and full dynamics are checked in unscaled physical units.
    for key in ('mass','bias','maps','bounds','tasks','interface_acc','body_dynamics','arm_dynamics','wrench_power'):
        assert worst[key]<1e-7,(key,worst)
    for key in ('tau','acceleration','force','legacy_tau'):assert worst[key]<2e-5,(key,worst)
    result=dict(samples=len(records),passed=True,worst=worst,records=records,
        meaning='Matching full and partitioned model, identical QP and certified hierarchy; no ADMM iterations.')
    (out/'equivalence_test.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(samples=len(records),worst=worst),indent=2))
if __name__=='__main__':main()
