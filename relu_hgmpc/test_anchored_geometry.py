import pathlib,json
import numpy as np
import pinocchio as pin
import native
from anchored_geometry import mount_transform,relative_arm,linearize,error_dynamics

def main():
    root=pathlib.Path(__file__).resolve().parents[1];p=native.Plant(str(root))
    model=pin.buildModelFromUrdf(str(root/'models/b2_z1.urdf'),pin.JointModelFreeFlyer())
    s0=p.truth();C=mount_transform(model,s0['q']);rng=np.random.default_rng(4081)
    worst=dict(closure=0.,jacobian=0.,velocity=0.,acceleration=0.)
    for _ in range(12):
        q=p.integrate(s0['q'],rng.normal(size=24)*.04);v=rng.normal(size=24)*.3;a=rng.normal(size=24)*.4
        s=p.evaluate(q,v);D=pin.SE3(s['pose'])*pin.exp6(rng.normal(size=6)*.2)
        vd=rng.normal(size=6)*.2;ad=rng.normal(size=6)*.3
        e,J,bt=linearize(s,C,D,vd);ee,de,Je,b=error_dynamics(s,C,D,vd,ad)
        def direct(qq,dd):
            st=p.evaluate(qq,np.zeros(24));F=relative_arm(st,C)
            Y=pin.SE3(st['pose'])*F.inverse();return pin.log6((dd*F.inverse()).inverse()*Y).vector
        worst['closure']=max(worst['closure'],float(np.max(abs(e-direct(q,D)))),float(np.max(abs(e-ee))))
        eps=1e-6
        Jnum=np.column_stack([(direct(p.integrate(q,eps*np.eye(24)[j]),D)-direct(p.integrate(q,-eps*np.eye(24)[j]),D))/(2*eps) for j in range(24)])
        worst['jacobian']=max(worst['jacobian'],float(np.max(abs(J-Jnum))),float(np.max(abs(J-Je))))
        eps=1e-4
        ep=direct(p.integrate(q,eps*v+.5*eps**2*a),D*pin.exp6(eps*vd+.5*eps**2*ad))
        em=direct(p.integrate(q,-eps*v+.5*eps**2*a),D*pin.exp6(-eps*vd+.5*eps**2*ad))
        worst['velocity']=max(worst['velocity'],float(np.max(abs((ep-em)/(2*eps)-de))),float(np.max(abs(J@v+bt-de))))
        worst['acceleration']=max(worst['acceleration'],float(np.max(abs((ep-2*e+em)/eps**2-(Je@a+b)))))
    assert worst['closure']<1e-10 and worst['jacobian']<1e-7 and worst['velocity']<1e-6 and worst['acceleration']<1e-5,worst
    print(json.dumps(worst,indent=2))
if __name__=='__main__':main()
