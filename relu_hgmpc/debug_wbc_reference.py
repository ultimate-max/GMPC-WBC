import pathlib,numpy as np
import native
from common_wbc_reference import CommonReference
root=pathlib.Path(__file__).resolve().parents[1];p=native.Plant(str(root));r=CommonReference(p,'normal')
for t in [1.49,1.495,1.5,1.505,1.51,2.]:
    q=r.configuration(t);s=p.evaluate(q,np.zeros(24));v,a=r.base_derivatives(t,h=.001)
    print('time',t,'basev',v,'basea',a,'legcond',np.linalg.cond(s['Jc'][:,6:18]),'q',q[7:19],flush=True)
    arm_a=(r.arm(t+.001)+r.arm(t-.001)-2*r.arm(t))/1e-6
    print('arm acc',arm_a,flush=True)
