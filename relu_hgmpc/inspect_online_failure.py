import pathlib,numpy as np
from scipy.optimize import linprog
root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/online_wbc_mpc'
a=np.genfromtxt(out/'double_fixed5_r1.csv',names=True,delimiter=',')
for i in range(600,len(a),10):
    print({k:float(a[k][i]) for k in ['time','task_residual_score','tracking_residual_score','torque_margin','friction_margin','acceleration_margin','base_position_error_mm']})
path=pathlib.Path('/tmp/online_mpc_failed_qp.npz')
if path.exists():
    z=np.load(path);G,lo,hi=z['G'],z['lo'],z['hi'];print('shape',G.shape,'time',z['time'])
    print('initial joint reference minus measured',z['anchor_q'][7:]-z['q'][7:])
    print('initial desired joint velocities',z['anchor_v'][6:])
    for omitted in [None, 'trust','slew','acceleration']:
        keep=np.ones(len(lo),bool)
        # Per stage: tau18,cone20,slew18,acc18,q18,trust18,v18,base6 =134.
        for k in range(10):
            if omitted=='trust':keep[k*134+92:k*134+110]=False
            if omitted=='slew':keep[k*134+38:k*134+56]=False
            if omitted=='acceleration':keep[k*134+56:k*134+74]=False
        il=np.flatnonzero(keep&np.isfinite(lo));iu=np.flatnonzero(keep&np.isfinite(hi))
        lp=linprog(np.zeros(G.shape[1]),A_ub=np.vstack((-G[il],G[iu])),b_ub=np.r_[-lo[il],hi[iu]],bounds=[(None,None)]*G.shape[1],method='highs')
        print('omit',omitted,lp.message)
        if omitted=='trust' and lp.success:
            error=np.maximum(lo-G@lp.x,G@lp.x-hi)
            indices=np.argsort(error)[-12:]
            print('trust violated',[(int(i//134),int(i%134-92),float(error[i])) for i in indices])
