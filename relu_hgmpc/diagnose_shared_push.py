"""Deterministic fixed-clock replay; capture failed mathematical subproblem."""
import sys,pathlib,json,traceback
import numpy as np
import scipy.linalg as la
from scipy.optimize import linprog
from geometric_wbc import GeometricWBC
from shared_hierarchy import HierarchyQP
from online_osqp import OnlineOSQP
import run_shared_wbc
root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/shared_geometric_wbc'
mode=sys.argv[1] if len(sys.argv)>1 else 'allocation'
captured=[]
def wrap(cls,label):
    original=cls.solve_level
    def call(self,i,C,d,G,lo,hi,warm):
        try:return original(self,i,C,d,G,lo,hi,warm)
        except Exception as e:
            np.savez(out/f'diagnose_{mode}_{label}.npz',C=C,d=d,G=G,lo=lo,hi=hi,warm=warm)
            rows=1/np.maximum(la.norm(G,axis=1),1e-10)
            il=np.flatnonzero(np.isfinite(lo));iu=np.flatnonzero(np.isfinite(hi))
            A=np.vstack((-G[il]*rows[il,None],G[iu]*rows[iu,None]));b=np.r_[-lo[il]*rows[il],hi[iu]*rows[iu]]
            feasible=linprog(np.zeros(G.shape[1]),A_ub=A,b_ub=b,bounds=[(None,None)]*G.shape[1],method='highs')
            # Minimal common normalized violation to distinguish numerical
            # failure from an empty linear feasible set.
            relaxation=linprog(np.r_[np.zeros(G.shape[1]),1.],A_ub=np.column_stack((A,-np.ones(len(b)))),b_ub=b,
                bounds=[(None,None)]*G.shape[1]+[(0,None)],method='highs')
            captured.append(dict(layer=label,level=i,variables=G.shape[1],rows=len(lo),error=str(e),
                feasible=bool(feasible.success),feasibility_status=feasible.message,
                min_normalized_relaxation=float(relaxation.x[-1]) if relaxation.success else None,
                traceback=traceback.format_exc()))
            raise
    cls.solve_level=call
wrap(GeometricWBC,'wbc');wrap(HierarchyQP,'upper')
osqp_solve=OnlineOSQP.solve
def capture_osqp(self,H,g,G,lo,hi,t,dt,N):
    try:return osqp_solve(self,H,g,G,lo,hi,t,dt,N)
    except Exception as e:
        np.savez(out/f'diagnose_{mode}_osqp.npz',H=H,g=g,G=G,lo=lo,hi=hi)
        rs=1/np.maximum(la.norm(G,axis=1),1e-10);il=np.flatnonzero(np.isfinite(lo));iu=np.flatnonzero(np.isfinite(hi))
        A=np.vstack((-G[il]*rs[il,None],G[iu]*rs[iu,None]));b=np.r_[-lo[il]*rs[il],hi[iu]*rs[iu]]
        f=linprog(np.zeros(G.shape[1]),A_ub=A,b_ub=b,bounds=[(None,None)]*G.shape[1],method='highs')
        captured.append(dict(layer='upper_osqp',variables=G.shape[1],rows=len(lo),error=str(e),feasible=bool(f.success),feasibility_status=f.message))
        raise
OnlineOSQP.solve=capture_osqp
sys.argv=['run_shared_wbc','--clock','fixed','--backend',mode,'--case','push','--name',f'diagnose_{mode}_push']
try:run_shared_wbc.main()
except Exception:pass
(out/f'diagnosis_{mode}.json').write_text(json.dumps(captured,indent=2))
print(json.dumps(captured,indent=2))
