"""Exact task-output locks, with original-objective KKT and full bound checks."""
import numpy as np
import scipy.linalg as la
from geometric_wbc import GeometricWBC
from central_relu import residuals
from online_osqp import OnlineOSQP
import quadprog
from scipy.optimize import nnls

class HierarchyQP:
    def solve_level(self,i,C,d,G,lo,hi,warm):
        H=2*C.T@C;g=2*C.T@d;scale=max(1.,np.max(np.abs(H)))
        zero=la.norm(G,axis=1)<1e-10
        il=np.flatnonzero(np.isfinite(lo)&~zero);iu=np.flatnonzero(np.isfinite(hi)&~zero)
        A=np.vstack((G[il],-G[iu]));b=np.r_[lo[il],-hi[iu]]
        rs=1/np.maximum(la.norm(A,axis=1),1e-12);A*=rs[:,None];b*=rs
        sol=quadprog.solve_qp(H/scale+1e-8*np.eye(len(g)),-g/scale,A.T,b)
        x=sol[0];dual=sol[4]*rs;y=np.zeros(len(lo))
        np.add.at(y,il,-dual[:len(il)]);np.add.at(y,iu,dual[len(il):])
        res=residuals(H/scale,g/scale,G,lo,hi,x,y)
        if max(res[k] for k in ('qp_primal_relative','qp_dual_relative','qp_normal_relative'))<=self.qp_tol:
            self.active_by_level[i]=np.flatnonzero(np.abs(y)>1e-8)
            self.dual_by_level[i]=y.copy()
            return x,res,int(sol[3][0])
        active=np.flatnonzero((A@x-b<1e-7)&(sol[4]>1e-9))
        for _ in range(len(g)+2):
            E=A[active]
            Z=la.null_space(E,rcond=1e-12) if len(active) else np.eye(len(g))
            direction=Z@la.lstsq(C@Z,-d-C@x,cond=1e-12)[0]
            ad=A@direction;slack=A@x-b;alpha=np.ones(len(b))
            blocking=(ad<-1e-10);blocking[active]=False
            alpha[blocking]=np.maximum(0,slack[blocking])/(-ad[blocking])
            j=int(np.argmin(alpha));step=min(1.,alpha[j]);x=x+step*direction
            if step<1.-1e-10:
                active=np.unique(np.r_[active,j]);continue
            grad=(H@x+g)/scale
            yface=la.lstsq(E.T,grad,cond=1e-12)[0] if len(active) else np.zeros(0)
            if len(active) and np.min(yface)<-1e-8:
                active=np.delete(active,np.argmin(yface));continue
            dual=np.zeros(len(b));dual[active]=np.maximum(0,yface)*rs[active]
            y=np.zeros(len(lo));np.add.at(y,il,-dual[:len(il)]);np.add.at(y,iu,dual[len(il):])
            res=residuals(H/scale,g/scale,G,lo,hi,x,y)
            error=max(res[k] for k in ('qp_primal_relative','qp_dual_relative','qp_normal_relative'))
            if error<=self.qp_tol:
                self.active_by_level[i]=np.flatnonzero(np.abs(y)>1e-8);self.fallbacks+=1
                self.dual_by_level[i]=y.copy()
                return x,res,int(sol[3][0])+1
            break
        np.savez('/tmp/shared_hierarchy_failed.npz',C=C,d=d,G=G,lo=lo,hi=hi,warm=warm)
        raise RuntimeError('Original hierarchy polishing failed')
    def __init__(self,engine_factory=OnlineOSQP):
        self.engine_factory=engine_factory
        self.active_by_level={};self.dual_by_level={};self.fallbacks=0;self.qp_tol=2e-7;self.engine=None;self.engine_dim=None
    def solve(self,levels,G,lo,hi,initial):
        x=initial.copy();Z=np.eye(len(x));locks=[];res=[];iterations=0;fast=0
        self.fallbacks=0;dims=[];osqp_calls=0;face_fast=0
        for i,(C,d) in enumerate(levels):
            if not Z.shape[1]:break
            Cr=C if i==0 else C@Z;dr=C@x+d
            if i==len(levels)-1:
                z=la.solve(Cr.T@Cr,-Cr.T@dr,assume_a='pos')
            else:z=la.lstsq(Cr,-dr,cond=1e-10,lapack_driver='gelsy')[0]
            scale=max(1.,2*np.max(np.sum(Cr*Cr,axis=0)))
            hx=2*Cr.T@(Cr@z)/scale;g=2*Cr.T@dr/scale
            err=float(np.max(np.abs(hx+g))/max(1.,np.max(np.abs(hx)),np.max(np.abs(g))))
            candidate=x+Z@z;gx=G@candidate
            bound=max(0.,np.max(lo-gx),np.max(gx-hi))
            rr=dict(qp_primal_relative=float(bound),qp_dual_relative=err,qp_normal_relative=float(bound))
            if err<1e-10 and bound<1e-9:
                active=np.array([],int);count=0;fast+=1
            else:
                Gr=G if i==0 else G@Z;lr=lo-G@x;ur=hi-G@x
                # A higher-level feasible point must remain feasible at z=0.
                # Normalize only nonzero inherited rows; otherwise 1e-13 of
                # bound roundoff is magnified by up to 1e12 in the QP solver.
                Gs=Gr.copy();ls=lr.copy();us=ur.copy();zero=la.norm(Gr,axis=1)<1e-10
                if np.any(zero) and max(0.,np.max(lr[zero]),np.max(-ur[zero]))>2e-7:
                    raise RuntimeError('Inherited zero-row bound exceeds physical roundoff tolerance')
                Gs[zero]=0;ls[zero]=-np.inf;us[zero]=np.inf
                if i>0:
                    ls[(ls>0)&(ls<2e-7)]=0;us[(us<0)&(us>-2e-7)]=0
                if i==len(levels)-1:
                    if self.engine is None or self.engine_dim!=Z.shape[1]:
                        self.engine=self.engine_factory();self.engine_dim=Z.shape[1]
                    # Reduced coordinates rotate between solves; do not shift
                    # old primal coordinates as if they were torque blocks.
                    # Active physical row IDs remain reusable and are certified.
                    self.engine.last=None
                    H=2*Cr.T@Cr;g=2*Cr.T@dr
                    z,y,count,oi=self.engine.solve(H/scale,g/scale,Gs,ls,us,0.,1.,1)
                    rr=residuals(H/scale,g/scale,Gr,lr,ur,z,y)
                    active=np.flatnonzero(np.abs(y)>1e-8)
                    osqp_calls+=int(not oi['active_face_fastpath']);face_fast+=oi['active_face_fastpath']
                else:
                    z,rr,count=self.solve_level(i,Cr,dr,Gs,ls,us,np.zeros(Z.shape[1]))
                    y=self.dual_by_level[i]
                    H=2*Cr.T@Cr;g=2*Cr.T@dr
                    rr=residuals(H/scale,g/scale,Gr,lr,ur,z,y)
                    active=self.active_by_level[i]
                if max(rr[k] for k in ('qp_primal_relative','qp_dual_relative','qp_normal_relative'))>self.qp_tol:
                    raise RuntimeError(f'Original inherited-row KKT check: {rr}')
            x+=Z@z;iterations+=count
            res.append(max(rr[k] for k in ('qp_primal_relative','qp_dual_relative','qp_normal_relative')))
            locks.append((C,C@x))
            if i<len(levels)-1:
                face=np.vstack((Cr,Gr[active])) if len(active) else Cr
                norm=la.norm(face,axis=1)
                face=face[norm>1e-10]/norm[norm>1e-10,None]
                Q,R,_=la.qr(face.T,mode='full',pivoting=True)
                diag=np.abs(np.diag(R));rank=int(np.sum(diag>1e-10*diag.max())) if len(diag) else 0
                Z=Z@Q[:,rank:]
            dims.append(Z.shape[1])
        lock=max(np.max(np.abs(C@x-value)) for C,value in locks)
        bound=max(0.,np.max(lo-G@x),np.max(G@x-hi))
        if lock>2e-7 or bound>2e-5:raise RuntimeError(f'Hierarchy final gate: {lock=}, {bound=}')
        return x,dict(hierarchy_kkt=max(res),hierarchy_lock=float(lock),hierarchy_bound=float(bound),
                      hierarchy_iterations=iterations,hierarchy_fast_levels=fast,hierarchy_fallbacks=self.fallbacks,
                      hierarchy_osqp_calls=osqp_calls,hierarchy_active_face_fast=face_fast,
                      hierarchy_dimensions=dims)
