#include "mpc.hpp"
Vec shiftPlan(Robot& robot,const Vec& old_q,const Vec& plan,int horizon,double dt) {
  Vec shifted=plan;
  for(int k=0;k<horizon;++k) {
    int next=std::min(k+1,horizon-1);shifted.segment(k*LOCAL,LOCAL)=plan.segment(next*LOCAL,LOCAL);
    for(int offset:{0,NX}) {
      Vec delta=plan.segment(next*LOCAL+offset,NV),v=plan.segment(next*LOCAL+offset+NV,NV);
      if(k==horizon-1) {
        delta=plan.segment(next*LOCAL+NX,NV);v=plan.segment(next*LOCAL+NX+NV,NV);
        if(offset==NX) {delta+=dt*v+0.5*dt*dt*plan.segment(next*LOCAL+ACC,NV);v+=dt*plan.segment(next*LOCAL+ACC,NV);}
      }
      Vec qpred=robot.integrate(old_q,delta);
      Vec rebased=robot.difference(robot.q,qpred);
      shifted.segment(k*LOCAL+offset,NV)=rebased;
      shifted.segment(k*LOCAL+offset+NV,NV)=robot.rebaseVelocity(old_q,delta,v);
    }
  }
  shifted.head(NV).setZero();shifted.segment(NV,NV)=robot.v;
  return shifted;
}
Vec shiftCentralDual(const Vec& y,const Problem& p) {
  Vec shifted=y;int m=p.stages[0].A.rows(),N=p.horizon;
  for(int k=0;k<N;++k) {
    int next=std::min(k+1,N-1);
    shifted.segment(k*m,m)=y.segment(next*m,m);
    shifted.segment(N*m+k*NX,NX)=y.segment(N*m+next*NX,NX);
  }
  return shifted;
}
