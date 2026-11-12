#include "mpc.hpp"
#include <iostream>
#include <stdexcept>
int main(int argc,char** argv) {
 try {
  if(argc!=2) throw std::runtime_error("Project root required");
  Robot robot(argv[1]);robot.snapshot();
  Vec base=robot.q,delta=Vec::LinSpaced(NV,-0.02,0.02),v=Vec::LinSpaced(NV,0.03,-0.03);
  robot.q=robot.integrate(base,0.3*delta);
  const double eps=1e-6;
  Vec analytic=robot.rebaseVelocity(base,delta,v);
  Vec fd=(robot.difference(robot.q,robot.integrate(base,delta+eps*v))-
          robot.difference(robot.q,robot.integrate(base,delta)))/eps;
  if((analytic-fd).norm()>1e-7) throw std::runtime_error("Analytic chart derivative mismatch");
  robot.v=v;Vec current=robot.q;auto s=robot.snapshot();
  robot.q=robot.integrate(current,eps*v);auto next=robot.snapshot();robot.q=current;
  Vec drift=(next.Jfeet-s.Jfeet)*v/eps;
  if((drift-s.contact_drift).norm()>1e-6) throw std::runtime_error("Moving-contact Jdot*v mismatch");
  robot.subsystems(argv[1]); // validates coupled model at nonzero pose/velocity
  const int N=8;const double dt=0.04;Vec plan=Vec::Zero(N*LOCAL),a=Vec::Zero(NV);a.tail(18).setConstant(0.1);
  for(int k=0;k<N;++k) {
    for(int offset:{0,NX}) {
      double t=(k+(offset==NX?1:0))*dt;
      plan.segment(k*LOCAL+offset,NV)=0.5*t*t*a;plan.segment(k*LOCAL+offset+NV,NV)=t*a;
    }
    plan.segment(k*LOCAL+ACC,NV)=a;
  }
  robot.q=robot.integrate(base,0.5*dt*dt*a);robot.v=dt*a;
  Vec warm=shiftPlan(robot,base,plan,N,dt);
  for(int k=1;k<N;++k) if((warm.segment((k-1)*LOCAL+NX,NX)-warm.segment(k*LOCAL,NX)).norm()>1e-8)
    throw std::runtime_error("Shift/rebase breaks state continuity");
  if((warm.segment((N-1)*LOCAL+NX,NV)-0.5*((N+1)*(N+1)-1)*dt*dt*a).norm()>1e-8)
    throw std::runtime_error("Terminal extrapolation failed");
  QP q;q.P=Mat::Identity(1,1);q.q=Vec::Zero(1);q.A=Mat::Identity(1,1);q.l=Vec::Ones(1);q.u=10*Vec::Ones(1);
  Solver solver(q,1e-8,1);auto capped=solver.solve(nullptr,true);
  if(capped.iterations>1 || capped.converged) throw std::runtime_error("Finite-iteration budget not enforced");
  solver.budget(1000,1e-8);solver.warmStart(capped.x,&capped.y);auto solved=solver.solve();
  if(std::abs(solved.x[0]-1)>1e-6) throw std::runtime_error("Capped solve cannot resume");
  std::cout<<"PASS: analytic rebase, moving contact drift, subsystem dynamics, horizon shift, capped/resumed solve\n";
  return 0;
 } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
