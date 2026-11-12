#include "mpc.hpp"
#include <omp.h>
#include <memory>
#include <fstream>
#include <exception>
#include <stdexcept>
#include <cmath>

namespace {
constexpr double BIG=1e20;
struct Rows {
  std::vector<Vec> rows; std::vector<double> lo,hi;
  void add(const Vec& row,double lower,double upper) { rows.push_back(row); lo.push_back(lower); hi.push_back(upper); }
  void block(const Mat& a,const Vec& l,const Vec& u) { for(int i=0;i<a.rows();++i) add(a.row(i).transpose(),l[i],u[i]); }
  void bound(int index,double l,double u) { Vec a=Vec::Zero(LOCAL);a[index]=1;add(a,l,u); }
  void finish(QP& p) {
    p.A.resize(rows.size(),LOCAL);p.l.resize(rows.size());p.u.resize(rows.size());
    for(int i=0;i<int(rows.size());++i) {p.A.row(i)=rows[i].transpose();p.l[i]=lo[i];p.u[i]=hi[i];}
  }
};
}
Problem makeProblem(const Snapshot& s,const Vec& velocity,const Eigen::Vector3d& target,int horizon,double dt,const std::vector<Eigen::Vector3d>& targets) {
  if(horizon<1 || dt<=0)
    throw std::runtime_error("Positive horizon/dt required");
  Problem result; result.horizon=horizon;result.dt=dt;
  result.targets=targets.empty()?std::vector<Eigen::Vector3d>(horizon,target):targets;
  if(int(result.targets.size())!=horizon) throw std::runtime_error("Reference horizon size mismatch");
  result.initial=Vec::Zero(NX);result.initial.tail(NV)=velocity;
  for(int k=0;k<horizon;++k) {
    QP p; p.P=Mat::Zero(LOCAL,LOCAL);p.q=Vec::Zero(LOCAL);
    // Coordinates: dq = difference(q_current,q_pred), not subtraction of quaternion entries.
    // Frozen tangent/dynamics model around q_current, v_current; local small-motion regime.
    Vec w=Vec::Constant(NV,1.0);w.head(3).setConstant(600);w.segment(3,3).setConstant(300);
    const double terminal=(k==horizon-1?4.0:1.0);
    Mat W=w.asDiagonal(); Mat H=W+2000*s.Jtip.transpose()*s.Jtip;
    p.P.block(NX,NX,NV,NV)=terminal*H;
    p.q.segment(NX,NV)=-terminal*(W*s.home_delta+2000*s.Jtip.transpose()*(result.targets[k]-s.tip));
    p.P.block(NX+NV,NX+NV,NV,NV).diagonal().setConstant(0.5*terminal);
    p.P.block(ACC,ACC,NV,NV).diagonal().setConstant(0.002);
    // Scale torque and force decision variables (100 Nm and 200 N units).
    p.P.block(TORQUE,TORQUE,18,18).diagonal().setConstant(0.1);
    p.P.block(FORCE,FORCE,12,12).diagonal().setConstant(0.04);
    Rows rows; Mat a=Mat::Zero(NX,LOCAL);
    a.block(0,NX,NX,NX).setIdentity();a.block(0,0,NX,NX)=-Mat::Identity(NX,NX);
    a.block(0,NV,NV,NV)=-dt*Mat::Identity(NV,NV);
    a.block(0,ACC,NV,NV)=-0.5*dt*dt*Mat::Identity(NV,NV);
    a.block(NV,ACC,NV,NV)=-dt*Mat::Identity(NV,NV);
    rows.block(a,Vec::Zero(NX),Vec::Zero(NX));
    // Full coupled inverse dynamics, including gravity. Normalize equation rows for solver scaling.
    a=Mat::Zero(NV,LOCAL);a.block(0,ACC,NV,NV)=s.M;
    a.block(6,TORQUE,18,18)=-100*Mat::Identity(18,18);
    a.block(0,FORCE,NV,12)=-200*s.Jfeet.transpose();
    Vec rhs=-s.h;
    for(int i=0;i<NV;++i) { double scale=std::max(1.0,a.row(i).norm());a.row(i)/=scale;rhs[i]/=scale; }
    rows.block(a,rhs,rhs);
    // Linearized fixed-foot acceleration with drift and Baumgarte stabilization.
    a=Mat::Zero(12,LOCAL);a.block(0,ACC,12,NV)=s.Jfeet;
    a.block(0,0,12,NV)=s.contact_kp*s.Jfeet;
    a.block(0,NV,12,NV)=s.contact_kd*s.Jfeet;
    Vec contact_rhs=-s.contact_drift-s.contact_kp*s.contact_error;
    rows.block(a,contact_rhs,contact_rhs);
    for(int i=0;i<NV;++i) {
      rows.bound(ACC+i,-10,10);
      rows.bound(NX+i,-0.2,0.2); // local trust region, metres/radians
      if(i>=6) {
        rows.bound(NX+i,std::max(-0.2,s.qmin_delta[i-6]),std::min(0.2,s.qmax_delta[i-6]));
        rows.bound(NX+NV+i,-s.velocity_limit[i-6],s.velocity_limit[i-6]);
        rows.bound(TORQUE+i-6,-s.torque_limit[i-6]/100,s.torque_limit[i-6]/100);
      }
    }
    const double mu=0.6/std::sqrt(2.0); // inscribed square: guarantees ||f_xy|| <= 0.6 f_z
    for(int f=0;f<4;++f) {
      rows.bound(FORCE+3*f+2,0,4.0); // 0...800 N
      for(int axis=0;axis<2;++axis) for(int sign:{-1,1}) {
        Vec row=Vec::Zero(LOCAL);row[FORCE+3*f+axis]=sign;row[FORCE+3*f+2]=-mu;
        rows.add(row,-BIG,0);
      }
    }
    rows.finish(p);result.stages.push_back(std::move(p));
  }
  return result;
}
QP centralized(const Problem& p) {
  const int n=p.horizon, m=p.stages[0].A.rows();
  QP c;c.P=Mat::Zero(n*LOCAL,n*LOCAL);c.q=Vec::Zero(n*LOCAL);
  c.A=Mat::Zero(n*m+n*NX,n*LOCAL);c.l=Vec::Zero(c.A.rows());c.u=c.l;
  for(int k=0;k<n;++k) {
    const auto& s=p.stages[k];int col=k*LOCAL;
    c.P.block(col,col,LOCAL,LOCAL)=s.P;c.q.segment(col,LOCAL)=s.q;
    c.A.block(k*m,col,m,LOCAL)=s.A;c.l.segment(k*m,m)=s.l;c.u.segment(k*m,m)=s.u;
    const int r=n*m+k*NX;
    c.A.block(r,col,NX,NX).setIdentity();
    if(k==0) {c.l.segment(r,NX)=p.initial;c.u.segment(r,NX)=p.initial;}
    else c.A.block(r,col-LOCAL+NX,NX,NX)=-Mat::Identity(NX,NX);
  }
  return c;
}
DistributedResult consensus(const Problem& p,int threads,int max_iter,double tolerance,double rho,const std::string& trace) {
  double start=now();int n=p.horizon;
  // A two-block consensus algorithm: all stage variables are one separable block,
  // all shared boundary states z are the other. Not naive multi-block ADMM.
  std::vector<std::unique_ptr<Solver>> solvers;
  std::vector<Vec> x(n,Vec::Zero(LOCAL)),u(n,Vec::Zero(2*NX)),z(n+1,Vec::Zero(NX));
  z[0]=p.initial;
  for(int k=0;k<n;++k) {
    QP local=p.stages[k];local.P.topLeftCorner(2*NX,2*NX).diagonal().array()+=rho;
    solvers.push_back(std::make_unique<Solver>(local,1e-8));
  }
  DistributedResult result;result.setup_ms=now()-start;
  std::ofstream log;if(!trace.empty()) {log.open(trace);log<<"iteration,primal_inf,dual_inf,objective\n";}
  start=now();
  for(int it=1;it<=max_iter;++it) {
    std::exception_ptr error;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(int k=0;k<n;++k) {
      try {
        Vec gradient=p.stages[k].q;
        gradient.head(NX)+=rho*(u[k].head(NX)-z[k]);
        gradient.segment(NX,NX)+=rho*(u[k].tail(NX)-z[k+1]);
        x[k]=solvers[k]->solve(&gradient).x;
      } catch(...) {
        #pragma omp critical
        { if(!error) error=std::current_exception(); }
      }
    }
    if(error) std::rethrow_exception(error);
    auto old=z;
    for(int k=1;k<n;++k) z[k]=0.5*(x[k-1].segment(NX,NX)+u[k-1].tail(NX)+x[k].head(NX)+u[k].head(NX));
    z[n]=x[n-1].segment(NX,NX)+u[n-1].tail(NX);
    double primal=0,dual=0,obj=0;
    for(int k=0;k<n;++k) {
      Vec residual(2*NX);residual<<x[k].head(NX)-z[k],x[k].segment(NX,NX)-z[k+1];
      u[k]+=residual;primal=std::max(primal,infnorm(residual));
      dual=std::max(dual,rho*std::max(infnorm(z[k]-old[k]),infnorm(z[k+1]-old[k+1])));
      obj+=objective(p.stages[k],x[k]);
    }
    if(log) log<<it<<','<<primal<<','<<dual<<','<<obj<<'\n';
    result.iterations=it;result.primal=primal;result.dual=dual;
    if(primal<tolerance && dual<tolerance) {result.converged=true;break;}
  }
  result.solve_ms=now()-start;result.x.resize(n*LOCAL);
  for(int k=0;k<n;++k) result.x.segment(k*LOCAL,LOCAL)=x[k];
  return result;
}
