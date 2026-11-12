#include "mpc.hpp"
#include <memory>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <omp.h>
namespace {
constexpr int V=12,X=24,A=48,T=60,F=66;
struct Local { QP qp; Mat C; int width,nforce; std::vector<int> coordinates,shared; };
Local assemble(const Problem& p,const Snapshot& full,const SubsystemModel& sub,const Eigen::Vector3d& target) {
  const int nf=sub.Jfeet.rows(),L=72+nf,W=66+nf,N=p.horizon;
  Local result;result.width=L;result.nforce=nf;result.coordinates=sub.coordinates;
  std::vector<QP> stages;
  for(int k=0;k<N;++k) {
    QP s;s.P=Mat::Zero(L,L);s.q=Vec::Zero(L);
    const double terminal=k==N-1?4:1;
    Vec weights=Vec::Ones(V),reference(V);
    weights.head(3).setConstant(600./3);weights.segment(3,3).setConstant(300./3);
    for(int j=0;j<V;++j) reference[j]=full.home_delta[sub.coordinates[j]];
    Mat H=weights.asDiagonal();Vec g=-H*reference;
    if(sub.name=="arm") {
      Mat J(3,V);for(int j=0;j<V;++j) J.col(j)=full.Jtip.col(sub.coordinates[j]);
      H+=2000*J.transpose()*J;g-=2000*J.transpose()*((p.targets.empty()?target:p.targets[k])-full.tip);
    }
    s.P.block(X,X,V,V)=terminal*H;s.q.segment(X,V)=terminal*g;
    for(int j=0;j<V;++j) {
      s.P(X+V+j,X+V+j)=0.5*terminal*(j<6?1./3:1);
      s.P(A+j,A+j)=0.002*(j<6?1./3:1);
    }
    s.P.block(T,T,6,6).diagonal().setConstant(0.1);
    if(nf) s.P.block(F,F,nf,nf).diagonal().setConstant(0.04);
    std::vector<Vec> rows;std::vector<double> lower,upper;
    auto add=[&](const Vec& a,double l,double u){rows.push_back(a);lower.push_back(l);upper.push_back(u);};
    auto bound=[&](int index,double l,double u){Vec a=Vec::Zero(L);a[index]=1;add(a,l,u);};
    for(int j=0;j<X;++j) {
      Vec a=Vec::Zero(L);a[X+j]=1;a[j]=-1;
      if(j<V) {a[V+j]=-p.dt;a[A+j]=-0.5*p.dt*p.dt;} else a[A+j-V]=-p.dt;
      add(a,0,0);
    }
    Mat dyn=Mat::Zero(V,L);dyn.block(0,A,V,V)=sub.M;
    dyn.block(6,T,6,6)=-100*Mat::Identity(6,6);
    if(nf) dyn.block(0,F,V,nf)=-200*sub.Jfeet.transpose();
    // Interface wrench is expressed at the common base origin in base axes,
    // with [force; moment] ordering, conjugate to Pinocchio's free-flyer twist.
    dyn.block(0,W,6,6)=-200*Mat::Identity(6,6);
    for(int j=0;j<V;++j) {double scale=std::max(1.,dyn.row(j).norm());add(dyn.row(j).transpose()/scale,-sub.h[j]/scale,-sub.h[j]/scale);}
    for(int j=0;j<nf;++j) {
      Vec a=Vec::Zero(L);a.segment(A,V)=sub.Jfeet.row(j).transpose();
      a.head(V)=full.contact_kp*sub.Jfeet.row(j).transpose();
      a.segment(V,V)=full.contact_kd*sub.Jfeet.row(j).transpose();
      const int footrow=(sub.name=="front"?0:6)+j;
      double rhs=-full.contact_drift[footrow]-full.contact_kp*full.contact_error[footrow];add(a,rhs,rhs);
    }
    for(int j=0;j<V;++j) {
      bound(A+j,-10,10);bound(X+j,-0.2,0.2);
      if(j>=6) {
        int i=sub.coordinates[j]-6;
        bound(X+j,std::max(-0.2,full.qmin_delta[i]),std::min(0.2,full.qmax_delta[i]));
        bound(X+V+j,-full.velocity_limit[i],full.velocity_limit[i]);
        bound(T+j-6,-full.torque_limit[i]/100,full.torque_limit[i]/100);
      }
    }
    for(int foot=0;foot<nf/3;++foot) {
      bound(F+3*foot+2,0,4);
      for(int axis=0;axis<2;++axis) for(int sign:{-1,1}) {
        Vec a=Vec::Zero(L);a[F+3*foot+axis]=sign;a[F+3*foot+2]=-0.6/std::sqrt(2.);
        add(a,-1e20,0);
      }
    }
    s.A.resize(rows.size(),L);s.l.resize(rows.size());s.u.resize(rows.size());
    for(int j=0;j<int(rows.size());++j) {s.A.row(j)=rows[j].transpose();s.l[j]=lower[j];s.u[j]=upper[j];}
    stages.push_back(std::move(s));
  }
  const int m=stages[0].A.rows();auto& q=result.qp;
  q.P=Mat::Zero(N*L,N*L);q.q=Vec::Zero(N*L);q.A=Mat::Zero(N*(m+X),N*L);q.l=Vec::Zero(q.A.rows());q.u=q.l;
  result.C=Mat::Zero(N*18,N*L);
  for(int k=0;k<N;++k) {
    int c=k*L,r=k*m;auto& s=stages[k];
    q.P.block(c,c,L,L)=s.P;q.q.segment(c,L)=s.q;
    q.A.block(r,c,m,L)=s.A;q.l.segment(r,m)=s.l;q.u.segment(r,m)=s.u;
    r=N*m+k*X;q.A.block(r,c,X,X).setIdentity();
    if(k) q.A.block(r,c-L+X,X,X)=-Mat::Identity(X,X);
    else for(int j=0;j<V;++j) {
      q.l[r+V+j]=p.initial[NV+sub.coordinates[j]];q.u[r+V+j]=q.l[r+V+j];
    }
    for(int j=0;j<6;++j) {
      result.C(k*18+j,c+X+j)=1;
      result.C(k*18+6+j,c+X+V+j)=1;
      result.C(k*18+12+j,c+W+j)=1;
    }
  }
  for(int k=0;k<N;++k) {
    for(int j=0;j<6;++j) result.shared.push_back(k*L+X+j);
    for(int j=0;j<6;++j) result.shared.push_back(k*L+X+V+j);
    for(int j=0;j<6;++j) result.shared.push_back(k*L+W+j);
  }
  return result;
}
}
DistributedResult subsystemConsensus(const Problem& p,const Snapshot& full,
    const std::vector<SubsystemModel>& models,const Eigen::Vector3d& target,
    int threads,double rho,const std::string& trace) {
  if(models.size()!=3) throw std::runtime_error("Expected front/rear/arm");
  double t=now();std::vector<Local> local;
  std::vector<std::unique_ptr<Solver>> solvers;
  std::vector<Vec> x(3),u(3,Vec::Zero(18*p.horizon)),z=u,c=u;
  for(auto& model:models) {
    local.push_back(assemble(p,full,model,target));auto& l=local.back();
    QP qp=l.qp;qp.P+=rho*l.C.transpose()*l.C;
    solvers.push_back(std::make_unique<Solver>(qp,1e-8));
  }
  DistributedResult result;result.setup_ms=now()-t;
  std::ofstream log(trace);log<<"iteration,primal_inf,dual_inf,local_objective\n";
  t=now();
  for(int it=1;it<=20000;++it) {
    std::exception_ptr error;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(int i=0;i<3;++i) {
      try {Vec g=local[i].qp.q+rho*local[i].C.transpose()*(u[i]-z[i]);x[i]=solvers[i]->solve(&g).x;c[i]=local[i].C*x[i];}
      catch(...) {
        #pragma omp critical
        {if(!error) error=std::current_exception();}
      }
    }
    if(error) std::rethrow_exception(error);
    auto old=z;
    Vec mean=(c[0]+u[0]+c[1]+u[1]+c[2]+u[2])/3;
    for(int i=0;i<3;++i) for(int k=0;k<p.horizon;++k) {
      z[i].segment(k*18,12)=mean.segment(k*18,12);
      // Projection onto sum_i F_i=0 (action/reaction), not F_i=F_j.
      z[i].segment(k*18+12,6)=c[i].segment(k*18+12,6)+u[i].segment(k*18+12,6)-mean.segment(k*18+12,6);
    }
    result.primal=0;result.dual=0;double value=0;
    for(int i=0;i<3;++i) {
      u[i]+=c[i]-z[i];result.primal=std::max(result.primal,infnorm(c[i]-z[i]));
      result.dual=std::max(result.dual,rho*infnorm(z[i]-old[i]));value+=objective(local[i].qp,x[i]);
    }
    log<<it<<','<<result.primal<<','<<result.dual<<','<<value<<'\n';result.iterations=it;
    if(result.primal<2e-7 && result.dual<2e-6) {result.converged=true;break;}
  }
  result.solve_ms=now()-t;
  // Assemble a full-body command for comparison; averaging is only a reporting
  // operation. Finite-iteration consensus errors are still checked on full QP.
  result.x=Vec::Zero(LOCAL*p.horizon);
  for(int i=0;i<3;++i) for(int k=0;k<p.horizon;++k) {
    int c0=k*local[i].width,g=k*LOCAL;auto& ids=local[i].coordinates;
    for(int j=0;j<V;++j) {
      double weight=j<6?1./3:1.;int dest=ids[j];
      result.x[g+dest]+=weight*x[i][c0+j];result.x[g+NV+dest]+=weight*x[i][c0+V+j];
      result.x[g+NX+dest]+=weight*x[i][c0+X+j];result.x[g+NX+NV+dest]+=weight*x[i][c0+X+V+j];
      result.x[g+ACC+dest]+=weight*x[i][c0+A+j];
      if(j>=6) result.x[g+TORQUE+dest-6]=x[i][c0+T+j-6];
    }
    if(i<2) result.x.segment(g+FORCE+6*i,6)=x[i].segment(c0+F,6);
  }
  return result;
}

struct SubsystemController::Impl {
  std::vector<std::unique_ptr<Solver>> solvers;
  std::vector<Vec> x,y,u,z;
};
SubsystemController::SubsystemController():impl(std::make_unique<Impl>()) {}
SubsystemController::~SubsystemController()=default;
DistributedResult SubsystemController::step(const Problem& p,const Snapshot& full,
    const std::vector<SubsystemModel>& models,const Eigen::Vector3d& target,
    const OnlineBudget& budget,const Vec* shifted_global,bool shift_memory) {
  double t=now();const int N=p.horizon,nshared=18*N;
  auto& mem=*impl;bool first=mem.solvers.empty();
  std::vector<Local> local;std::vector<Vec> c(3,Vec::Zero(nshared));
  if(first) {mem.u.assign(3,Vec::Zero(nshared));mem.z=mem.u;mem.x.resize(3);mem.y.resize(3);}
  for(int i=0;i<3;++i) {
    local.push_back(assemble(p,full,models[i],target));auto& l=local.back();
    QP augmented=l.qp;for(int index:l.shared) augmented.P(index,index)+=budget.rho;
    if(first) {
      mem.solvers.push_back(std::make_unique<Solver>(augmented,budget.tolerance,budget.inner));
      mem.x[i]=Vec::Zero(l.qp.q.size());mem.y[i]=Vec::Zero(l.qp.l.size());
    } else {
      mem.solvers[i]->budget(budget.inner,budget.tolerance);mem.solvers[i]->update(augmented);
      if(shift_memory) {
        Vec prevx=mem.x[i],prevu=mem.u[i],prevy=mem.y[i];
        const int stage_rows=(l.qp.A.rows()-N*X)/N;
        for(int k=0;k<N;++k) {
          int next=std::min(k+1,N-1);
          mem.x[i].segment(k*l.width,l.width)=prevx.segment(next*l.width,l.width);
          mem.u[i].segment(k*18,18)=prevu.segment(next*18,18);
          mem.y[i].segment(k*stage_rows,stage_rows)=prevy.segment(next*stage_rows,stage_rows);
          mem.y[i].segment(N*stage_rows+k*X,X)=prevy.segment(N*stage_rows+next*X,X);
        }
      }
    }
    if(shifted_global) for(int k=0;k<N;++k) {
      const int g=k*LOCAL,loc=k*l.width;
      for(int j=0;j<V;++j) {
        int id=l.coordinates[j];
        mem.x[i][loc+j]=(*shifted_global)[g+id];mem.x[i][loc+V+j]=(*shifted_global)[g+NV+id];
        mem.x[i][loc+X+j]=(*shifted_global)[g+NX+id];mem.x[i][loc+X+V+j]=(*shifted_global)[g+NX+NV+id];
        mem.x[i][loc+A+j]=(*shifted_global)[g+ACC+id];
        if(j>=6) mem.x[i][loc+T+j-6]=(*shifted_global)[g+TORQUE+id-6];
      }
      if(i<2) mem.x[i].segment(loc+F,6)=shifted_global->segment(g+FORCE+6*i,6);
    }
    for(int j=0;j<nshared;++j) c[i][j]=mem.x[i][l.shared[j]];
    mem.solvers[i]->warmStart(mem.x[i],&mem.y[i]);
  }
  auto project=[&]() {
    Vec mean=(c[0]+mem.u[0]+c[1]+mem.u[1]+c[2]+mem.u[2])/3;
    for(int i=0;i<3;++i) for(int k=0;k<N;++k) {
      mem.z[i].segment(k*18,12)=mean.segment(k*18,12);
      mem.z[i].segment(k*18+12,6)=c[i].segment(k*18+12,6)+mem.u[i].segment(k*18+12,6)-mean.segment(k*18+12,6);
    }
  };
  project();DistributedResult result;result.setup_ms=now()-t;t=now();
  for(int iteration=0;iteration<budget.outer;++iteration) {
    std::exception_ptr error;
    #pragma omp parallel for num_threads(budget.threads) schedule(static)
    for(int i=0;i<3;++i) {
      try {
        Vec gradient=local[i].qp.q;
        for(int j=0;j<nshared;++j) gradient[local[i].shared[j]]+=budget.rho*(mem.u[i][j]-mem.z[i][j]);
        auto solved=mem.solvers[i]->solve(&gradient,true);mem.x[i]=solved.x;mem.y[i]=solved.y;
        for(int j=0;j<nshared;++j) c[i][j]=mem.x[i][local[i].shared[j]];
      } catch(...) {
        #pragma omp critical
        {if(!error) error=std::current_exception();}
      }
    }
    if(error) std::rethrow_exception(error);
    auto old=mem.z;project();result.primal=0;result.dual=0;
    for(int i=0;i<3;++i) {
      mem.u[i]+=c[i]-mem.z[i];result.primal=std::max(result.primal,infnorm(c[i]-mem.z[i]));
      result.dual=std::max(result.dual,budget.rho*infnorm(mem.z[i]-old[i]));
    }
    result.iterations=iteration+1;
    if(result.primal<budget.tolerance && result.dual<budget.tolerance) {result.converged=true;break;}
  }
  result.solve_ms=now()-t;result.x=Vec::Zero(LOCAL*N);
  for(int i=0;i<3;++i) for(int k=0;k<N;++k) {
    int c0=k*local[i].width,g=k*LOCAL;auto& ids=local[i].coordinates;
    for(int j=0;j<V;++j) {
      double w=j<6?1./3:1;int d=ids[j];
      result.x[g+d]+=w*mem.x[i][c0+j];result.x[g+NV+d]+=w*mem.x[i][c0+V+j];
      result.x[g+NX+d]+=w*mem.x[i][c0+X+j];result.x[g+NX+NV+d]+=w*mem.x[i][c0+X+V+j];
      result.x[g+ACC+d]+=w*mem.x[i][c0+A+j];
      if(j>=6) result.x[g+TORQUE+d-6]=mem.x[i][c0+T+j-6];
    }
    if(i<2) result.x.segment(g+FORCE+6*i,6)=mem.x[i].segment(c0+F,6);
  }
  return result;
}
