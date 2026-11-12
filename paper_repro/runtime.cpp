#include "simulation.hpp"
#include <functional>
#include <numeric>
#include <omp.h>
extern "C" {
#include "central/acados_solver_rti_central.h"
#include "front/acados_solver_rti_front.h"
#include "rear/acados_solver_rti_rear.h"
#include "arm/acados_solver_rti_arm.h"
#include "diag_central.h"
#include "diag_front.h"
#include "diag_rear.h"
#include "diag_arm.h"
}
constexpr int HORIZON=50;
constexpr double PRED_DT=.01, CONTROL_DT=.02;
using Eval=int(*)(const double**,double**,casadi_int*,double*,int);
using Work=int(*)(casadi_int*,casadi_int*,casadi_int*,casadi_int*);
struct Function {
  Eval eval;std::vector<casadi_int> iw;std::vector<double> w;
  Function(Eval e,Work f):eval(e) {casadi_int a,b,c,d;f(&a,&b,&c,&d);iw.resize(c);w.resize(d);}
  void run(std::vector<const double*> a,std::vector<double*> b) {
    if(eval(a.data(),b.data(),iw.data(),w.data(),0)) throw std::runtime_error("CasADi evaluation failed");
  }
};
struct RtiSolver {
  int id,nx,nu,nq,nv,np,nj;std::string name;
  ocp_nlp_config* config;ocp_nlp_dims* dims;ocp_nlp_in* in;ocp_nlp_out* out;ocp_nlp_solver* solver;
  std::function<int()> solve_fn,free_fn;std::function<int(int,double*,int)> param_fn;
  std::vector<Vec> x,u,p,dual;std::unique_ptr<Function> terms;std::unique_ptr<Function> step;
  double solve_ms=0,wall_ms=0,lin_ms=0,qp_ms=0,relax=1;int status=0;
  RtiSolver(int index,int cond,double rho,double sigma,double usigma,double relaxation,double height_tolerance):id(index),relax(index<0?1:relaxation) {
#define INIT(NAME,INDEX,NX,NU,NQ,NP) if(index==INDEX) { \
    name=#NAME;nx=NX;nu=NU;nq=NQ;np=NP; \
    auto* c=rti_##NAME##_acados_create_capsule(); \
    if(rti_##NAME##_acados_create(c)) throw std::runtime_error("acados create " #NAME); \
    if(rti_##NAME##_acados_update_qp_solver_cond_N(c,cond)) throw std::runtime_error("condensing setup"); \
    config=rti_##NAME##_acados_get_nlp_config(c);dims=rti_##NAME##_acados_get_nlp_dims(c); \
    in=rti_##NAME##_acados_get_nlp_in(c);out=rti_##NAME##_acados_get_nlp_out(c);solver=rti_##NAME##_acados_get_nlp_solver(c); \
    solve_fn=[c](){return rti_##NAME##_acados_solve(c);}; \
    param_fn=[c](int k,double* p,int n){return rti_##NAME##_acados_update_params(c,k,p,n);}; \
    free_fn=[c](){rti_##NAME##_acados_free(c);return rti_##NAME##_acados_free_capsule(c);}; \
    step=std::make_unique<Function>(NAME##_step,NAME##_step_work); }
    INIT(central,-1,49,30,25,3)
    INIT(front,0,25,12,13,89)
    INIT(rear,1,25,12,13,89)
    INIT(arm,2,25,6,13,83)
#undef INIT
    if(index==0) terms=std::make_unique<Function>(front_terms,front_terms_work);
    if(index==1) terms=std::make_unique<Function>(rear_terms,rear_terms_work);
    if(index==2) terms=std::make_unique<Function>(arm_terms,arm_terms_work);
    nv=nx-nq;nj=nv-6;
    x.assign(HORIZON+1,Vec::Zero(nx));u.assign(HORIZON,Vec::Zero(nu));
    p.assign(HORIZON+1,Vec::Zero(np));dual.assign(HORIZON+1,Vec::Zero(6));
    int contacts=index<0?4:(index<2?2:0);
    if(contacts && height_tolerance>0) for(int k=1;k<HORIZON;++k) {
      Vec lo(9*contacts),hi(9*contacts);
      ocp_nlp_constraints_model_get(config,dims,in,k,"lh",lo.data());
      ocp_nlp_constraints_model_get(config,dims,in,k,"uh",hi.data());
      lo.tail(contacts).setConstant(-height_tolerance);hi.tail(contacts).setConstant(height_tolerance);
      ocp_nlp_constraints_model_set(config,dims,in,out,k,"lh",lo.data());
      ocp_nlp_constraints_model_set(config,dims,in,out,k,"uh",hi.data());
    }
    if(index>=0) for(int k=0;k<=HORIZON;++k) {
      int base=k==HORIZON?(index<2?30:27):(index<2?42:33);int ny=base+31+(k<HORIZON?nu:0);Mat W=Mat::Zero(ny,ny);
      ocp_nlp_cost_model_get(config,dims,in,k,"W",W.data());
      // Eq. (12) places consensus penalties only on running nodes.
      for(int j=base;j<base+6;++j) W(j,j)=k<HORIZON?rho:0;
      for(int j=base+6;j<base+31;++j) W(j,j)=sigma;
      for(int j=base+31;j<ny;++j) W(j,j)=usigma;
      ocp_nlp_cost_model_set(config,dims,in,k,"W",W.data());
    }
  }
  ~RtiSolver(){free_fn();}
  Vec mapState(const Vec& whole) {
    if(id<0) return whole;
    Vec z(nx);z<<whole.head(7),whole.segment(7+6*id,6),whole.segment(25,6),whole.segment(31+6*id,6);return z;
  }
  Vec mapControl(const Vec& whole) {
    if(id<0) return whole;
    Vec z(nu);z.head(6)=whole.segment(6*id,6);if(id<2) z.tail(6)=whole.segment(18+6*id,6);return z;
  }
  void seed(const Vec& whole,const Vec& input) {
    for(auto& z:x) z=mapState(whole);for(auto& z:u) z=mapControl(input);
    // Regulate effort about the same feasible static feed-forward for both OCPs.
    // This makes home + static support an exact common zero-cost equilibrium.
    int base=id<0?63:(id<2?30:27),ny=id<0?93:base+nu+31+nu;
    Vec ref=Vec::Zero(ny);ref.segment(base,nu)=mapControl(input);
    for(int k=0;k<HORIZON;++k) ocp_nlp_cost_model_set(config,dims,in,k,"yref",ref.data());
  }
  void shift() {
    for(int k=0;k<=HORIZON;++k) {
      x[k]=x[std::min(k+2,HORIZON)];
      // Only running nodes carry consensus constraints. New tail multipliers
      // have no predecessor; holding the unconstrained terminal dual winds up.
      if(k+2<HORIZON) dual[k]=dual[k+2];else dual[k].setZero();
    }
    for(int k=0;k<HORIZON;++k) u[k]=u[std::min(k+2,HORIZON-1)];
  }
  void prepare(const Vec& truth) {
    Vec measured=mapState(truth);x[0]=measured;
    ocp_nlp_constraints_model_set(config,dims,in,out,0,"lbx",measured.data());
    ocp_nlp_constraints_model_set(config,dims,in,out,0,"ubx",measured.data());
    for(int k=0;k<=HORIZON;++k) {
      ocp_nlp_out_set(config,dims,out,in,k,"x",x[k].data());
      if(k<HORIZON) ocp_nlp_out_set(config,dims,out,in,k,"u",u[k].data());
      param_fn(k,p[k].data(),np);
    }
  }
  void solve() {
    double start=now();status=solve_fn();wall_ms=now()-start;
    double sec=0;ocp_nlp_get(solver,"time_tot",&sec);solve_ms=1000*sec;
    ocp_nlp_get(solver,"time_lin",&sec);lin_ms=1000*sec;
    ocp_nlp_get(solver,"time_qp",&sec);qp_ms=1000*sec;
    for(int k=0;k<=HORIZON;++k) {
      Vec old=x[k];
      ocp_nlp_out_get(config,dims,out,k,"x",x[k].data());
      x[k]=old+relax*(x[k]-old);x[k].segment(3,4).normalize();
      if(k<HORIZON) {old=u[k];ocp_nlp_out_get(config,dims,out,k,"u",u[k].data());u[k]=old+relax*(u[k]-old);}
    }
  }
};

Vec truthState(const Robot& r) {
  Vec z(49);z.head(25)=r.q;z.tail(24)=r.v;
  Eigen::Matrix3d R=Eigen::Quaterniond(r.q[6],r.q[3],r.q[4],r.q[5]).toRotationMatrix();
  z.segment(25,3)=R*r.v.head(3);z.segment(28,3)=R*r.v.segment(3,3);return z;
}
struct Controller {
  std::vector<std::unique_ptr<RtiSolver>> s; bool distributed;int threads;double alpha;bool rollout;bool edge=true;double dual_step=1;
  Function full_rollout{central_step,central_step_work};
  double update_ms=0,solve_ms=0,parallel_ms=0,total_ms=0,residual=0,residual_max=0;int failures=0;
  std::vector<Vec> merged_x,merged_u;
  Controller(bool d,int nt,int cond,double rho,double sigma,double usigma,double coupling,double relaxation,bool use_rollout,double height_tolerance):distributed(d),threads(d?nt:1),alpha(coupling),rollout(use_rollout) {
    if(d) for(int i=0;i<3;++i) s.push_back(std::make_unique<RtiSolver>(i,cond,rho,sigma,usigma,relaxation,height_tolerance));
    else s.push_back(std::make_unique<RtiSolver>(-1,cond,rho,sigma,usigma,relaxation,height_tolerance));
    merged_x.assign(HORIZON+1,Vec::Zero(49));merged_u.assign(HORIZON,Vec::Zero(30));
  }
  void seed(const Vec& x,const Vec& u) {for(auto& v:s) v->seed(x,u);}
  void run(const Vec& truth,const Eigen::Vector3d& origin,double time,bool shift) {
    double start=now();if(shift) for(auto& v:s) v->shift();
    if(distributed && rollout) {
      Vec global=truth,following(49),input=Vec::Zero(30),parameter=origin;
      for(int k=0;k<=HORIZON;++k) {
        for(auto& v:s) v->x[k]=v->mapState(global);
        if(k<HORIZON) {
          for(int i=0;i<3;++i) {input.segment(6*i,6)=s[i]->u[k].head(6);if(i<2) input.segment(18+6*i,6)=s[i]->u[k].tail(6);}
          full_rollout.run({global.data(),input.data(),parameter.data()},{following.data()});global=following;
        }
      }
    }
    if(distributed && !edge) for(int k=0;k<=HORIZON;++k) {
      // Algorithm 1 maps a common whole-body iterate to every local problem.
      // Reconcile base copies before the next RTI linearization and proximal term.
      Vec pose=Vec::Zero(7),twist=Vec::Zero(6);
      for(auto& v:s) {pose+=v->x[k].head(7)/3;twist+=v->x[k].segment(13,6)/3;}
      pose.segment(3,4).normalize();
      for(auto& v:s) {v->x[k].head(7)=pose;v->x[k].segment(13,6)=twist;}
    }
    for(auto& v:s) v->x[0]=v->mapState(truth);
    if(distributed) {
      std::vector<std::vector<Mat>> hs(3,std::vector<Mat>(HORIZON,Mat::Zero(6,6)));
      std::vector<std::vector<Vec>> ws(3,std::vector<Vec>(HORIZON,Vec::Zero(6)));
      // Snapshot all remote data before solving: Jacobi, never Gauss-Seidel.
#pragma omp parallel for num_threads(threads) schedule(static)
      for(int i=0;i<3;++i) for(int k=0;k<HORIZON;++k)
        s[i]->terms->run({s[i]->x[k].data(),s[i]->u[k].data()},{hs[i][k].data(),ws[i][k].data()});
      for(int k=0;k<=HORIZON;++k) {
        Vec z=Vec::Zero(6);for(auto& v:s) z+=v->x[k].segment(v->nq,6)/3;
        for(int i=0;i<3;++i) {
          auto& v=*s[i];v.p[k].head(3)=reference(origin,time+k*PRED_DT);
          Mat h=Mat::Zero(6,6);Vec w=Vec::Zero(6);
          for(int j=0;j<3;++j) if(i!=j) {int kk=std::min(k,HORIZON-1);h+=hs[j][kk];w+=ws[j][kk]-(1-alpha)*hs[j][kk]*s[j]->x[kk].segment(13,6);}
          v.p[k].segment(3,36)=Eigen::Map<Vec>(h.data(),36);v.p[k].segment(39,6)=w;
          v.p[k].segment(45,6)=(edge?Vec((3*z-v.x[k].segment(13,6))/2):z)-v.dual[k];v.p[k].segment(51,25)=v.x[k];v.p[k].segment(76,v.nu)=v.u[std::min(k,HORIZON-1)];v.p[k][v.np-1]=alpha;
        }
      }
    } else for(int k=0;k<=HORIZON;++k) s[0]->p[k]=reference(origin,time+k*PRED_DT);
    for(auto& v:s) v->prepare(truth);
    update_ms=now()-start;double solve_start=now();
#pragma omp parallel for num_threads(threads) schedule(static)
    for(int i=0;i<int(s.size());++i) s[i]->solve();
    parallel_ms=now()-solve_start;solve_ms=0;failures=0;
    for(auto& v:s) {solve_ms=std::max(solve_ms,v->solve_ms);failures+=v->status!=0;}
    residual=0;residual_max=0;
    if(distributed) {
      for(int k=0;k<=HORIZON;++k) {
        Vec& x=merged_x[k];x.setZero();
        for(int i=0;i<3;++i) {
          auto& v=*s[i];x.head(7)+=v.x[k].head(7)/3;x.segment(25,6)+=v.x[k].segment(13,6)/3;
          x.segment(7+6*i,6)=v.x[k].segment(7,6);x.segment(31+6*i,6)=v.x[k].tail(6);
          if(k<HORIZON) {merged_u[k].segment(6*i,6)=v.u[k].head(6);if(i<2) merged_u[k].segment(18+6*i,6)=v.u[k].tail(6);}
        }
        x.segment(3,4).normalize();
        for(int i=0;i<3;++i) {
          if(k<HORIZON) s[i]->dual[k]+=dual_step*(edge?1.5:1.)*(s[i]->x[k].segment(13,6)-x.segment(25,6));
          else s[i]->dual[k].setZero();
          for(int j=i+1;j<3;++j) {
            double n=(s[i]->x[k].segment(13,6)-s[j]->x[k].segment(13,6)).norm();
            residual+=n*n;residual_max=std::max(residual_max,n);
          }
        }
      }
      residual=std::sqrt(residual);
    } else {merged_x=s[0]->x;merged_u=s[0]->u;}
    total_ms=now()-start;
  }
};

int main(int argc,char** argv) {
 try {
  if(argc<2) throw std::runtime_error("Usage: paper_rti ROOT --mode central|distributed --duration 6 --name trial --threads 3");
  std::string root=argv[1];std::map<std::string,std::string> args;
  for(int i=2;i+1<argc;i+=2) args[argv[i]]=argv[i+1];
  auto opt=[&](std::string key,std::string def){return args.count(key)?args[key]:def;};
  std::string mode=opt("--mode","central"),name=opt("--name",mode);
  if(mode!="central" && mode!="distributed") throw std::runtime_error("Invalid mode");
  if(name.find('/')!=std::string::npos || name.find("..")!=std::string::npos) throw std::runtime_error("Invalid name");
  int steps=std::lround(std::stod(opt("--duration","6"))/CONTROL_DT),threads=std::stoi(opt("--threads","3"));
  bool stable_profile=mode=="distributed" && opt("--profile","stable")=="stable";
  int outer=std::stoi(opt("--outer",stable_profile?"10":"1"));
  if(steps<1 || threads<1 || threads>32 || outer<1) throw std::runtime_error("Invalid execution budget");
  Robot robot(root);auto initial=robot.snapshot();Simulation sim(root,robot);
  sim.model->opt.cone=mjCONE_ELLIPTIC;sim.model->opt.iterations=100;sim.model->opt.noslip_iterations=10;
  for(int g=0;g<sim.model->ngeom;++g) {sim.model->geom_solref[2*g]=.005;sim.model->geom_solref[2*g+1]=1;sim.model->geom_solimp[5*g]=.99;sim.model->geom_solimp[5*g+1]=.999;sim.model->geom_solimp[5*g+2]=.0001;sim.model->geom_friction[3*g]=.6;}
  mj_forward(sim.model,sim.data);sim.read(robot);sim.verifyGroundTruth(robot,initial);
  Vec state=truthState(robot),u=Vec::Zero(30);Mat Jb=initial.Jfeet.leftCols(6).transpose();
  Vec f=Jb.transpose()*(Jb*Jb.transpose()).ldlt().solve(initial.h.head(6));
  u.tail(12)=f;u.head(18)=initial.h.tail(18)-initial.Jfeet.rightCols(18).transpose()*f;
  int cond=std::stoi(opt("--condensing","50"));double rho=std::stod(opt("--rho","200")),sigma=std::stod(opt("--sigma","100"));
  double usigma=std::stod(opt("--input-prox","0"));
  Controller ctrl(mode=="distributed",threads,cond,rho,sigma,usigma,std::stod(opt("--coupling","1")),std::stod(opt("--relax",stable_profile?"0.2":"1")),opt("--rollout","0")=="1",std::stod(opt("--height-tolerance","0")));ctrl.seed(state,u);
  ctrl.dual_step=std::stod(opt("--dual-step",stable_profile?"0.05":"1"));ctrl.edge=opt("--admm-form","edge")=="edge";
  if(opt("--self-test","0")=="1") {
    if(!ctrl.distributed) throw std::runtime_error("Self-test needs distributed capsules");
    for(auto& s:ctrl.s) {
      for(int k=0;k<=HORIZON;++k) s->dual[k].setConstant(k+1);
      s->shift();
      for(int k=0;k<=HORIZON;++k) {
        double expected=k+2<HORIZON?k+3:0;
        if((s->dual[k].array()-expected).abs().maxCoeff()>1e-14) throw std::runtime_error("Dual shift regression");
        s->dual[k].setZero();
      }
    }
    ctrl.run(state,initial.tip,.8,false);
    for(auto& s:ctrl.s) if(s->dual[HORIZON].norm()!=0) throw std::runtime_error("Unconstrained terminal dual updated");
    std::cout<<"PASS: actual C++ dual shift and terminal update regression"<<std::endl;return 0;
  }
  int bootsteps=std::stoi(opt("--bootstrap",mode=="central"?"30":"200"));double bt=now();
  Function fullstep(central_step,central_step_work);Vec next(49),param=initial.tip;
  for(int i=0;i<bootsteps;++i) {
    ctrl.run(state,initial.tip,0,false);double defect=0;
    for(int k=0;k<HORIZON;++k) {fullstep.run({ctrl.merged_x[k].data(),ctrl.merged_u[k].data(),param.data()},{next.data()});defect+=(next-ctrl.merged_x[k+1]).squaredNorm();}
    if(i%20==0) std::cout<<"BOOT "<<i<<" residual="<<ctrl.residual<<" defect="<<std::sqrt(defect)<<" status="<<ctrl.failures<<std::endl;
    if(ctrl.residual<1e-3 && std::sqrt(defect)<1e-4 && !ctrl.failures) break;
  }
  double boot=now()-bt;
  std::filesystem::create_directories(root+"/results/paper_repro");
  std::ofstream log(root+"/results/paper_repro/"+name+".csv");log<<std::setprecision(12);
  log<<"time,error_m,base_z,tilt_rad,foot_drift_m,contacts,update_ms,max_local_solve_ms,parallel_wall_ms,controller_ms,interface_l2,interface_max,full_defect_l2,full_defect_max,solver_failures,torque_clips,critical_lin_ms,critical_qp_ms,net_interface_wrench_max";
  for(int j=0;j<25;++j) log<<",q"<<j;log<<'\n';
  double err2=0,peakerr=0,peakdef=0;int failed=0,clips_total=0;
  for(int t=0;t<steps;++t) {
    sim.read(robot);state=truthState(robot);
    double update_sum=0,solve_sum=0,parallel_sum=0,total_sum=0,lin_sum=0,qp_sum=0;
    for(int iter=0;iter<outer;++iter) {
      ctrl.run(state,initial.tip,t*CONTROL_DT,t>0 && iter==0);
      update_sum+=ctrl.update_ms;solve_sum+=ctrl.solve_ms;parallel_sum+=ctrl.parallel_ms;total_sum+=ctrl.total_ms;
      auto critical=std::max_element(ctrl.s.begin(),ctrl.s.end(),[](const auto& a,const auto& b){return a->solve_ms<b->solve_ms;});
      lin_sum+=(*critical)->lin_ms;qp_sum+=(*critical)->qp_ms;
    }
    ctrl.update_ms=update_sum;ctrl.solve_ms=solve_sum;ctrl.parallel_ms=parallel_sum;ctrl.total_ms=total_sum;
    auto snapshot=robot.snapshot();Vec qstart=robot.q,vstart=robot.v;
    double defect=0,maxdef=0;
    for(int k=0;k<HORIZON;++k) {
      fullstep.run({ctrl.merged_x[k].data(),ctrl.merged_u[k].data(),param.data()},{next.data()});
      double d=(next-ctrl.merged_x[k+1]).norm();defect+=d*d;maxdef=std::max(maxdef,d);
    }
    defect=std::sqrt(defect);peakdef=std::max(peakdef,defect);
    double net_wrench=0;
    if(ctrl.distributed) for(int k=0;k<HORIZON;++k) {
      Vec net=Vec::Zero(6);
      for(auto& local:ctrl.s) {
        Mat H(6,6);Vec momentum(6),prediction(local->nx);
        local->terms->run({local->x[k].data(),local->u[k].data()},{H.data(),momentum.data()});
        local->step->run({local->x[k].data(),local->u[k].data(),local->p[k].data()},{prediction.data()});
        net+=(H*prediction.segment(13,6)-momentum)/PRED_DT;
      }
      net_wrench=std::max(net_wrench,net.norm());
    }
    double error=(snapshot.tip-reference(initial.tip,t*CONTROL_DT)).norm();err2+=error*error;peakerr=std::max(peakerr,error);
    Eigen::Matrix3d R=Eigen::Quaterniond(robot.q[6],robot.q[3],robot.q[4],robot.q[5]).toRotationMatrix();
    double tilt=std::acos(std::clamp(R(2,2),-1.,1.)),drift=0;
    for(int j=0;j<4;++j) drift=std::max(drift,snapshot.contact_error.segment(3*j,3).norm());
    int contacts=sim.stanceContacts(),clips=0;failed+=ctrl.failures;
    if(ctrl.failures || !std::isfinite(defect) || !std::isfinite(ctrl.residual)) {
      std::ofstream failure(root+"/results/paper_repro/"+name+"_failure.json");
      failure<<"{\"time\":"<<t*CONTROL_DT<<",\"solver_failures\":"<<ctrl.failures<<",\"reason\":\"solver status or nonfinite prediction; no command applied\"}";
      throw std::runtime_error("Solver validity failed at t="+std::to_string(t*CONTROL_DT)+" failures="+std::to_string(ctrl.failures));
    }
    if(!ctrl.merged_u[0].allFinite() || !robot.q.allFinite() || robot.q[2]<.35 || tilt>.5)
      throw std::runtime_error("Physical validity failed at "+std::to_string(t*CONTROL_DT));
    Vec accel=(ctrl.merged_x[1].tail(18)-state.tail(18))/PRED_DT;
    for(int j=0;j<20;++j) {
      sim.read(robot);double elapsed=(j+1)*.001;
      Vec qr=qstart.tail(18)+elapsed*vstart.tail(18)+.5*elapsed*elapsed*accel;
      Vec vr=vstart.tail(18)+elapsed*accel;
      Vec command=ctrl.merged_u[0].head(18);
      for(int k=0;k<18;++k) {
        int dof=sim.model->jnt_dofadr[sim.joints[k]];
        command[k]+=(k<12?40:20)*(qr[k]-robot.q[7+k])+(k<12?2:1)*(vr[k]-robot.v[6+k]);
        command[k]+=sim.model->dof_damping[dof]*robot.v[6+k]+sim.model->dof_frictionloss[dof]*std::tanh(vr[k]/.005);
      }
      clips+=sim.apply(command,snapshot.torque_limit);
    }
    clips_total+=clips;
    log<<t*CONTROL_DT<<','<<error<<','<<qstart[2]<<','<<tilt<<','<<drift<<','<<contacts<<','<<ctrl.update_ms<<','<<ctrl.solve_ms<<','<<ctrl.parallel_ms<<','<<ctrl.total_ms<<','<<ctrl.residual<<','<<ctrl.residual_max<<','<<defect<<','<<maxdef<<','<<ctrl.failures<<','<<clips;
    log<<','<<lin_sum<<','<<qp_sum<<','<<net_wrench;
    for(int j=0;j<25;++j) log<<','<<qstart[j];log<<'\n';
    if(t%50==0) std::cout<<name<<" t="<<t*CONTROL_DT<<" error_mm="<<1000*error<<" total_ms="<<ctrl.total_ms<<" residual="<<ctrl.residual<<" defect="<<defect<<" statuses="<<ctrl.failures<<std::endl;
  }
  std::cout<<"SUMMARY "<<name<<" bootstrap_ms="<<boot<<" rmse_mm="<<1000*std::sqrt(err2/steps)<<" max_mm="<<1000*peakerr<<" max_defect="<<peakdef<<" failed_solves="<<failed<<" clips="<<clips_total<<std::endl;
  return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
