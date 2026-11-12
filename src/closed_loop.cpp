#include "mpc.hpp"
#include <mujoco/mujoco.h>
#include <Eigen/Geometry>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <cmath>
#include <stdexcept>

struct Simulation {
  mjModel* model; mjData* data;
  std::vector<int> joints,actuators;std::vector<int> feet;
  Simulation(const std::string& root,Robot& robot) {
    char error[2048];model=mj_loadXML((root+"/models/b2_z1.xml").c_str(),nullptr,error,sizeof(error));
    if(!model) throw std::runtime_error(error);
    data=mj_makeData(model);mj_resetDataKeyframe(model,data,0);
    if(model->nv!=24 || model->nu!=18 || std::abs(model->opt.timestep-0.001)>1e-12)
      throw std::runtime_error("Unexpected MuJoCo dimensions/timestep");
    for(auto& name:robot.jointNames()) {
      int j=mj_name2id(model,mjOBJ_JOINT,name.c_str()),a=mj_name2id(model,mjOBJ_ACTUATOR,(name+"_motor").c_str());
      if(j<0 || a<0) throw std::runtime_error("Missing joint/actuator "+name);
      joints.push_back(j);actuators.push_back(a);
    }
    for(auto name:{"FL_foot","FR_foot","RL_foot","RR_foot"}) feet.push_back(mj_name2id(model,mjOBJ_BODY,name));
    mj_forward(model,data);
  }
  ~Simulation() {mj_deleteData(data);mj_deleteModel(model);}
  void read(Robot& r) {
    r.q.head(3)=Eigen::Map<Vec>(data->qpos,3);
    r.q.segment(3,4)<<data->qpos[4],data->qpos[5],data->qpos[6],data->qpos[3];
    Eigen::Matrix3d R=Eigen::Quaterniond(r.q[6],r.q[3],r.q[4],r.q[5]).toRotationMatrix();
    r.v.head(3)=R.transpose()*Eigen::Map<Vec>(data->qvel,3);
    r.v.segment(3,3)=Eigen::Map<Vec>(data->qvel+3,3);
    for(int i=0;i<18;++i) {r.q[7+i]=data->qpos[model->jnt_qposadr[joints[i]]];r.v[6+i]=data->qvel[model->jnt_dofadr[joints[i]]];}
  }
  int stanceContacts() {
    bool active[4]={false,false,false,false};
    for(int i=0;i<data->ncon;++i) {
      int b1=model->geom_bodyid[data->contact[i].geom[0]],b2=model->geom_bodyid[data->contact[i].geom[1]];
      for(int f=0;f<4;++f) if((b1==0 && b2==feet[f]) || (b2==0 && b1==feet[f])) active[f]=true;
    }
    return int(active[0])+int(active[1])+int(active[2])+int(active[3]);
  }
  void verifyGroundTruth(Robot& robot,const Snapshot& s) {
    Mat transform=Mat::Zero(24,24);
    Eigen::Matrix3d R=Eigen::Quaterniond(robot.q[6],robot.q[3],robot.q[4],robot.q[5]).toRotationMatrix();
    transform.topLeftCorner(3,3)=R;transform.block(3,3,3,3).setIdentity();
    for(int i=0;i<18;++i) transform(model->jnt_dofadr[joints[i]],6+i)=1;
    for(int f=0;f<4;++f) {
      Vec position=Eigen::Map<Vec>(data->xpos+3*feet[f],3);
      if((position-s.feet.segment(3*f,3)).norm()>1e-8) throw std::runtime_error("Pinocchio/MuJoCo foot position mismatch");
      position[2]-=0.032;
      Eigen::Matrix<double,3,24,Eigen::RowMajor> j;
      mj_jac(model,data,j.data(),nullptr,position.data(),feet[f]);
      if((j*transform-s.Jfeet.middleRows(3*f,3)).norm()>1e-8)
        throw std::runtime_error("Pinocchio/MuJoCo contact Jacobian mismatch");
    }
  }
  int apply(const Vec& torque,const Vec& limits) {
    int clipped=0;
    for(int i=0;i<18;++i) {double v=std::clamp(torque[i],-limits[i],limits[i]);clipped+=v!=torque[i];data->ctrl[actuators[i]]=v;}
    mj_step(model,data);return clipped;
  }
};

Eigen::Vector3d reference(const Eigen::Vector3d& origin,double time) {
  const double t=std::max(0.0,time-0.5),r=std::min(1.0,t),ramp=r*r*r*(10+r*(-15+6*r));
  const double phase=2*M_PI*0.2*t;
  return origin+ramp*Eigen::Vector3d(0.025*std::sin(phase),0.015*(1-std::cos(phase)),0.010*std::sin(phase));
}
int main(int argc,char** argv) {
 try {
  if(argc<2) throw std::runtime_error("Usage: b2z1_closed_loop ROOT --mode central|admm [--duration 6 --outer 10 --inner 100 --central 500 --name run]");
  const std::string root=argv[1];std::map<std::string,std::string> args;
  for(int i=2;i+1<argc;i+=2) args[argv[i]]=argv[i+1];
  auto str=[&](const std::string& key,const std::string& fallback){return args.count(key)?args[key]:fallback;};
  const std::string mode=str("--mode","central"),name=str("--name",mode);
  if(mode!="central" && mode!="admm") throw std::runtime_error("Unknown mode");
  if(name.find('/')!=std::string::npos || name.find("..")!=std::string::npos) throw std::runtime_error("Invalid result name");
  const double duration=std::stod(str("--duration","6")),dt=0.04;
  const int N=8,steps=int(std::round(duration/dt)),substeps=40;
  OnlineBudget budget;budget.outer=std::stoi(str("--outer","10"));budget.inner=std::stoi(str("--inner","100"));
  budget.threads=std::stoi(str("--threads","3"));budget.rho=std::stod(str("--rho","1"));
  budget.tolerance=1e-5;int central_cap=std::stoi(str("--central","500"));
  bool warm=str("--warm","1")=="1";
  if(steps<1 || budget.outer<1 || budget.inner<1 || budget.threads<1 || central_cap<1) throw std::runtime_error("Invalid simulation budget");
  Robot robot(root);auto initial=robot.snapshot(); // defines fixed world footholds
  Simulation sim(root,robot);sim.read(robot);
  sim.model->opt.cone=mjCONE_ELLIPTIC;
  sim.model->opt.iterations=100;
  sim.model->opt.noslip_iterations=10;
  // Default soft friction permits systematic stance creep under tangential
  // preload. These explicit contact settings are identical for both solvers.
  for(int g=0;g<sim.model->ngeom;++g) {
    sim.model->geom_solref[2*g]=0.005;sim.model->geom_solref[2*g+1]=1;
    sim.model->geom_solimp[5*g]=0.99;sim.model->geom_solimp[5*g+1]=0.999;sim.model->geom_solimp[5*g+2]=0.0001;
    sim.model->geom_friction[3*g]=0.6;
  }
  mj_forward(sim.model,sim.data);
  sim.verifyGroundTruth(robot,initial);
  const Eigen::Vector3d tip_origin=initial.tip;
  auto parts=robot.subsystems(root,false); // preload model caches outside timed loop
  std::unique_ptr<Solver> central;auto distributed=std::make_unique<SubsystemController>();
  auto p=makeProblem(initial,robot.v,tip_origin,N,dt);auto global=centralized(p);
  double boot=now();Solution previous;DistributedResult dr;
  if(mode=="central") {
    central=std::make_unique<Solver>(global,1e-6,20000);previous=central->solve();
  } else {
    OnlineBudget b=budget;b.outer=3000;b.inner=2000;b.tolerance=1e-6;
    dr=distributed->step(p,initial,parts,tip_origin,b);previous=dr;
    if(!dr.converged || violation(global,dr.x)>1e-4) throw std::runtime_error("ADMM bootstrap failed");
  }
  boot=now()-boot;Vec previous_q=robot.q;
  const auto directory=root+"/results/closed_loop";std::filesystem::create_directories(directory);
  std::ofstream log(directory+"/"+name+".csv");log<<std::setprecision(12);
  log<<"time,target_x,target_y,target_z,tip_x,tip_y,tip_z,error_m,base_z,tilt_rad,foot_drift_m,contacts,model_ms,update_ms,solve_ms,controller_ms,iterations,converged,primal,dual,qp_violation,dynamics_residual,torque_clips";
  for(int j=0;j<25;++j) log<<",q"<<j;log<<'\n';
  double squared_error=0,max_error=0,max_tilt=0,max_drift=0,min_height=1e10;
  std::vector<double> timings;int total_clips=0,allfeet=0,limited=0;
  for(int step=0;step<steps;++step) {
    sim.read(robot);double tick=now();auto s=robot.snapshot();
    if(mode=="admm") parts=robot.subsystems(root,false);
    double model_ms=now()-tick;Eigen::Vector3d target=reference(tip_origin,step*dt);
    std::vector<Eigen::Vector3d> preview;for(int k=0;k<N;++k) preview.push_back(reference(tip_origin,(step+k+1)*dt));
    Vec shifted=step?shiftPlan(robot,previous_q,previous.x,N,dt):previous.x;
    Vec shifted_dual;
    Solution answer;double primal=0,dual=0,update_ms=0,solve_ms=0;
    if(mode=="central") {
      double update_start=now();
      p=makeProblem(s,robot.v,target,N,dt,preview);global=centralized(p);
      if(!warm) central=std::make_unique<Solver>(global,1e-5,central_cap);
      else {central->budget(central_cap,1e-5);central->update(global);shifted_dual=step?shiftCentralDual(previous.y,p):previous.y;central->warmStart(shifted,&shifted_dual);}
      update_ms=now()-update_start;answer=central->solve(nullptr,true);solve_ms=answer.solve_ms;
    } else {
      p.horizon=N;p.dt=dt;p.initial=Vec::Zero(NX);p.initial.tail(NV)=robot.v;
      p.targets=preview;
      if(!warm) distributed=std::make_unique<SubsystemController>();
      dr=distributed->step(p,s,parts,target,budget,warm?&shifted:nullptr,step>0);
      answer=dr;primal=dr.primal;dual=dr.dual;update_ms=dr.setup_ms;solve_ms=dr.solve_ms;
    }
    const double controller_ms=now()-tick;
    // Diagnostics are outside controller timing, identical full-model checks.
    if(mode=="admm") {p=makeProblem(s,robot.v,target,N,dt,preview);global=centralized(p);}
    double qp_error=violation(global,answer.x);
    Vec tau=100*answer.x.segment(TORQUE,18),force=200*answer.x.segment(FORCE,12);
    Vec acceleration=answer.x.segment(ACC,NV),generalized=Vec::Zero(NV);generalized.tail(18)=tau;
    double dynamics=(s.M*acceleration+s.h-generalized-s.Jfeet.transpose()*force).norm();
    Vec qstart=robot.q,vstart=robot.v;
    double err=(s.tip-target).norm();Eigen::Matrix3d R=Eigen::Quaterniond(robot.q[6],robot.q[3],robot.q[4],robot.q[5]).toRotationMatrix();
    double tilt=std::acos(std::clamp(R(2,2),-1.,1.)),drift=0;
    for(int f=0;f<4;++f) drift=std::max(drift,s.contact_error.segment(3*f,3).norm());
    int contacts=sim.stanceContacts();if(contacts==4) ++allfeet;
    squared_error+=err*err;max_error=std::max(max_error,err);max_tilt=std::max(max_tilt,tilt);max_drift=std::max(max_drift,drift);min_height=std::min(min_height,robot.q[2]);
    if(!robot.q.allFinite() || robot.q[2]<0.35 || tilt>0.5 || qp_error>0.5)
      throw std::runtime_error("Closed-loop validity gate failed at t="+std::to_string(step*dt)+" qp="+std::to_string(qp_error));
    int clips=0;
    for(int j=0;j<substeps;++j) {
      sim.read(robot);double elapsed=(j+1)*0.001;
      Vec qref=qstart.tail(18)+elapsed*vstart.tail(18)+0.5*elapsed*elapsed*acceleration.tail(18);
      Vec vref=vstart.tail(18)+elapsed*acceleration.tail(18);
      Vec command=tau;
      for(int joint=0;joint<18;++joint) {
        const double kp=joint<12?40:20,kd=joint<12?2:1;
        command[joint]+=kp*(qref[joint]-robot.q[7+joint])+kd*(vref[joint]-robot.v[6+joint]);
        // The upstream Z1 URDF specifies 1--2 Nm Coulomb friction and joint
        // damping, neither of which is in Pinocchio's rigid-body RNEA. Apply
        // the same smooth model-based compensation for both MPC variants.
        // Uses measured velocity + planned velocity, never solver contact-force
        // truth or external body forces from MuJoCo.
        int dof=sim.model->jnt_dofadr[sim.joints[joint]];
        command[joint]+=sim.model->dof_damping[dof]*robot.v[6+joint]
          +sim.model->dof_frictionloss[dof]*std::tanh(vref[joint]/0.005);
      }
      clips+=sim.apply(command,s.torque_limit);
    }
    previous=answer;previous_q=qstart;timings.push_back(controller_ms);total_clips+=clips;limited+=!answer.converged;
    log<<step*dt;
    for(int j=0;j<3;++j) log<<','<<target[j];for(int j=0;j<3;++j) log<<','<<s.tip[j];
    log<<','<<err<<','<<qstart[2]<<','<<tilt<<','<<drift<<','<<contacts<<','<<model_ms<<','<<update_ms<<','<<solve_ms<<','<<controller_ms<<','<<answer.iterations<<','<<answer.converged<<','<<primal<<','<<dual<<','<<qp_error<<','<<dynamics<<','<<clips;
    for(int j=0;j<25;++j) log<<','<<qstart[j];log<<'\n';
    if(step%25==0) {log.flush();std::cout<<name<<" t="<<step*dt<<" error_mm="<<1000*err<<" height="<<qstart[2]<<" contacts="<<contacts<<" ms="<<controller_ms<<" qp="<<qp_error<<std::endl;}
  }
  std::sort(timings.begin(),timings.end());double mean=0;for(double x:timings) mean+=x/timings.size();
  if(max_error>0.005 || max_drift>0.005 || max_tilt>0.05 || double(allfeet)/steps<0.99 || total_clips)
    throw std::runtime_error("Tracking/stance/actuator quality gate failed; CSV retained for diagnosis");
  auto quantile=[&](double probability){return timings[std::min(timings.size()-1,size_t(probability*(timings.size()-1)))];};
  std::ofstream report(directory+"/"+name+".json");report<<std::setprecision(12);
  report<<"{\n\"mode\":\""<<mode<<"\",\n\"duration_s\":"<<duration<<",\n\"warm_start\":"<<(warm?"true":"false")<<",\n\"outer_cap\":"<<budget.outer<<",\n\"inner_cap\":"<<budget.inner<<",\n\"central_cap\":"<<central_cap<<",\n\"threads\":"<<(mode=="admm"?budget.threads:1)<<",\n\"bootstrap_ms\":"<<boot<<",\n\"tracking_rmse_mm\":"<<1000*std::sqrt(squared_error/steps)<<",\n\"tracking_max_mm\":"<<1000*max_error<<",\n\"min_base_height_m\":"<<min_height<<",\n\"max_tilt_rad\":"<<max_tilt<<",\n\"max_foot_drift_m\":"<<max_drift<<",\n\"all_feet_contact_fraction\":"<<double(allfeet)/steps<<",\n\"torque_clip_count\":"<<total_clips<<",\n\"not_converged_cycles\":"<<limited<<",\n\"controller_mean_ms\":"<<mean<<",\n\"controller_p50_ms\":"<<quantile(0.5)<<",\n\"controller_p95_ms\":"<<quantile(0.95)<<",\n\"controller_max_ms\":"<<timings.back()<<"\n}\n";
  std::cout<<name<<" COMPLETE rmse_mm="<<1000*std::sqrt(squared_error/steps)<<" mean_ms="<<mean<<" p95_ms="<<quantile(0.95)<<std::endl;
  return 0;
 } catch(const std::exception& e) {std::cerr<<"ERROR: "<<e.what()<<std::endl;return 1;}
}
