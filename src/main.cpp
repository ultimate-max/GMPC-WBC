#include "mpc.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <omp.h>
#include <cstdlib>
int main(int argc,char** argv) {
 try {
  if(argc<2) throw std::runtime_error("Usage: b2z1_demo PROJECT_ROOT [--verify|--benchmark]");
  const std::string root=argv[1],mode=(argc>2?argv[2]:"--verify");
  Robot robot(root);robot.verify();
  if(mode!="--verify" && mode!="--benchmark" && mode!="--subsystems") throw std::runtime_error("Unknown mode");
  auto s=robot.snapshot();
  Eigen::Vector3d target=s.tip+Eigen::Vector3d(0.025,0.015,0.01);
  Problem p=makeProblem(s,robot.v,target);
  double t=now();QP global=centralized(p);double assembly=now()-t;
  Solver solver(global);Solution central=solver.solve();
  const double obj=objective(global,central.x),cv=violation(global,central.x);
  std::filesystem::create_directories(root+"/results");
  const std::string label=mode=="--benchmark"?"benchmark":(mode=="--subsystems"?"subsystem_verification":"verification");
  std::ofstream csv(root+"/results/"+label+".csv");
  csv<<"method,threads,assembly_ms,setup_ms,solve_ms,iterations,objective,constraint_violation,consensus_primal,consensus_dual,first_accel_error,converged,rho\n";
  csv<<std::setprecision(12)<<"central_osqp,1,"<<assembly<<','<<central.setup_ms<<','<<central.solve_ms<<','<<central.iterations<<','<<obj<<','<<cv<<",0,0,0,1,0\n";
  std::cout<<std::setprecision(8)<<"Central: objective="<<obj<<" violation="<<cv<<" setup_ms="<<central.setup_ms<<" solve_ms="<<central.solve_ms<<'\n';
  if(cv>1e-5) throw std::runtime_error("Central constraint violation too large");
  std::vector<int> thread_counts=mode=="--benchmark"?std::vector<int>{1,4}:std::vector<int>{4};
  if(mode=="--subsystems") thread_counts.clear();
  for(int threads:thread_counts) {
    const double rho=std::getenv("MPC_RHO")?std::stod(std::getenv("MPC_RHO")):100.0;
    auto distributed=consensus(p,threads,10000,2e-6,rho,root+"/results/admm_trace_"+std::to_string(threads)+".csv");
    double dv=violation(global,distributed.x),value=objective(global,distributed.x);
    double first=(distributed.x.segment(ACC,NV)-central.x.segment(ACC,NV)).norm();
    csv<<"temporal_consensus,"<<threads<<",0,"<<distributed.setup_ms<<','<<distributed.solve_ms<<','<<distributed.iterations<<','<<value<<','<<dv<<','<<distributed.primal<<','<<distributed.dual<<','<<first<<','<<distributed.converged<<','<<rho<<'\n';csv.flush();
    std::cout<<"Consensus "<<threads<<" threads: converged="<<distributed.converged<<" iterations="<<distributed.iterations<<" objective="<<value<<" violation="<<dv<<" primal="<<distributed.primal<<" dual="<<distributed.dual<<" first_accel_error="<<first<<" solve_ms="<<distributed.solve_ms<<'\n';
    if(!distributed.converged || dv>1e-5 || std::abs(value-obj)>2e-3 || first>0.02)
      throw std::runtime_error("Consensus accuracy gate failed; inspect residual trace");
  }
  auto parts=robot.subsystems(root);
  std::vector<int> subsystem_threads=mode=="--benchmark"?std::vector<int>{1,3}:std::vector<int>{3};
  for(int threads:subsystem_threads) {
    const double rho=std::getenv("SUBSYSTEM_RHO")?std::stod(std::getenv("SUBSYSTEM_RHO")):1.0;
    auto distributed=subsystemConsensus(p,s,parts,target,threads,rho,root+"/results/subsystem_trace_"+std::to_string(threads)+".csv");
    double dv=violation(global,distributed.x),value=objective(global,distributed.x);
    double first=(distributed.x.segment(ACC,NV)-central.x.segment(ACC,NV)).norm();
    csv<<"physical_subsystems,"<<threads<<",0,"<<distributed.setup_ms<<','<<distributed.solve_ms<<','<<distributed.iterations<<','<<value<<','<<dv<<','<<distributed.primal<<','<<distributed.dual<<','<<first<<','<<distributed.converged<<','<<rho<<'\n';csv.flush();
    std::cout<<"Physical front/rear/arm "<<threads<<" threads: converged="<<distributed.converged<<" iterations="<<distributed.iterations<<" objective="<<value<<" violation="<<dv<<" primal="<<distributed.primal<<" dual="<<distributed.dual<<" first_accel_error="<<first<<" solve_ms="<<distributed.solve_ms<<'\n';
    if(!distributed.converged || dv>1e-5 || std::abs(value-obj)>2e-3 || first>0.02)
      throw std::runtime_error("Physical subsystem accuracy gate failed");
    Vec generalized=Vec::Zero(NV);generalized.tail(18)=100*distributed.x.segment(TORQUE,18);
    double dyn_error=(s.M*distributed.x.segment(ACC,NV)+s.h-generalized-200*s.Jfeet.transpose()*distributed.x.segment(FORCE,12)).norm();
    std::cout<<"Reassembled subsystem first-command dynamics residual="<<dyn_error<<'\n';
    if(dyn_error>0.01) throw std::runtime_error("Physical subsystem command violates full dynamics");
  }
  // Predicted configurations use the SE(3) exponential through Pinocchio integrate.
  std::ofstream prediction(root+"/results/prediction.csv");
  prediction<<std::setprecision(17);
  prediction<<"time,tip_x,tip_y,tip_z,target_error,base_x,base_y,base_z";
  for(int i=0;i<25;++i) prediction<<",q"<<i;
  prediction<<'\n';
  const Vec q0=robot.q;
  for(int k=0;k<=p.horizon;++k) {
    Vec delta=Vec::Zero(NV);
    if(k) delta=central.x.segment((k-1)*LOCAL+NX,NV);
    robot.q=robot.integrate(q0,delta);auto pose=robot.snapshot();
    prediction<<k*p.dt<<','<<pose.tip.transpose().format(Eigen::IOFormat(Eigen::FullPrecision,0,",",","))<<','<<(pose.tip-target).norm()<<','<<robot.q.head(3).transpose().format(Eigen::IOFormat(Eigen::FullPrecision,0,",",","));
    for(int i=0;i<25;++i) prediction<<','<<robot.q[i];
    prediction<<'\n';
  }
  // Independent nonlinear dynamics residual for the first command at the actual state.
  Vec acc=central.x.segment(ACC,NV),tau=100*central.x.segment(TORQUE,18),f=200*central.x.segment(FORCE,12);
  Vec generalized=Vec::Zero(NV);generalized.tail(18)=tau;
  double dynamics_residual=(s.M*acc+s.h-generalized-s.Jfeet.transpose()*f).norm();
  if(dynamics_residual>1e-3) throw std::runtime_error("First-step inverse dynamics residual failed");
  std::cout<<"First-step dynamics residual="<<dynamics_residual<<" N/Nm; results saved in "<<root<<"/results\n";
  std::cout<<"This is a frozen-model MPC solve, not a physics-engine closed-loop experiment.\n";
  return 0;
 } catch(const std::exception& e) {std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
}
