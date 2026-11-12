#pragma once
#include "robot.hpp"
#include "qp.hpp"
// One stage: [s_k(48), s_{k+1}(48), a(24), tau(18), f_world(12)].
constexpr int NV=24, NX=48, LOCAL=150, ACC=96, TORQUE=120, FORCE=138;
struct Problem { std::vector<QP> stages; Vec initial; int horizon; double dt; std::vector<Eigen::Vector3d> targets; };
Problem makeProblem(const Snapshot& s, const Vec& velocity, const Eigen::Vector3d& target, int horizon=8, double dt=0.04,
                    const std::vector<Eigen::Vector3d>& targets={});
QP centralized(const Problem& p);
struct DistributedResult : Solution { double primal=0, dual=0; };
DistributedResult consensus(const Problem& p, int threads, int max_iter, double tolerance,
                            double rho, const std::string& trace="");
DistributedResult subsystemConsensus(const Problem& p,const Snapshot& full,
    const std::vector<SubsystemModel>& models,const Eigen::Vector3d& target,
    int threads,double rho,const std::string& trace);
struct OnlineBudget {
  int outer=10,inner=100,threads=3;
  double rho=1.0,tolerance=1e-5;
};
class SubsystemController {
 public:
  SubsystemController();~SubsystemController();
  DistributedResult step(const Problem& p,const Snapshot& full,
    const std::vector<SubsystemModel>& models,const Eigen::Vector3d& target,
    const OnlineBudget& budget,const Vec* shifted_global=nullptr,bool shift_memory=true);
 private:
  struct Impl;std::unique_ptr<Impl> impl;
};
Vec shiftPlan(Robot& robot,const Vec& old_q,const Vec& plan,int horizon,double dt=0.04);
Vec shiftCentralDual(const Vec& y,const Problem& p);
