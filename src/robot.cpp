#include "robot.hpp"
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>
namespace {
// Ground contact lies 32 mm below the sphere centre. Use the SAME point for
// force transmission and no-slip velocity; a centre Jacobian omits the moment
// of tangential ground forces about the foot centre.
Mat contactJacobian(const Mat& j) {
  Eigen::Matrix3d rx;rx<<0,0.032,0,-0.032,0,0,0,0,0;
  return j.topRows(3)-rx*j.bottomRows(3);
}
}
struct Robot::Impl {
  pinocchio::Model model;
  std::unique_ptr<pinocchio::Data> data;
  std::vector<pinocchio::Model> parts;
  std::vector<std::unique_ptr<pinocchio::Data>> part_data;
  Vec foot_anchor;
};
Robot::Robot(const std::string& root): impl(std::make_unique<Impl>()) {
  pinocchio::urdf::buildModel(root+"/models/b2_z1.urdf", pinocchio::JointModelFreeFlyer(), impl->model);
  impl->data = std::make_unique<pinocchio::Data>(impl->model);
  const auto& m=impl->model;
  if(m.nq!=25 || m.nv!=24) throw std::runtime_error("Unexpected model dimensions");
  q=Vec::Zero(m.nq); v=Vec::Zero(m.nv);
  std::ifstream in(root+"/models/q_home.txt");
  for(int i=0;i<m.nq;++i) if(!(in>>q[i])) throw std::runtime_error("Missing/invalid q_home");
  home=q;
}
Robot::~Robot()=default;
Vec Robot::integrate(const Vec& pos, const Vec& delta) { return pinocchio::integrate(impl->model,pos,delta); }
Vec Robot::difference(const Vec& from, const Vec& to) { return pinocchio::difference(impl->model,from,to); }
Vec Robot::rebaseVelocity(const Vec& old_q,const Vec& delta,const Vec& chart_velocity) {
  Mat ji=Mat::Zero(24,24),jd=Mat::Zero(24,24);
  Vec predicted=integrate(old_q,delta);
  pinocchio::dIntegrate(impl->model,old_q,delta,ji,pinocchio::ARG1);
  pinocchio::dDifference(impl->model,q,predicted,jd,pinocchio::ARG1);
  return jd*ji*chart_velocity;
}
std::vector<std::string> Robot::jointNames() const {
  return {impl->model.names.begin()+2,impl->model.names.end()};
}
Snapshot Robot::snapshot() {
  auto& m=impl->model; auto& d=*impl->data;
  Snapshot s;
  pinocchio::crba(m,d,q);
  s.M=d.M.selfadjointView<Eigen::Upper>();
  s.h=pinocchio::nonLinearEffects(m,d,q,v);
  pinocchio::computeJointJacobians(m,d,q);
  pinocchio::forwardKinematics(m,d,q,v,Vec::Zero(m.nv));
  pinocchio::updateFramePlacements(m,d);
  s.Jfeet=Mat::Zero(12,m.nv);
  s.feet=Vec::Zero(12);s.contact_drift=Vec::Zero(12);
  const std::vector<std::string> names={"FL_foot","FR_foot","RL_foot","RR_foot"};
  for(int i=0;i<4;++i) {
    Mat j=Mat::Zero(6,m.nv);
    pinocchio::getFrameJacobian(m,d,m.getFrameId(names[i]),pinocchio::LOCAL_WORLD_ALIGNED,j);
    s.Jfeet.middleRows(3*i,3)=contactJacobian(j);
    const auto id=m.getFrameId(names[i]);
    s.feet.segment(3*i,3)=d.oMf[id].translation();
    const auto acceleration=pinocchio::getFrameClassicalAcceleration(m,d,id,pinocchio::LOCAL_WORLD_ALIGNED);
    s.contact_drift.segment(3*i,3)=acceleration.linear()+acceleration.angular().cross(Eigen::Vector3d(0,0,-0.032));
  }
  Mat j=Mat::Zero(6,m.nv); auto tip=m.getFrameId("tool_tip");
  pinocchio::getFrameJacobian(m,d,tip,pinocchio::LOCAL_WORLD_ALIGNED,j);
  s.Jtip=j.topRows(3); s.tip=d.oMf[tip].translation();
  s.torque_limit=m.effortLimit.tail(18); s.velocity_limit=m.velocityLimit.tail(18);
  s.qmin_delta=m.lowerPositionLimit.tail(18)-q.tail(18);
  s.qmax_delta=m.upperPositionLimit.tail(18)-q.tail(18);
  s.home_delta=difference(q,home);
  if(impl->foot_anchor.size()==0) impl->foot_anchor=s.feet;
  s.contact_error=s.feet-impl->foot_anchor;
  return s;
}
void Robot::verify() {
  auto s=snapshot();
  if(Eigen::LLT<Mat>(s.M).info()!=Eigen::Success) throw std::runtime_error("Mass matrix not SPD");
  Vec delta=Vec::LinSpaced(24,-0.03,0.03);
  if((difference(q,integrate(q,delta))-delta).norm()>1e-10) throw std::runtime_error("Manifold round-trip failed");
  Vec a=Vec::LinSpaced(24,-0.2,0.2);
  Vec tau=pinocchio::rnea(impl->model,*impl->data,q,v,a);
  if((tau-s.M*a-s.h).norm()>1e-8) throw std::runtime_error("RNEA != M a + h");
  const Vec saved=q; double eps=1e-7;
  q=integrate(saved,eps*delta); auto next=snapshot(); q=saved;
  if(((next.tip-s.tip)/eps-s.Jtip*delta).norm()>1e-6) throw std::runtime_error("Tip Jacobian finite difference failed");
  std::cout<<"Model checks passed: nq=25 nv=24, manifold/RNEA/Jacobian consistency\n";
}
std::vector<SubsystemModel> Robot::subsystems(const std::string& root,bool verbose) {
  std::vector<SubsystemModel> result;
  Mat reconstructed=Mat::Zero(24,24);Vec bias=Vec::Zero(24);
  const std::vector<std::string> names={"front","rear","arm"};
  if(impl->parts.empty()) for(auto& name:names) {
    pinocchio::Model m;pinocchio::urdf::buildModel(root+"/models/"+name+".urdf",pinocchio::JointModelFreeFlyer(),m);
    impl->parts.push_back(m);impl->part_data.push_back(std::make_unique<pinocchio::Data>(m));
  }
  for(int part=0;part<3;++part) {
    const auto& name=names[part];auto& model=impl->parts[part];auto& data=*impl->part_data[part];
    Vec pos=pinocchio::neutral(model),vel=Vec::Zero(model.nv);
    pos.head(7)=q.head(7);
    SubsystemModel sub;sub.name=name;sub.coordinates={0,1,2,3,4,5};
    for(pinocchio::JointIndex j=2;j<model.njoints;++j) {
      auto whole=impl->model.getJointId(model.names[j]);
      pos[model.joints[j].idx_q()]=q[impl->model.joints[whole].idx_q()];
      sub.coordinates.push_back(impl->model.joints[whole].idx_v());
    }
    if(model.nv!=12) throw std::runtime_error("Expected six-joint subsystem");
    Mat R=Mat::Zero(12,24);for(int i=0;i<12;++i) R(i,sub.coordinates[i])=1;
    vel=R*v;
    pinocchio::crba(model,data,pos);sub.M=data.M.selfadjointView<Eigen::Upper>();
    sub.h=pinocchio::nonLinearEffects(model,data,pos,vel);
    if(Eigen::LLT<Mat>(sub.M).info()!=Eigen::Success) throw std::runtime_error("Subsystem mass matrix not SPD");
    pinocchio::computeJointJacobians(model,data,pos);pinocchio::updateFramePlacements(model,data);
    sub.Jfeet=Mat::Zero(name=="arm"?0:6,12);
    const std::vector<std::string> legs=name=="front"?std::vector<std::string>{"FL_foot","FR_foot"}:std::vector<std::string>{"RL_foot","RR_foot"};
    if(name!="arm") for(int i=0;i<2;++i) {
      Mat j=Mat::Zero(6,12);pinocchio::getFrameJacobian(model,data,model.getFrameId(legs[i]),pinocchio::LOCAL_WORLD_ALIGNED,j);
      sub.Jfeet.middleRows(3*i,3)=contactJacobian(j);
    }
    reconstructed+=R.transpose()*sub.M*R;bias+=R.transpose()*sub.h;
    result.push_back(sub);
  }
  auto full=snapshot();double em=(full.M-reconstructed).norm(),eh=(full.h-bias).norm();
  if(verbose) std::cout<<"Subsystem reconstruction: mass_error="<<em<<" bias_error="<<eh<<'\n';
  if(em>1e-9 || eh>1e-8) throw std::runtime_error("Physical decomposition does not reconstruct full dynamics");
  return result;
}
