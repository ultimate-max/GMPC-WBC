#include "robot.hpp"
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
