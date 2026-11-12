#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "simulation.hpp"
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <optional>
namespace py=pybind11;
#include "online_assembly.hpp"
#include "wbc_native.hpp"
struct Plant {
  Robot robot; Simulation sim; pinocchio::Model model; std::unique_ptr<pinocchio::Data> d;
  bool instrument=false;std::vector<Vec> trace;
  bool fast_reference=false;
  bool reference_hold_gil=false;
  // Separate opt-in channel preserves the established 129-column trace ABI.
  bool instrument_base=false;std::vector<Vec> base_trace;
  Plant(const std::string& root):robot(root),sim(root,robot) {
    pinocchio::urdf::buildModel(root+"/models/b2_z1.urdf",pinocchio::JointModelFreeFlyer(),model);
    d=std::make_unique<pinocchio::Data>(model);
    sim.model->opt.cone=mjCONE_ELLIPTIC;sim.model->opt.iterations=100;sim.model->opt.noslip_iterations=10;
    for(int g=0;g<sim.model->ngeom;++g) {
      sim.model->geom_solref[2*g]=.005;sim.model->geom_solref[2*g+1]=1;
      sim.model->geom_solimp[5*g]=.99;sim.model->geom_solimp[5*g+1]=.999;sim.model->geom_solimp[5*g+2]=.0001;
      sim.model->geom_friction[3*g]=.6;
    }
    mj_forward(sim.model,sim.data);sim.read(robot);sim.verifyGroundTruth(robot,robot.snapshot());
  }
  py::dict evaluate(const Vec& q,const Vec& v) {
    robot.q=q;robot.v=v;auto s=robot.snapshot();
    pinocchio::computeJointJacobians(model,*d,q);
    pinocchio::forwardKinematics(model,*d,q,v,Vec::Zero(24));pinocchio::updateFramePlacements(model,*d);
    auto id=model.getFrameId("tool_tip");Mat J=Mat::Zero(6,24);
    pinocchio::getFrameJacobian(model,*d,id,pinocchio::LOCAL,J);
    py::dict out;
    out["q"]=q;out["v"]=v;out["M"]=s.M;out["h"]=s.h;out["Jc"]=s.Jfeet;
    out["cdotv"]=s.contact_drift;out["ce"]=s.contact_error;out["feet"]=s.feet;
    out["J"]=J;out["jdotv"]=Vec(pinocchio::getFrameAcceleration(model,*d,id,pinocchio::LOCAL).toVector());
    out["pose"]=Mat(d->oMf[id].toHomogeneousMatrix());out["tau_max"]=s.torque_limit;
    out["v_max"]=s.velocity_limit;out["q_min"]=Vec(model.lowerPositionLimit.tail(18));
    out["q_max"]=Vec(model.upperPositionLimit.tail(18));
    Mat Jf=Mat::Zero(12,24);Vec fdot=Vec::Zero(12);const char* feet[]={"FL_foot","FR_foot","RL_foot","RR_foot"};
    for(int i=0;i<4;++i){auto fi=model.getFrameId(feet[i]);Mat j=Mat::Zero(6,24);pinocchio::getFrameJacobian(model,*d,fi,pinocchio::LOCAL_WORLD_ALIGNED,j);Jf.middleRows(3*i,3)=j.topRows(3);fdot.segment(3*i,3)=pinocchio::getFrameClassicalAcceleration(model,*d,fi,pinocchio::LOCAL_WORLD_ALIGNED).linear();}
    out["Jf"]=Jf;out["fdot"]=fdot;return out;
  }
  py::dict truth() {sim.read(robot);auto r=evaluate(robot.q,robot.v);r["contacts"]=sim.stanceContacts();return r;}
  void set_timestep(double dt) {
    if(!std::isfinite(dt) || dt<=0 || dt>.001)
      throw std::runtime_error("Simulation timestep must be in (0, 1 ms]");
    if(sim.data->time!=0.) throw std::runtime_error("Set timestep before stepping");
    sim.model->opt.timestep=dt;
  }
  py::tuple contact_kinematics(const Vec& q,const Vec& v) {
    Mat J=Mat::Zero(12,24);Vec drift=Vec::Zero(12);
    {
      std::optional<py::gil_scoped_release> release;
      if(!reference_hold_gil) release.emplace();
      pinocchio::computeJointJacobians(model,*d,q);
      pinocchio::forwardKinematics(model,*d,q,v,Vec::Zero(24));pinocchio::updateFramePlacements(model,*d);
      const std::vector<std::string> names={"FL_foot","FR_foot","RL_foot","RR_foot"};
      Eigen::Matrix3d rx;rx<<0,.032,0,-.032,0,0,0,0,0;
      for(int i=0;i<4;++i){
        const auto id=model.getFrameId(names[i]);Mat j=Mat::Zero(6,24);
        pinocchio::getFrameJacobian(model,*d,id,pinocchio::LOCAL_WORLD_ALIGNED,j);
        J.middleRows(3*i,3)=j.topRows(3)-rx*j.bottomRows(3);
        auto a=pinocchio::getFrameClassicalAcceleration(model,*d,id,pinocchio::LOCAL_WORLD_ALIGNED);
        drift.segment(3*i,3)=a.linear()+a.angular().cross(Eigen::Vector3d(0,0,-.032));
      }
    }
    return py::make_tuple(J,drift);
  }
  py::tuple segment_state(const Vec& q0,const Mat& c,double dt,double t,bool acceleration) {
    Vec q,V,a=Vec::Zero(24);
    {
      std::optional<py::gil_scoped_release> release;
      if(!reference_hold_gil) release.emplace();
      double u=std::clamp(t/dt,0.,1.);Vec x=Vec::Zero(24),dx=x,ddx=x;
      for(int k=0;k<6;++k)x+=std::pow(u,k)*c.row(k).transpose();
      for(int k=1;k<6;++k)dx+=k*std::pow(u,k-1)/dt*c.row(k).transpose();
      if(acceleration)for(int k=2;k<6;++k)ddx+=k*(k-1)*std::pow(u,k-2)/(dt*dt)*c.row(k).transpose();
      q=robot.integrate(q0,x);V=dx;a=ddx;
      Eigen::Matrix<double,6,1> xx=x.head<6>(),vv=dx.head<6>();
      Eigen::Matrix<double,6,6> L,Lp,Lm;
      pinocchio::Jlog6(pinocchio::exp6(xx),L);auto J=L.inverse().eval();V.head<6>()=J*vv;
      if(acceleration){
        const double eps=1e-5;
        pinocchio::Jlog6(pinocchio::exp6((xx+eps*vv).eval()),Lp);
        pinocchio::Jlog6(pinocchio::exp6((xx-eps*vv).eval()),Lm);
        a.head<6>()=J*ddx.head<6>()+(Lp.inverse()-Lm.inverse())/(2*eps)*vv;
      }
    }
    return py::make_tuple(q,V,a);
  }
  void set_base_wrench(const Eigen::Vector3d& force,const Eigen::Vector3d& torque) {
    if(!force.allFinite() || !torque.allFinite()) throw std::runtime_error("Non-finite external wrench");
    int body=-1;
    for(int j=0;j<sim.model->njnt;++j) if(sim.model->jnt_type[j]==mjJNT_FREE) {
      if(body>=0) throw std::runtime_error("More than one floating base");
      body=sim.model->jnt_bodyid[j];
    }
    if(body<0) throw std::runtime_error("No floating base");
    // MuJoCo applied Cartesian force and torque, world axes, at body COM.
    Eigen::Map<Eigen::Matrix<double,6,1>> wrench(sim.data->xfrc_applied+6*body);
    wrench.head<3>()=force;wrench.tail<3>()=torque;
  }
  void record(const Vec& cmd,const Vec& pd,const Vec& friction) {
    if(instrument_base) {
      // Pre-step truth at the same timestamp as the contact/actuator channel.
      // [time, xyz, quaternion_xyzw, body_twist, world_point_velocity_and_omega]
      Vec base=Vec::Zero(20);base[0]=sim.data->time;
      base.segment(1,7)=robot.q.head(7);base.segment(8,6)=robot.v.head(6);
      Eigen::Matrix3d R=Eigen::Quaterniond(robot.q[6],robot.q[3],robot.q[4],robot.q[5]).toRotationMatrix();
      base.segment(14,3)=R*robot.v.head(3);base.segment(17,3)=R*robot.v.segment(3,3);
      base_trace.push_back(base);
    }
    if(!instrument)return;
    Vec row=Vec::Zero(129);row[0]=sim.data->time;
    std::vector<Eigen::Vector3d> points(4);
    for(int f=0;f<4;++f)points[f]=Eigen::Map<Eigen::Vector3d>(sim.data->xpos+3*sim.feet[f])+Eigen::Vector3d(0,0,-.032);
    for(int c=0;c<sim.data->ncon;++c) {
      const auto& ct=sim.data->contact[c];int b1=sim.model->geom_bodyid[ct.geom[0]],b2=sim.model->geom_bodyid[ct.geom[1]];
      for(int f=0;f<4;++f)if((b1==0&&b2==sim.feet[f])||(b2==0&&b1==sim.feet[f])) {
        mjtNum wrench[6];mj_contactForce(sim.model,sim.data,c,wrench);
        row[1+f]=1;row[5+f]+=wrench[0];points[f]=Eigen::Map<const Eigen::Vector3d>(ct.pos);
      }
    }
    for(int f=0;f<4;++f) {
      Eigen::Vector3d center=Eigen::Map<Eigen::Vector3d>(sim.data->xpos+3*sim.feet[f]);
      Eigen::Matrix<double,3,24,Eigen::RowMajor> jc,jr,jcenter;
      mj_jac(sim.model,sim.data,jc.data(),jr.data(),points[f].data(),sim.feet[f]);
      mj_jac(sim.model,sim.data,jcenter.data(),nullptr,center.data(),sim.feet[f]);
      Vec vel=Eigen::Map<Vec>(sim.data->qvel,24);
      row.segment(9+3*f,3)=center;row.segment(21+3*f,3)=jc*vel;
      row.segment(33+3*f,3)=jcenter*vel;row.segment(45+3*f,3)=jr*vel;
    }
    row.segment(57,18)=cmd;row.segment(75,18)=pd;row.segment(93,18)=friction;
    row.segment(111,18)=robot.v.tail(18);trace.push_back(row);
  }
  Mat take_trace() {Mat result(trace.size(),129);for(size_t i=0;i<trace.size();++i)result.row(i)=trace[i].transpose();trace.clear();return result;}
  Mat take_base_trace() {Mat result(base_trace.size(),20);for(size_t i=0;i<base_trace.size();++i)result.row(i)=base_trace[i].transpose();base_trace.clear();return result;}
  int advance(const Vec& tau,const Vec& acc,int ticks) {
    sim.read(robot);Vec q0=robot.q,v0=robot.v;int clips=0;
    for(int i=0;i<ticks;++i) {
      sim.read(robot);double t=(i+1)*sim.model->opt.timestep;
      Vec qr=q0.tail(18)+t*v0.tail(18)+.5*t*t*acc.tail(18),vr=v0.tail(18)+t*acc.tail(18),cmd=tau,pd=Vec::Zero(18),friction=Vec::Zero(18);
      for(int j=0;j<18;++j) {
        int k=sim.model->jnt_dofadr[sim.joints[j]];
        pd[j]=(j<12?40:20)*(qr[j]-robot.q[7+j])+(j<12?2:1)*(vr[j]-robot.v[6+j]);cmd[j]+=pd[j];
        friction[j]=sim.model->dof_damping[k]*robot.v[6+j]+sim.model->dof_frictionloss[k]*std::tanh(vr[j]/.005);cmd[j]+=friction[j];
      }
      record(cmd,pd,friction);clips+=sim.apply(cmd,model.effortLimit.tail(18));
    }return clips;
  }
  int advance_reference(const Vec& tau,const Vec& qref,const Vec& vref,int ticks) {
    // WBC (20) supplies configuration and velocity targets explicitly. Reuse
    // the same actuator gains, friction compensation and physical torque limit.
    int clips=0;
    for(int i=0;i<ticks;++i) {
      sim.read(robot);double t=(i+1)*sim.model->opt.timestep;Vec cmd=tau,pd=Vec::Zero(18),friction=Vec::Zero(18);
      for(int j=0;j<18;++j) {
        int k=sim.model->jnt_dofadr[sim.joints[j]];
        double qr=qref[7+j]+t*vref[6+j],vr=vref[6+j];
        pd[j]=(j<12?40:20)*(qr-robot.q[7+j])+(j<12?2:1)*(vr-robot.v[6+j]);
        friction[j]=sim.model->dof_damping[k]*robot.v[6+j]+sim.model->dof_frictionloss[k]*std::tanh(vr/.005);
        cmd[j]+=pd[j]+friction[j];
      }
      record(cmd,pd,friction);
      clips+=sim.apply(cmd,model.effortLimit.tail(18));
    }return clips;
  }
};
#include "runtime_native.hpp"
PYBIND11_MODULE(native,m) {
  py::class_<ControlRuntime>(m,"ControlRuntime")
    .def(py::init<const std::string&,const std::string&,const std::string&,const std::string&,const Vec&,double,bool,bool>())
    .def("publish",&ControlRuntime::publish)
    .def("set_gait",&ControlRuntime::setGait)
    .def("observation",&ControlRuntime::observation)
    .def("start",&ControlRuntime::start)
    .def("running",&ControlRuntime::running)
    .def("finish",&ControlRuntime::finish,py::call_guard<py::gil_scoped_release>())
    .def("results",&ControlRuntime::results);
  py::class_<NativeWbc>(m,"WbcKernel")
    .def(py::init<const std::string&,const std::string&,const std::string&,const std::string&,Vec,Vec,double,bool>())
    .def("set_previous",&NativeWbc::setPrevious)
    .def("control",[](NativeWbc& w,py::dict ss,py::dict rr){
      WState s;WReference r;
      s.q=ss["q"].cast<Vec>();s.v=ss["v"].cast<Vec>();s.M=ss["M"].cast<Mat>();s.h=ss["h"].cast<Vec>();
      s.Jc=ss["Jc"].cast<Mat>();s.cdot=ss["cdotv"].cast<Vec>();s.ce=ss["ce"].cast<Vec>();s.J=ss["J"].cast<Mat>();s.jdot=ss["jdotv"].cast<Vec>();s.pose=ss["pose"].cast<Mat>();
      s.qmin=ss["q_min"].cast<Vec>();s.qmax=ss["q_max"].cast<Vec>();s.vmax=ss["v_max"].cast<Vec>();s.taumax=ss["tau_max"].cast<Vec>();
      if(ss.contains("Jf")){s.Jf=ss["Jf"].cast<Mat>();s.fdot=ss["fdot"].cast<Vec>();s.feet=ss["feet"].cast<Vec>();}
      r.q=rr["q"].cast<Vec>();r.v=rr["v"].cast<Vec>();r.a=rr["a"].cast<Vec>();r.tau=rr["tau"].cast<Vec>();r.force=rr["force"].cast<Vec>();
      r.bp=rr["base_pose"].cast<Mat>();r.bv=rr["base_twist"].cast<Vec>();r.ba=rr["base_acc"].cast<Vec>();r.ep=rr["ee_pose"].cast<Mat>();r.ev=rr["ee_twist"].cast<Vec>();r.ea=rr["ee_acc"].cast<Vec>();
      if(rr.contains("mode")){r.mode=rr["mode"].cast<Vec>();r.fp=rr["fp"].cast<Vec>();r.fv=rr["fv"].cast<Vec>();r.fa=rr["fa"].cast<Vec>();}if(rr.contains("fnmin")){r.fnmin=rr["fnmin"].cast<Vec>();r.fnmax=rr["fnmax"].cast<Vec>();}
      WResult o;{py::gil_scoped_release release;o=w.control(s,r);}
      py::dict d;d["tau"]=o.tau;d["acc"]=o.acc;d["force"]=o.force;d["wrench"]=o.wrench;
      d["bound"]=o.bound;d["lock"]=o.lock;d["kkt"]=o.kkt;d["dynamics"]=o.dynamics;d["contact"]=o.contact;d["interface"]=o.interface;
      d["assembly_ms"]=o.assembly_ms;d["solve_ms"]=o.solve_ms;d["total_ms"]=o.total_ms;
      d["G"]=o.G;d["lo"]=o.lo;d["hi"]=o.hi;for(int i=0;i<3;++i){d[py::str("C"+std::to_string(i))]=o.tasks[i].C;d[py::str("d"+std::to_string(i))]=o.tasks[i].d;}return d;
    });
  m.def("assemble_online_mpc",&assemble_online_mpc);
  py::class_<Plant>(m,"Plant").def(py::init<const std::string&>()).def("truth",&Plant::truth)
    .def("set_timestep",&Plant::set_timestep)
    .def_readwrite("fast_reference",&Plant::fast_reference)
    .def_readwrite("reference_hold_gil",&Plant::reference_hold_gil)
    .def("contact_kinematics",&Plant::contact_kinematics).def("segment_state",&Plant::segment_state)
    .def("set_base_wrench",&Plant::set_base_wrench)
    .def("evaluate",&Plant::evaluate).def("advance",&Plant::advance,py::call_guard<py::gil_scoped_release>()).def("advance_reference",&Plant::advance_reference,py::call_guard<py::gil_scoped_release>())
    .def_readwrite("instrument",&Plant::instrument).def("take_trace",&Plant::take_trace)
    .def_readwrite("instrument_base",&Plant::instrument_base).def("take_base_trace",&Plant::take_base_trace)
    .def("integrate",[](Plant& p,const Vec&q,const Vec&v){return p.robot.integrate(q,v);})
    .def("difference",[](Plant& p,const Vec&a,const Vec&b){return p.robot.difference(a,b);});
}
