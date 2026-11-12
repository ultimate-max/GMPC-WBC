#pragma once
// Included after Plant. All functions called by the control thread use C++
// objects only; Python only publishes immutable policies and reads snapshots.
#include <thread>
#include <mutex>
#include <atomic>
#include <sched.h>
struct RuntimePolicy {long sequence;double t,dt;Mat q0,c,tau,force;};
struct RuntimeObservation {double time=0;long sequence=0;Vec q,v,tau,acc;int contacts=4;};
struct RuntimeRow {Vec values,q,v,tau,acc,rq,rv;Mat ep;};
class ControlRuntime {
  Plant plant;NativeWbc wbc;double dt;bool push;
  std::thread thread;std::atomic<bool> live{false},stop{false};
  std::mutex obs_mutex,policy_mutex;
  std::shared_ptr<const RuntimeObservation> latest;
  std::shared_ptr<const RuntimePolicy> pending,active;
  std::vector<RuntimeRow> rows;std::string failure;
  long rejected=0,overwritten=0,accepted=0;
  Vec legs;double legtime=0;double elapsed=0;Mat gait;
  static pinocchio::SE3 pose(const Vec&q){return pinocchio::SE3(Eigen::Quaterniond(q[6],q[3],q[4],q[5]).toRotationMatrix(),q.head<3>());}
  WState evaluate(const Vec&q,const Vec&v){
    plant.robot.q=q;plant.robot.v=v;auto s=plant.robot.snapshot();auto&m=plant.model;auto&d=*plant.d;
    pinocchio::computeJointJacobians(m,d,q);pinocchio::forwardKinematics(m,d,q,v,Vec::Zero(24));pinocchio::updateFramePlacements(m,d);
    auto id=m.getFrameId("tool_tip");Mat J=Mat::Zero(6,24);pinocchio::getFrameJacobian(m,d,id,pinocchio::LOCAL,J);
    WState state={q,v,s.h,s.contact_drift,s.contact_error,pinocchio::getFrameAcceleration(m,d,id,pinocchio::LOCAL).toVector(),m.lowerPositionLimit.tail(18),m.upperPositionLimit.tail(18),s.velocity_limit,s.torque_limit,s.M,s.Jfeet,J,d.oMf[id].toHomogeneousMatrix()};
    state.Jf=Mat::Zero(12,24);state.fdot=Vec::Zero(12);state.feet=Vec::Zero(12);
    const char* names[]={"FL_foot","FR_foot","RL_foot","RR_foot"};
    for(int i=0;i<4;++i){auto fi=m.getFrameId(names[i]);Mat j=Mat::Zero(6,24);pinocchio::getFrameJacobian(m,d,fi,pinocchio::LOCAL_WORLD_ALIGNED,j);state.Jf.middleRows(3*i,3)=j.topRows(3);state.fdot.segment(3*i,3)=pinocchio::getFrameClassicalAcceleration(m,d,fi,pinocchio::LOCAL_WORLD_ALIGNED).linear();state.feet.segment(3*i,3)=d.oMf[fi].translation();}return state;
  }
  std::pair<Mat,Vec> contact(const Vec&q,const Vec&v){
    auto&m=plant.model;auto&d=*plant.d;pinocchio::computeJointJacobians(m,d,q);pinocchio::forwardKinematics(m,d,q,v,Vec::Zero(24));pinocchio::updateFramePlacements(m,d);
    Mat J=Mat::Zero(12,24);Vec drift=Vec::Zero(12);const char* names[]={"FL_foot","FR_foot","RL_foot","RR_foot"};Eigen::Matrix3d rx;rx<<0,.032,0,-.032,0,0,0,0,0;
    for(int i=0;i<4;++i){auto id=m.getFrameId(names[i]);Mat j=Mat::Zero(6,24);pinocchio::getFrameJacobian(m,d,id,pinocchio::LOCAL_WORLD_ALIGNED,j);J.middleRows(3*i,3)=j.topRows(3)-rx*j.bottomRows(3);auto a=pinocchio::getFrameClassicalAcceleration(m,d,id,pinocchio::LOCAL_WORLD_ALIGNED);drift.segment(3*i,3)=a.linear()+a.angular().cross(Eigen::Vector3d(0,0,-.032));}return {J,drift};
  }
  std::array<Vec,3> raw(const RuntimePolicy&p,double t,bool acceleration=true){
    using V6=Eigen::Matrix<double,6,1>;using M6=Eigen::Matrix<double,6,6>;
    double age=std::clamp(t-p.t,0.,10*p.dt);int idx=std::min(int(age/p.dt),9);double u=std::clamp((age-idx*p.dt)/p.dt,0.,1.);auto c=p.c.middleRows(6*idx,6);
    Vec x=Vec::Zero(24),dx=x,dd=x;for(int k=0;k<6;++k)x+=std::pow(u,k)*c.row(k).transpose();for(int k=1;k<6;++k)dx+=k*std::pow(u,k-1)/p.dt*c.row(k).transpose();if(acceleration)for(int k=2;k<6;++k)dd+=k*(k-1)*std::pow(u,k-2)/(p.dt*p.dt)*c.row(k).transpose();
    Vec q=plant.robot.integrate(p.q0.row(idx).transpose(),x),v=dx,a=dd;M6 L,Lp,Lm;pinocchio::Jlog6(pinocchio::exp6(V6(x.head(6))),L);M6 J=L.inverse();v.head(6)=J*dx.head(6);
    if(acceleration){pinocchio::Jlog6(pinocchio::exp6(V6(x.head(6)+1e-5*dx.head(6))),Lp);pinocchio::Jlog6(pinocchio::exp6(V6(x.head(6)-1e-5*dx.head(6))),Lm);a.head(6)=J*dd.head(6)+(Lp.inverse()-Lm.inverse())/(2e-5)*dx.head(6);}return {q,v,a};
  }
  Vec legVelocity(double t,const Vec&l){auto s=raw(*active,t,false);s[0].segment(7,12)=l;auto c=contact(s[0],Vec::Zero(24));return c.first.block(0,6,12,12).partialPivLu().solve(-c.first.leftCols(6)*s[1].head(6));}
  WReference reference(double t){
    if(gait.rows()){
      auto st=raw(*active,t);auto s=evaluate(st[0],st[1]);int k=std::clamp(int((t-active->t)/active->dt),0,9);
      WReference r={st[0],st[1],st[2],active->tau.row(k).transpose(),active->force.row(k).transpose(),st[1].head(6),st[2].head(6),s.J*st[1],s.J*st[2]+s.jdot,pose(st[0]).toHomogeneousMatrix(),s.pose};
      Vec g=gait.row(std::min(int(std::lround(t/.0025)),int(gait.rows()-1))).transpose();r.mode=g.head(4);r.fp=g.segment(4,12);r.fv=g.segment(16,12);r.fa=g.segment(28,12);r.fnmin=g.segment(40,4);r.fnmax=g.tail(4);return r;
    }
    while(legtime<t-1e-10){double h=std::min(.005,t-legtime);Vec k1=legVelocity(legtime,legs),k2=legVelocity(legtime+h/2,legs+h*k1/2),k3=legVelocity(legtime+h/2,legs+h*k2/2),k4=legVelocity(legtime+h,legs+h*k3);legs+=h*(k1+2*k2+2*k3+k4)/6;legtime+=h;}
    auto st=raw(*active,t);st[0].segment(7,12)=legs;auto c=contact(st[0],Vec::Zero(24));st[1].segment(6,12)=c.first.block(0,6,12,12).partialPivLu().solve(-c.first.leftCols(6)*st[1].head(6));c=contact(st[0],st[1]);st[2].segment(6,12)=c.first.block(0,6,12,12).partialPivLu().solve(-c.second-c.first.leftCols(6)*st[2].head(6));
    auto s=evaluate(st[0],st[1]);int k=std::clamp(int((t-active->t)/active->dt),0,9);
    return {st[0],st[1],st[2],active->tau.row(k).transpose(),active->force.row(k).transpose(),st[1].head(6),st[2].head(6),s.J*st[1],s.J*st[2]+s.jdot,pose(st[0]).toHomogeneousMatrix(),s.pose};
  }
  void run(double duration,int cpu){
    if(cpu>=0){cpu_set_t set;CPU_ZERO(&set);CPU_SET(cpu,&set);pthread_setaffinity_np(pthread_self(),sizeof(set),&set);}
    using Clock=std::chrono::steady_clock;auto origin=Clock::now()+std::chrono::milliseconds(30);double start=now()+30;
    try{
      for(int k=0;k<std::lround(duration/dt)&&!stop;++k){double t=k*dt;std::this_thread::sleep_until(origin+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(t)));double begin=now(),lag=begin-start-1000*t;
        plant.sim.read(plant.robot);Vec q=plant.robot.q,v=plant.robot.v;WState s=evaluate(q,v);int contacts=plant.sim.stanceContacts();double stamp=now();
        std::shared_ptr<const RuntimePolicy> candidate;
        if(policy_mutex.try_lock()){candidate=std::move(pending);policy_mutex.unlock();}
        bool installed=false;
        if(candidate){if((!active||candidate->sequence>active->sequence)&&candidate->t<=t+1e-8&&t-candidate->t<=.08&&t+.04<candidate->t+.4){active=candidate;legs=raw(*active,t)[0].segment(7,12);legtime=t;++accepted;installed=true;}else ++rejected;}
        if(!active||t+.04>=active->t+.4)throw std::runtime_error("No unexpired MPC policy");
        double exchange=now()-stamp;stamp=now();WReference r=reference(t);double reference_ms=now()-stamp;
        WResult sol=wbc.control(s,r);double control=now()-begin;
        if(!sol.tau.allFinite()||q[2]<.35||(s.qmin-q.tail(18)).maxCoeff()>1e-5||(q.tail(18)-s.qmax).maxCoeff()>1e-5||(v.tail(18).cwiseAbs()-s.vmax).maxCoeff()>1e-4)throw std::runtime_error("Measured physical state gate");
        auto o=std::make_shared<RuntimeObservation>();o->time=t;o->sequence=k;o->q=q;o->v=v;o->tau=sol.tau;o->acc=sol.acc;o->contacts=contacts;
        if(obs_mutex.try_lock()){latest=o;obs_mutex.unlock();}
        if(push)plant.set_base_wrench((t>=6&&t<6.15)?Eigen::Vector3d(120,0,0):Eigen::Vector3d::Zero(),Eigen::Vector3d::Zero());
        double response=now()-start-1000*t;int clips=plant.advance_reference(sol.tau,r.q,r.v,std::lround(dt/.0005));double loop=now()-begin;
        Vec data(19);data<<t,control,sol.total_ms,reference_ms,exchange,lag,response,loop,contacts,clips,double(active->sequence),1000*(t-active->t),double(installed),sol.bound,sol.lock,sol.kkt,sol.dynamics,sol.contact,sol.interface;
        rows.push_back({data,q,v,sol.tau,sol.acc,r.q,r.v,s.pose});
      }
    }catch(const std::exception&e){failure=e.what();}
    elapsed=(now()-start)/1000.;live=false;
  }
 public:
  ControlRuntime(const std::string&root,const std::string&body,const std::string&arm,const std::string&lib,const Vec&previous,double h,bool dual,bool disturbance):plant(root),wbc(root+"/models/b2_z1.urdf",body,arm,lib,plant.robot.q,previous,h,dual),dt(h),push(disturbance){
    if(std::abs(h-.0025)>1e-12&&std::abs(h-.005)>1e-12)throw std::runtime_error("Invalid control interval");
    plant.set_timestep(.0005);plant.instrument=true;plant.instrument_base=true;auto o=std::make_shared<RuntimeObservation>();o->q=plant.robot.q;o->v=plant.robot.v;o->tau=previous;o->acc=Vec::Zero(24);latest=o;
  }
  ~ControlRuntime(){stop=true;if(thread.joinable())thread.join();}
  void setGait(const Mat&g){if(thread.joinable()||g.cols()!=48||g.rows()<2||!g.allFinite())throw std::runtime_error("Invalid gait table");for(int i=0;i<g.rows();++i){if((g.row(i).head(4).array()<0).any()||(g.row(i).head(4).array()>1).any()||g.row(i).head(4).sum()<2)throw std::runtime_error("Invalid gait contacts");}gait=g;}
  void publish(long sequence,double time,double step,const Mat&q0,const Mat&c,const Mat&tau,const Mat&force){
    if(q0.rows()!=10||q0.cols()!=25||c.rows()!=60||c.cols()!=24||tau.rows()!=10||tau.cols()!=18||force.rows()!=10||force.cols()!=12||std::abs(step-.04)>1e-10||!std::isfinite(time)||!q0.allFinite()||!c.allFinite()||!tau.allFinite()||!force.allFinite())throw std::runtime_error("Invalid policy packet");
    auto p=std::make_shared<RuntimePolicy>(RuntimePolicy{sequence,time,step,q0,c,tau,force});std::lock_guard<std::mutex> lock(policy_mutex);overwritten+=bool(pending);pending=p;
  }
  py::dict observation(){std::shared_ptr<const RuntimeObservation> o;{std::lock_guard<std::mutex>l(obs_mutex);o=latest;}py::dict d;d["time"]=o->time;d["sequence"]=o->sequence;d["q"]=o->q;d["v"]=o->v;d["tau"]=o->tau;d["acc"]=o->acc;d["contacts"]=o->contacts;return d;}
  void start(double duration,int cpu){if(thread.joinable()||!pending||duration<=0||duration>12)throw std::runtime_error("Start requires initial policy and unused runtime");rows.reserve(std::lround(duration/dt));plant.trace.reserve(std::lround(duration/.0005));plant.base_trace.reserve(std::lround(duration/.0005));live=true;thread=std::thread(&ControlRuntime::run,this,duration,cpu);}
  bool running(){return live;}
  void finish(){stop=true;if(thread.joinable())thread.join();}
  py::dict results(){if(live)throw std::runtime_error("Wait for runtime completion");if(thread.joinable())thread.join();py::dict d;d["failure"]=failure;d["wall_seconds"]=elapsed;d["accepted"]=accepted;d["rejected"]=rejected;d["overwritten"]=overwritten;
    Mat data(rows.size(),19),q(rows.size(),25),v(rows.size(),24),tau(rows.size(),18),acc(rows.size(),24),rq(rows.size(),25),rv(rows.size(),24),ee(rows.size(),16);
    for(size_t i=0;i<rows.size();++i){auto&r=rows[i];data.row(i)=r.values.transpose();q.row(i)=r.q.transpose();v.row(i)=r.v.transpose();tau.row(i)=r.tau.transpose();acc.row(i)=r.acc.transpose();rq.row(i)=r.rq.transpose();rv.row(i)=r.rv.transpose();ee.row(i)=Eigen::Map<Vec>(r.ep.data(),16).transpose();}
    d["rows"]=data;d["q"]=q;d["v"]=v;d["tau"]=tau;d["acc"]=acc;d["reference_q"]=rq;d["reference_v"]=rv;d["ee_pose_columns"]=ee;d["trace"]=plant.take_trace();d["base"]=plant.take_base_trace();return d;
  }
};
