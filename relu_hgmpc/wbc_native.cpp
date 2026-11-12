#include "wbc_native.hpp"
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <dlfcn.h>
#include <stdexcept>
#include <limits>
#include <numeric>
namespace {
using V6=Eigen::Matrix<double,6,1>;using M6=Eigen::Matrix<double,6,6>;
using SE3=pinocchio::SE3;
double inf(const Vec& x){return x.size()?x.cwiseAbs().maxCoeff():0.;}
SE3 pose(const Vec&q){return SE3(Eigen::Quaterniond(q[6],q[3],q[4],q[5]).toRotationMatrix(),q.head<3>());}
SE3 pose(const Mat&T){return SE3(T.topLeftCorner<3,3>(),T.topRightCorner<3,1>());}
M6 ad(const Vec&v){return pinocchio::Motion(V6(v)).toActionMatrix();}
struct Err {Vec e,de,b;Mat J;};
Err error(const SE3&T,const Vec&V,const Vec&drift,const Mat&J,const SE3&D,const Vec&Vd,const Vec&Ad){
  SE3 E=D.inverse()*T;M6 X=E.inverse().toActionMatrix(),L,Lp,Lm;
  Vec tr=X*Vd,eta=V-tr;pinocchio::Jlog6(E,L);
  pinocchio::Jlog6(E*pinocchio::exp6(V6(1e-5*eta)),Lp);
  pinocchio::Jlog6(E*pinocchio::exp6(V6(-1e-5*eta)),Lm);
  return {pinocchio::log6(E).toVector(),L*eta,L*(drift+ad(eta)*tr-X*Ad)+(Lp-Lm)/(2e-5)*eta,L*J};
}
Mat select(const Mat&A,const std::vector<int>&ii,const std::vector<int>&jj){Mat B(ii.size(),jj.size());for(size_t i=0;i<ii.size();++i)for(size_t j=0;j<jj.size();++j)B(i,j)=A(ii[i],jj[j]);return B;}
Vec select(const Vec&a,const std::vector<int>&ii){Vec b(ii.size());for(size_t i=0;i<ii.size();++i)b[i]=a[ii[i]];return b;}
Mat stack(const std::vector<Mat>&xs){int n=0;for(auto&x:xs)n+=x.rows();Mat a(n,xs.front().cols());n=0;for(auto&x:xs){a.middleRows(n,x.rows())=x;n+=x.rows();}return a;}
Vec stack(const std::vector<Vec>&xs){int n=0;for(auto&x:xs)n+=x.size();Vec a(n);n=0;for(auto&x:xs){a.segment(n,x.size())=x;n+=x.size();}return a;}
Mat kernel(const Mat&A){if(!A.rows())return Mat::Identity(A.cols(),A.cols());Eigen::JacobiSVD<Mat> s(A,Eigen::ComputeFullV);auto d=s.singularValues();int rank=0;for(int i=0;i<d.size();++i)rank+=d[i]>1e-10*d[0];return s.matrixV().rightCols(A.cols()-rank);}
double bound(const Mat&G,const Vec&l,const Vec&u,const Vec&x){Vec gx=G*x;return std::max({0.,(l-gx).maxCoeff(),(gx-u).maxCoeff()});}
double residual(const Mat&H,const Vec&g,const Mat&G,const Vec&l,const Vec&u,const Vec&x,const Vec&y){
  Vec gx=G*x,hx=H*x,gy=G.transpose()*y;double ds=std::max({1.,inf(hx),inf(g),inf(gy)}),ps=1,pr=0,nr=0;
  for(int i=0;i<G.rows();++i){double r=1/std::max(1.,G.row(i).norm()),v=std::clamp(gx[i],l[i],u[i]);ps=std::max({ps,std::abs(r*gx[i]),std::abs(r*v)});pr=std::max(pr,std::abs(r*(gx[i]-v)));nr=std::max(nr,std::abs(r*gx[i]-std::clamp(r*gx[i]+y[i]/r,r*l[i],r*u[i])));}
  return std::max({pr/ps,nr/ps,inf(hx+g+gy)/ds});
}
}
struct NativeWbc::Impl {
  pinocchio::Model full,body,arm;std::unique_ptr<pinocchio::Data> db,da;
  SE3 mount;Mat JiB,JiA,Ub,Ua,cone;Vec prev,scale;double dt;bool dual;
  void* library=nullptr;
  using Qp=int(*)(double*,double*,int,double*,double*,double*,double*,double*,int,int,int*,int*,int*,double*,int);
  Qp qp=nullptr;std::vector<int> bi,ai;
  Impl(const std::string&f,const std::string&b,const std::string&a,const std::string&lib,Vec home,Vec previous,double h,bool split):prev(previous),dt(h),dual(split){
    pinocchio::urdf::buildModel(f,pinocchio::JointModelFreeFlyer(),full);
    pinocchio::urdf::buildModelFromXML(b,pinocchio::JointModelFreeFlyer(),body);
    pinocchio::urdf::buildModelFromXML(a,pinocchio::JointModelFreeFlyer(),arm);
    db=std::make_unique<pinocchio::Data>(body);da=std::make_unique<pinocchio::Data>(arm);
    pinocchio::Data d(full);pinocchio::forwardKinematics(full,d,home);pinocchio::updateFramePlacements(full,d);
    mount=pose(home).inverse()*d.oMf[full.getFrameId("z1_link00")];
    JiB=Mat::Zero(6,18);JiA=Mat::Zero(6,12);JiB.leftCols(6)=mount.inverse().toActionMatrix();JiA.leftCols(6)=JiB.leftCols(6);
    Ub=Mat::Identity(18,18).topRows(12);Ua=Mat::Identity(18,18).bottomRows(6);
    scale=Vec::Constant(18,100.);scale.tail(6).setConstant(10.);
    for(int i=0;i<18;++i)bi.push_back(i);for(int i=0;i<6;++i)ai.push_back(i);for(int i=18;i<24;++i)ai.push_back(i);
    cone=Mat::Zero(20,12);double mu=.6/std::sqrt(2.);Mat c(5,3);c<<1,0,-mu,-1,0,-mu,0,1,-mu,0,-1,-mu,0,0,1;for(int i=0;i<4;++i)cone.block(5*i,3*i,5,3)=c;
    // Match the installed, pinned quadprog 0.1.13 C kernel exactly. Each call
    // owns all work arrays. No Python API or Python lock is used in control().
    library=dlopen(lib.c_str(),RTLD_NOW|RTLD_LOCAL);if(!library)throw std::runtime_error(dlerror());
    qp=reinterpret_cast<Qp>(dlsym(library,"qpgen2_"));if(!qp)throw std::runtime_error("quadprog kernel symbol missing");
  }
  ~Impl(){if(library)dlclose(library);}
  Vec solve(int level,const Mat&C,const Vec&d,const Mat&G,const Vec&l,const Vec&u,std::vector<int>&active,double&cert){
    Mat H=2*C.transpose()*C;Vec g=2*C.transpose()*d;double os=std::max(1.,H.cwiseAbs().maxCoeff());H/=os;g/=os;
    if(level==2){Vec z=C.completeOrthogonalDecomposition().solve(-d);cert=residual(H,g,G,l,u,z,Vec::Zero(l.size()));if(cert<1e-10){active.clear();return z;}}
    std::vector<Vec> cs;std::vector<double> bs,rs;std::vector<int> ids,signs;
    for(int sign:{1,-1})for(int i=0;i<G.rows();++i){double norm=G.row(i).norm();if(norm<1e-10){if(std::max(l[i],-u[i])>2e-7)throw std::runtime_error("Inherited zero bound");continue;}
      double bnd=sign==1?l[i]:-u[i];if(!std::isfinite(bnd))continue;if(level>0&&bnd>0&&bnd<2e-7)bnd=0;
      double sc=1/std::max(norm,1e-12);cs.push_back(sign*sc*G.row(i).transpose());bs.push_back(bnd*sc);rs.push_back(sc);ids.push_back(i);signs.push_back(sign);
    }
    int n=g.size(),m=cs.size(),nact=0,it[2]={0,0};Mat hc=H;hc.diagonal().array()+=1e-10;Vec av=-g,sol=Vec::Zero(n),lag=Vec::Zero(m),bv(m);Mat ac(n,m);
    for(int i=0;i<m;++i){ac.col(i)=cs[i];bv[i]=bs[i];}std::vector<int>ia(m);int r=std::min(n,m);Vec work=Vec::Zero(2*n+2*m+r*(r+5)/2);double obj=0;
    int status=qp(hc.data(),av.data(),n,sol.data(),lag.data(),&obj,ac.data(),bv.data(),m,0,ia.data(),&nact,it,work.data(),0);
    if(status)throw std::runtime_error("Native QP status "+std::to_string(status)+" level "+std::to_string(level));
    Vec y=Vec::Zero(G.rows());for(int i=0;i<m;++i)y[ids[i]]-=signs[i]*lag[i]*rs[i];
    cert=residual(H,g,G,l,u,sol,y);if(cert>2e-7)throw std::runtime_error("Native original KKT "+std::to_string(cert));
    active.clear();for(int i=0;i<y.size();++i)if(std::abs(y[i])>1e-8)active.push_back(i);return sol;
  }
  WResult control(const WState&s,const WReference&r){
    double start=now();WResult out;Mat T,F,W,Mb,Ma;Vec a0,f0,w0,hb,ha;Mat BAb,BAa;Vec bbb,baa;
    Vec ce=s.ce;for(int i=0;i<4;++i)ce.segment(3*i,2).setZero();Vec restoration=-s.cdot-100*ce-20*s.Jc*s.v;
    if(dual){
      pinocchio::crba(body,*db,s.q);Mat mbody=db->M.selfadjointView<Eigen::Upper>();Mb=select(mbody,bi,bi);hb=select(Vec(pinocchio::nonLinearEffects(body,*db,s.q,s.v)),bi);
      pinocchio::crba(arm,*da,s.q);Mat marm=da->M.selfadjointView<Eigen::Upper>();Ma=select(marm,ai,ai);ha=select(Vec(pinocchio::nonLinearEffects(arm,*da,s.q,s.v)),ai);
      Mat K=Mat::Zero(30,30);K.topLeftCorner(18,18)=Mb;K.topRightCorner(18,12)=-s.Jc.leftCols(18).transpose();K.bottomLeftCorner(12,18)=s.Jc.leftCols(18);
      Mat rhs=Mat::Zero(30,19);rhs.block(6,0,12,12).setIdentity();rhs.block(0,12,18,6)=JiB.transpose();rhs.col(18).head(18)=-hb;rhs.col(18).tail(12)=restoration;
      Mat zm=K.partialPivLu().solve(rhs),arhs=Mat::Zero(12,13);arhs.block(6,0,6,6).setIdentity();arhs.block(0,6,12,6)=-JiA.transpose();arhs.col(12)=-ha;Mat za=Ma.llt().solve(arhs);
      Mat D=JiB*zm.block(0,12,18,6)-JiA*za.block(0,6,12,6);Mat wr(6,19);wr.leftCols(18)=JiA*za.leftCols(6)*Ua-JiB*zm.topLeftCorner(18,12)*Ub;wr.col(18)=JiA*za.col(12)-JiB*zm.col(18).head(18);Mat wm=D.partialPivLu().solve(wr);W=wm.leftCols(18);w0=wm.col(18);
      BAb=zm.topLeftCorner(18,12)*Ub+zm.block(0,12,18,6)*W;bbb=zm.col(18).head(18)+zm.block(0,12,18,6)*w0;
      BAa=za.leftCols(6)*Ua+za.block(0,6,12,6)*W;baa=za.col(12)+za.block(0,6,12,6)*w0;
      T.resize(24,18);T.topRows(18)=BAb;T.bottomRows(6)=BAa.bottomRows(6);a0.resize(24);a0.head(18)=bbb;a0.tail(6)=baa.tail(6);
      F=zm.block(18,0,12,12)*Ub+zm.block(18,12,12,6)*W;f0=zm.col(18).tail(12)+zm.block(18,12,12,6)*w0;
    }else{
      Mat K=Mat::Zero(36,36);K.topLeftCorner(24,24)=s.M;K.topRightCorner(24,12)=-s.Jc.transpose();K.bottomLeftCorner(12,24)=s.Jc;
      Mat rhs=Mat::Zero(36,19);rhs.block(6,0,18,18).setIdentity();rhs.col(18).head(24)=-s.h;rhs.col(18).tail(12)=restoration;Mat z=K.partialPivLu().solve(rhs);T=z.topLeftCorner(24,18);a0=z.col(18).head(24);F=z.bottomLeftCorner(12,18);f0=z.col(18).tail(12);
      BAb=T.topRows(18);bbb=a0.head(18);BAa=stack(std::vector<Mat>{T.topRows(6),T.bottomRows(6)});baa=stack(std::vector<Vec>{a0.head(6),a0.tail(6)});
    }
    Mat A=T*scale.asDiagonal(),B=F*scale.asDiagonal();std::vector<Mat> gs;std::vector<Vec> ls,us;
    auto add=[&](Mat g,Vec l,Vec u){gs.push_back(g);ls.push_back(l);us.push_back(u);};
    Vec rate=Vec::Constant(18,1000.);rate.tail(6).setConstant(100.);Vec am=Vec::Constant(18,50.);am.tail(6).setConstant(30.);Vec vm=s.vmax.cwiseMin(4.);
    // Same bounds as both local WBCs, in a globally reordered row layout.
    add(Mat(scale.asDiagonal()),-.85*s.taumax,.85*s.taumax);add(Mat(scale.asDiagonal()),prev-dt*rate,prev+dt*rate);
    add(dt*A.bottomRows(18),-vm-s.v.tail(18)-dt*a0.tail(18),vm-s.v.tail(18)-dt*a0.tail(18));
    add(.5*dt*dt*A.bottomRows(18),s.qmin.array()+.02-s.q.tail(18).array()-dt*s.v.tail(18).array()-.5*dt*dt*a0.tail(18).array(),s.qmax.array()-.02-s.q.tail(18).array()-dt*s.v.tail(18).array()-.5*dt*dt*a0.tail(18).array());
    add(A.bottomRows(18),-am-a0.tail(18),am-a0.tail(18));Vec fl(20),fu(20);for(int i=0;i<20;++i){fl[i]=(i%5==4?10.:-std::numeric_limits<double>::infinity());fu[i]=(i%5==4?800.:0.);}add(cone*B,fl-cone*f0,fu-cone*f0);
    double eta=5.,coef=dt+.5*eta*dt*dt;add(coef*A.bottomRows(18),-eta*(s.q.tail(18)-s.qmin-Vec::Constant(18,.02))-(1+eta*dt)*s.v.tail(18)-coef*a0.tail(18),eta*(s.qmax-Vec::Constant(18,.02)-s.q.tail(18))-(1+eta*dt)*s.v.tail(18)-coef*a0.tail(18));
    out.G=stack(gs);out.lo=stack(ls);out.hi=stack(us);
    Mat Jb=Mat::Identity(24,24).topRows(6);Err eb=error(pose(s.q),s.v.head(6),Vec::Zero(6),Jb,pose(r.bp),r.bv,r.ba);
    Err ee=error(pose(s.pose),s.J*s.v,s.jdot,s.J,pose(r.ep),r.ev,r.ea);
    SE3 rel=(pose(s.q)*mount).inverse()*pose(s.pose);Mat X=rel.toActionMatrix(),Ja=Mat::Zero(6,24);Ja.rightCols(6)=s.J.rightCols(6);Vec w=Ja*s.v;Mat aw=ad(w);Vec dw=s.jdot+aw*(s.J*s.v-w);
    Err e={X*ee.e,X*(ee.de+aw*ee.e),X*(ee.b+2*aw*ee.de+ad(dw)*ee.e+aw*aw*ee.e),X*(ee.J-ad(ee.e)*Ja)};
    Vec weight(6);weight<<20,20,20,5,5,5;
    out.tasks.push_back({weight.asDiagonal()*e.J*A,weight.asDiagonal()*(e.J*a0+e.b+100*e.e+20*e.de)});
    out.tasks.push_back({weight.asDiagonal()*eb.J*A,weight.asDiagonal()*(eb.J*a0+eb.b+200*eb.e+28*eb.de)});
    Vec target=r.a+100*pinocchio::difference(full,s.q,r.q)+20*(r.v-s.v);
    out.tasks.push_back({stack(std::vector<Mat>{.1*A.bottomRows(18),.1*B,Mat::Identity(18,18),.2*Mat::Identity(18,18)}),stack(std::vector<Vec>{.1*(a0.tail(18)-target.tail(18)),.1*(f0-r.force),-prev.cwiseQuotient(scale),-.2*r.tau.cwiseQuotient(scale)})});
    out.assembly_ms=now()-start;double st=now();Vec x=Vec::Zero(18);Mat Z=Mat::Identity(18,18);std::vector<Vec> locks;
    for(int i=0;i<3;++i){auto&t=out.tasks[i];std::vector<int> active;Mat cr=t.C*Z,gr=out.G*Z;
      if(Z.cols()){double cert=0;Vec y=solve(i,cr,t.C*x+t.d,gr,out.lo-out.G*x,out.hi-out.G*x,active,cert);x+=Z*y;out.kkt=std::max(out.kkt,cert);}locks.push_back(t.C*x);
      if(i<2&&Z.cols()){std::vector<Vec>rows;for(int j=0;j<cr.rows();++j)if(cr.row(j).norm()>1e-10)rows.push_back(cr.row(j).transpose()/cr.row(j).norm());for(int j:active)if(gr.row(j).norm()>1e-10)rows.push_back(gr.row(j).transpose()/gr.row(j).norm());Mat face(rows.size(),Z.cols());for(size_t j=0;j<rows.size();++j)face.row(j)=rows[j].transpose();Z=(Z*kernel(face)).eval();}
    }
    out.solve_ms=now()-st;out.bound=bound(out.G,out.lo,out.hi,x);for(int i=0;i<3;++i)out.lock=std::max(out.lock,inf(out.tasks[i].C*x-locks[i]));
    out.tau=scale.cwiseProduct(x);out.acc=T*out.tau+a0;out.force=F*out.tau+f0;Vec gen=Vec::Zero(24);gen.tail(18)=out.tau;
    out.dynamics=inf(s.M*out.acc+s.h-gen-s.Jc.transpose()*out.force);out.contact=inf(s.Jc*out.acc-restoration);
    if(dual){out.wrench=W*out.tau+w0;Vec ab=BAb*out.tau+bbb,aa=BAa*out.tau+baa,gb=Vec::Zero(18),ga=Vec::Zero(12);gb.tail(12)=out.tau.head(12);ga.tail(6)=out.tau.tail(6);out.interface=inf(JiB*ab-JiA*aa);out.body=inf(Mb*ab+hb-gb-s.Jc.leftCols(18).transpose()*out.force-JiB.transpose()*out.wrench);out.arm=inf(Ma*aa+ha-ga+JiA.transpose()*out.wrench);}
    if(out.bound>2e-5||out.lock>2e-7||std::max({out.dynamics,out.contact,out.interface,out.body,out.arm})>1e-7)throw std::runtime_error("Native WBC physical certificate failed");
    prev=out.tau;out.total_ms=now()-start;return out;
  }
};
NativeWbc::NativeWbc(const std::string&f,const std::string&b,const std::string&a,const std::string&lib,Vec home,Vec prev,double dt,bool dual):p(std::make_unique<Impl>(f,b,a,lib,home,prev,dt,dual)){}
NativeWbc::~NativeWbc()=default;
WResult NativeWbc::control(const WState&s,const WReference&r){return p->control(s,r);}
void NativeWbc::setPrevious(const Vec&t){p->prev=t;}
