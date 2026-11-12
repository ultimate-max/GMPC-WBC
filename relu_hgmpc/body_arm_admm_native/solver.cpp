#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <omp.h>
#include <array>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace py = pybind11;
using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;
using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point a) {
  return std::chrono::duration<double, std::milli>(Clock::now()-a).count();
}
static double infnorm(const Vec& v) { return v.size() ? v.cwiseAbs().maxCoeff() : 0.; }
static double maxabs(const Mat& v) { return v.size() ? v.cwiseAbs().maxCoeff() : 0.; }
static double clip(double v, double l, double u) { return std::max(l, std::min(v,u)); }

// Set the linked OpenBLAS backend to one thread; each local solve has its own
// outer OpenMP worker. Weak symbols permit a sequential BLAS implementation.
extern "C" void openblas_set_num_threads(int) __attribute__((weak));
extern "C" int openblas_get_num_threads(void) __attribute__((weak));

struct Block {
  Mat H,G,E,B,gram,system;
  Vec g,l,u,e,rs,mrs,ls,us,c,x,y,z,value,relaxed,rhs,oldz,newx;
  Vec physical_gdual,physical_edual,gx,ex,hx,aty;
  Eigen::LLT<Mat> factor;
  Eigen::Index n=0,m=0,p=0;
  double factor_ms=0.;
  int factors=0;
  bool ok=true;

  void init(const Mat& Hin,const Vec& gin,const Mat& Gin,const Vec& lin,
            const Vec& uin,const Mat& Ein,const Vec& ein,const Vec& w,const Vec& xin) {
    H=Hin;g=gin;G=Gin;l=lin;u=uin;E=Ein;e=ein;x=xin;
    n=g.size();m=l.size();p=e.size();
    rs=Vec::Ones(m);mrs=Vec::Ones(m);
    for(Eigen::Index r=0;r<m;++r) {
      const double norm=G.row(r).norm();
      rs[r]=norm>1e-10 ? 1./norm : 1.;
      mrs[r]=1./std::max(1.,norm);
    }
    B.resize(m+p,n);
    B.topRows(m)=rs.asDiagonal()*G;
    B.bottomRows(p)=w.asDiagonal()*E;
    c=Vec::Zero(m+p);c.tail(p)=w.cwiseProduct(e);
    ls=rs.cwiseProduct(l);us=rs.cwiseProduct(u);
    gram=Mat::Zero(n,n);
    if(n) {
      gram.selfadjointView<Eigen::Lower>().rankUpdate(B.transpose());
      gram=gram.selfadjointView<Eigen::Lower>();
    }
    y=Vec::Zero(m+p);value=B*x+c;z=value;
    for(Eigen::Index r=0;r<m;++r) z[r]=clip(z[r],ls[r],us[r]);
    oldz=z;relaxed=z;rhs=Vec::Zero(n);newx=x;
    physical_gdual=Vec::Zero(m);physical_edual=Vec::Zero(p);
    gx=Vec::Zero(m);ex=Vec::Zero(p);hx=Vec::Zero(n);aty=Vec::Zero(n);
  }
  void refactor(double rho,double sigma) {
    if(!n) return;
    const auto st=Clock::now();
    system=H+rho*gram;system.diagonal().array()+=sigma;
    factor.compute(system);ok=(factor.info()==Eigen::Success);
    factor_ms+=elapsed(st);++factors;
  }
  void primal_update(double rho,double sigma,double alpha) {
    if(n) {
      rhs.noalias()=B.transpose()*(rho*(z-c)-y);
      rhs+=sigma*x-g;
      newx.noalias()=factor.solve(rhs);
      ok=newx.allFinite();x.swap(newx);
    }
    value.noalias()=B*x;value+=c;
    oldz=z;relaxed=alpha*value+(1.-alpha)*oldz;
  }
};

struct Metrics {
  double objective=0.,bound_raw=0.,primal_raw=0.,primal_relative=0.;
  double interface_raw=0.,scaled_interface=0.;
  double stationarity_raw=0.,stationarity=0.,normal=0.,interface_dual_balance=0.;
};

static Metrics measure(std::array<Block,2>& b,const Vec& w) {
  Metrics r;
  double normal_raw=0.,normal_scale=1.,dual_scale=1.;
  for(auto& s:b) {
    s.gx.noalias()=s.G*s.x;s.ex.noalias()=s.E*s.x;s.ex+=s.e;
    s.physical_gdual=s.rs.cwiseProduct(s.y.head(s.m));
    s.physical_edual=w.cwiseProduct(s.y.tail(s.p));
    s.hx.noalias()=s.H*s.x;
    s.aty.noalias()=s.G.transpose()*s.physical_gdual;
    s.aty.noalias()+=s.E.transpose()*s.physical_edual;
    const double ds=std::max({1.,infnorm(s.hx),infnorm(s.g),infnorm(s.aty)});
    const double dr=infnorm(s.hx+s.g+s.aty);
    r.stationarity_raw=std::max(r.stationarity_raw,dr);
    r.stationarity=std::max(r.stationarity,dr/ds);
    dual_scale=std::max(dual_scale,infnorm(s.physical_edual));
    r.objective+=.5*s.x.dot(s.hx)+s.g.dot(s.x);
    for(Eigen::Index j=0;j<s.m;++j) {
      const double proj=clip(s.gx[j],s.l[j],s.u[j]);
      const double metric=s.mrs[j], a=metric*s.gx[j], p=metric*proj;
      r.bound_raw=std::max(r.bound_raw,std::abs(s.gx[j]-proj));
      normal_scale=std::max({normal_scale,std::abs(a),std::abs(p)});
      r.primal_relative=std::max(r.primal_relative,metric*std::abs(s.gx[j]-proj));
      const double proj_normal=clip(a+s.physical_gdual[j]/metric,
                                    metric*s.l[j],metric*s.u[j]);
      normal_raw=std::max(normal_raw,std::abs(a-proj_normal));
    }
  }
  r.interface_raw=infnorm(b[0].ex-b[1].ex);
  r.scaled_interface=infnorm(w.cwiseProduct(b[0].ex-b[1].ex));
  r.interface_dual_balance=infnorm(b[0].physical_edual+b[1].physical_edual);
  r.primal_relative/=normal_scale;
  r.primal_raw=std::max(r.bound_raw,r.interface_raw);
  r.normal=std::max(normal_raw/normal_scale,r.interface_dual_balance/dual_scale);
  return r;
}

// Graph consensus ADMM for TWO local convex QPs. Algorithm block one is the
// product (x_body,x_arm), whose separable quadratic updates run in parallel.
// Block two is projection onto boxes and a common interface. No constraint is
// claimed feasible before the independent original-coordinate certificate.
static py::dict solve(const Mat& Hb,const Vec& gb,const Mat& Gb,const Vec& lb,const Vec& ub,
                      const Mat& Eb,const Vec& eb,
                      const Mat& Ha,const Vec& ga,const Mat& Ga,const Vec& la,const Vec& ua,
                      const Mat& Ea,const Vec& ea,const Vec& interface_scale,
                      py::object warm_x_b,py::object warm_x_a,
                      int max_iter,double tol,double interface_tol,double bound_tol,
                      double rho,double sigma,double alpha,int check_interval,
                      int adaptive_interval,double adaptive_threshold,bool parallel) {
  const auto total_start=Clock::now();
  if(!std::isfinite(tol)||!std::isfinite(interface_tol)||!std::isfinite(bound_tol)||
     !std::isfinite(rho)||!std::isfinite(sigma)||!std::isfinite(alpha)||!std::isfinite(adaptive_threshold)||
     max_iter<=0||tol<=0||interface_tol<=0||bound_tol<=0||rho<=0||sigma<=0||
     alpha<=0||alpha>=2||check_interval<=0||adaptive_interval<0||adaptive_threshold<=1.)
    throw std::invalid_argument("invalid solver settings");
  if(!interface_scale.allFinite()||(interface_scale.array()<=0).any()||
     eb.size()!=ea.size()||eb.size()!=interface_scale.size())
    throw std::invalid_argument("interface dimensions/scales do not match");
  auto validate=[](const Mat& H,const Vec& g,const Mat& G,const Vec& l,const Vec& u,
                   const Mat& E,const Vec& e,const Vec& x) {
    const Eigen::Index n=g.size(),m=l.size();
    if(H.rows()!=n||H.cols()!=n||G.rows()!=m||G.cols()!=n||u.size()!=m||
       E.rows()!=e.size()||E.cols()!=n||x.size()!=n)
      throw std::invalid_argument("local QP dimensions do not match");
    if(!H.allFinite()||!g.allFinite()||!G.allFinite()||!E.allFinite()||!e.allFinite()||!x.allFinite())
      throw std::invalid_argument("nonfinite local matrix/vector");
    if(maxabs(H-H.transpose())>1e-9*std::max(1.,maxabs(H)))
      throw std::invalid_argument("local H is not symmetric");
    for(Eigen::Index j=0;j<m;++j)
      if(std::isnan(l[j])||std::isnan(u[j])||l[j]>u[j]||
         l[j]==std::numeric_limits<double>::infinity()||u[j]==-std::numeric_limits<double>::infinity())
        throw std::invalid_argument("invalid local interval");
  };
  Vec xb=warm_x_b.is_none()?Vec::Zero(gb.size()):warm_x_b.cast<Vec>();
  Vec xa=warm_x_a.is_none()?Vec::Zero(ga.size()):warm_x_a.cast<Vec>();
  validate(Hb,gb,Gb,lb,ub,Eb,eb,xb);validate(Ha,ga,Ga,la,ua,Ea,ea,xa);
  std::array<Block,2> b;
  Metrics result;
  double setup_ms=0.,factor_ms=0.,iteration_ms=0.,coord_ms=0.,check_ms=0.;
  int iterations=0,checks=0,rho_updates=0,threads_used=1,blas_threads=-1;
  bool converged=false;
  std::string status="maximum iterations";
  {
    py::gil_scoped_release release;
    const auto ts=Clock::now();
    if(openblas_set_num_threads)openblas_set_num_threads(1);
    if(openblas_get_num_threads)blas_threads=openblas_get_num_threads();
    b[0].init(Hb,gb,Gb,lb,ub,Eb,eb,interface_scale,xb);
    b[1].init(Ha,ga,Ga,la,ua,Ea,ea,interface_scale,xa);
    const Eigen::Index p=eb.size();
    Vec common=.5*(b[0].z.tail(p)+b[1].z.tail(p));
    b[0].z.tail(p)=common;b[1].z.tail(p)=common;
    setup_ms=elapsed(ts);
    auto refactor=[&]() {
      const auto ft=Clock::now();
      #pragma omp parallel for num_threads(2) if(parallel) schedule(static)
      for(int i=0;i<2;++i) b[i].refactor(rho,sigma);
      factor_ms+=elapsed(ft);
      return b[0].ok&&b[1].ok;
    };
    if(!refactor())status="factorization failed";
    else for(int k=1;k<=max_iter;++k) {
      const auto it=Clock::now();
      #pragma omp parallel for num_threads(2) if(parallel) schedule(static)
      for(int i=0;i<2;++i) {
        if(i==0)threads_used=omp_get_num_threads();
        b[i].primal_update(rho,sigma,alpha);
      }
      iteration_ms+=elapsed(it);
      if(!b[0].ok||!b[1].ok) {status="nonfinite iterate";break;}
      const auto ct=Clock::now();
      for(auto& s:b)
        for(Eigen::Index j=0;j<s.m;++j)
          s.z[j]=clip(s.relaxed[j]+s.y[j]/rho,s.ls[j],s.us[j]);
      common=.5*(b[0].relaxed.tail(p)+b[0].y.tail(p)/rho+
                  b[1].relaxed.tail(p)+b[1].y.tail(p)/rho);
      for(auto& s:b) {
        s.z.tail(p)=common;
        s.y.noalias()+=rho*(s.relaxed-s.z);
      }
      coord_ms+=elapsed(ct);iterations=k;
      if(k%check_interval==0||k==max_iter) {
        const auto ck=Clock::now();result=measure(b,interface_scale);++checks;
        check_ms+=elapsed(ck);
        if(result.bound_raw<=bound_tol&&result.interface_raw<=interface_tol&&
           result.stationarity<=tol&&result.normal<=tol&&result.primal_relative<=tol) {
          converged=true;status="solved";break;
        }
      }
      if(adaptive_interval>0&&k%adaptive_interval==0&&k<max_iter) {
        const auto ck=Clock::now();
        double rp=0.,rd=0.;
        for(auto& s:b) {
          rp=std::max(rp,infnorm(s.value-s.z)/std::max({1.,infnorm(s.value),infnorm(s.z)}));
          s.hx.noalias()=s.H*s.x;s.aty.noalias()=s.B.transpose()*s.y;
          rd=std::max(rd,infnorm(s.hx+s.g+s.aty)/std::max({1.,infnorm(s.hx),infnorm(s.g),infnorm(s.aty)}));
        }
        const double proposed=std::clamp(rho*std::sqrt(std::max(rp,1e-16)/std::max(rd,1e-16)),1e-6,1e6);
        check_ms+=elapsed(ck);
        if(proposed>rho*adaptive_threshold||proposed<rho/adaptive_threshold) {
          rho=proposed;++rho_updates;
          if(!refactor()){status="factorization failed";break;}
        }
      }
    }
    const auto ck=Clock::now();result=measure(b,interface_scale);check_ms+=elapsed(ck);
  }
  py::dict out;
  out["x_b"]=b[0].x;out["x_a"]=b[1].x;
  out["y_g_b"]=b[0].physical_gdual;out["y_g_a"]=b[1].physical_gdual;
  out["y_e_b"]=b[0].physical_edual;out["y_e_a"]=b[1].physical_edual;
  out["interface_b"]=b[0].ex;out["interface_a"]=b[1].ex;
  out["objective"]=result.objective;out["primal_raw"]=result.primal_raw;
  out["bound_raw"]=result.bound_raw;out["primal_relative"]=result.primal_relative;
  out["interface_raw"]=result.interface_raw;out["scaled_interface"]=result.scaled_interface;
  out["local_stationarity"]=result.stationarity;out["local_stationarity_raw"]=result.stationarity_raw;
  out["normal"]=result.normal;out["interface_dual_balance"]=result.interface_dual_balance;
  out["iterations"]=iterations;out["converged"]=converged;out["status"]=status;
  out["checks"]=checks;out["rho_updates"]=rho_updates;out["rho_final"]=rho;
  out["setup_ms"]=setup_ms;out["factor_ms"]=factor_ms;out["iteration_ms"]=iteration_ms;
  out["coord_ms"]=coord_ms;out["check_ms"]=check_ms;
  out["factorization_count"]=b[0].factors+b[1].factors;
  out["factorization_count_b"]=b[0].factors;out["factorization_count_a"]=b[1].factors;
  out["factor_ms_b"]=b[0].factor_ms;out["factor_ms_a"]=b[1].factor_ms;
  out["parallel"]=parallel;out["threads_used"]=threads_used;out["blas_threads"]=blas_threads;
  out["total_ms"]=elapsed(total_start);
  return out;
}

PYBIND11_MODULE(body_arm_admm_native,m) {
#ifdef ADMM_BLAS_ENABLED
  m.attr("blas_enabled")=true;
#else
  m.attr("blas_enabled")=false;
#endif
  m.attr("openmp_enabled")=true;
  m.def("solve",&solve,py::arg("Hb"),py::arg("gb"),py::arg("Gb"),py::arg("lb"),py::arg("ub"),
        py::arg("Eb"),py::arg("eb"),py::arg("Ha"),py::arg("ga"),py::arg("Ga"),py::arg("la"),
        py::arg("ua"),py::arg("Ea"),py::arg("ea"),py::arg("interface_scale"),
        py::arg("warm_x_b")=py::none(),py::arg("warm_x_a")=py::none(),
        py::arg("max_iter")=20000,py::arg("tol")=2e-7,py::arg("interface_tol")=2e-7,
        py::arg("bound_tol")=2e-7,py::arg("rho")=.1,py::arg("sigma")=1e-6,
        py::arg("alpha")=1.6,py::arg("check_interval")=10,py::arg("adaptive_interval")=50,
        py::arg("adaptive_threshold")=5.,py::arg("parallel")=true);
}
