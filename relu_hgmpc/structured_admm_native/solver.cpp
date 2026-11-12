#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <limits>
#include <string>
#include <stdexcept>

namespace py=pybind11;
using Mat=Eigen::MatrixXd;
using Vec=Eigen::VectorXd;
using Clock=std::chrono::steady_clock;
static double elapsed(Clock::time_point a){return std::chrono::duration<double,std::milli>(Clock::now()-a).count();}
static double infnorm(const Vec& a){return a.size()?a.cwiseAbs().maxCoeff():0.;}
static double clipped(double x,double l,double u){return std::max(l,std::min(x,u));}

struct Metrics {
  double primal_raw=0.,primal_relative=0.,dual_raw=0.,dual_relative=0.,normal_relative=0.,objective=0.;
  double error()const{return std::max({primal_relative,dual_relative,normal_relative});}
};
static Metrics metrics(const Mat& H,const Vec& g,const Mat& A,const Vec& l,const Vec& u,
                       const Vec& x,const Vec& y, const Vec& metric_rows,
                       Vec& ax,Vec& hx,Vec& aty){
  Metrics out;
  ax.noalias()=A*x; hx.noalias()=H*x; aty.noalias()=A.transpose()*y;
  double ps=1.,normal=0.,pr=0.;
  for(Eigen::Index j=0;j<ax.size();++j){
    double proj=clipped(ax[j],l[j],u[j]), r=metric_rows[j], ra=r*ax[j], rp=r*proj;
    ps=std::max({ps,std::abs(ra),std::abs(rp)});
    out.primal_raw=std::max(out.primal_raw,std::abs(ax[j]-proj));
    pr=std::max(pr,r*std::abs(ax[j]-proj));
    double nproj=clipped(ra+y[j]/r,r*l[j],r*u[j]);
    normal=std::max(normal,std::abs(ra-nproj));
  }
  out.primal_relative=pr/ps;
  out.normal_relative=normal/ps;
  out.dual_raw=infnorm(hx+g+aty);
  out.dual_relative=out.dual_raw/std::max({1.,infnorm(hx),infnorm(g),infnorm(aty)});
  out.objective=.5*x.dot(hx)+g.dot(x);
  return out;
}

// One proximal two-block ADMM applied to a fixed convex QP. sigma multiplies
// ||x-x_previous||^2, not the original objective. H is never regularized in
// the problem supplied to the solver. Equality rows are ordinary ADMM rows;
// caller-side contact/hierarchy elimination is the mechanism for exact equalities.
static py::dict solve(const Mat& H,const Vec& g,const Mat& A,const Vec& l,const Vec& u,
    py::object warm_x,py::object warm_y,double tol,double primal_abs_tol,int max_iter,
    double rho,double sigma,double alpha,int check_interval,int adaptive_interval,
    double adaptive_threshold,bool objective_scaling,bool row_scaling,double preconditioner_shift){
  auto total_start=Clock::now();
  const Eigen::Index n=g.size(),m=l.size();
  if(n<=0 || H.rows()!=n || H.cols()!=n || A.cols()!=n || A.rows()!=m || u.size()!=m)
    throw std::invalid_argument("inconsistent QP dimensions");
  if(!H.allFinite()||!g.allFinite()||!A.allFinite()||rho<=0||sigma<=0||alpha<=0||alpha>=2||
     tol<=0||max_iter<=0||check_interval<=0||adaptive_interval<0||adaptive_threshold<=1.)
    throw std::invalid_argument("invalid QP values or solver settings");
  if((H-H.transpose()).cwiseAbs().maxCoeff()>1e-9*std::max(1.,H.cwiseAbs().maxCoeff()))
    throw std::invalid_argument("H must be symmetric");
  for(Eigen::Index i=0;i<m;++i)
    if(std::isnan(l[i])||std::isnan(u[i])||l[i]>u[i]||l[i]==std::numeric_limits<double>::infinity()||
       u[i]==-std::numeric_limits<double>::infinity()) throw std::invalid_argument("invalid bounds");
  Vec x=warm_x.is_none()?Vec::Zero(n):warm_x.cast<Vec>();
  Vec physical_y=warm_y.is_none()?Vec::Zero(m):warm_y.cast<Vec>();
  if(x.size()!=n||physical_y.size()!=m||!x.allFinite()||!physical_y.allFinite())
    throw std::invalid_argument("invalid warm start");
  Metrics result;
  double preprocess_ms=0.,factor_ms=0.,iteration_ms=0.,check_ms=0.;
  double coordinate_transform_ms=0.,coordinate_factor_ms=0.;
  Vec physical_x=x;
  int factorizations=0,checks=0,iterations=0,rho_updates=0;
  bool converged=false;
  const bool transformed=preconditioner_shift>=0.;
  std::string status="maximum iterations";
  double cost_scale=1.;
  {
    py::gil_scoped_release release;
    auto ts=Clock::now();
    Vec rs(m),metric_rs(m);
    Vec norm_sq=Vec::Zero(m);
    for(Eigen::Index j=0;j<n;++j)norm_sq.array()+=A.col(j).array().square();
    for(Eigen::Index i=0;i<m;++i)metric_rs[i]=1./std::max(1.,std::sqrt(norm_sq[i]));
    if(objective_scaling)cost_scale=std::max({1.,H.cwiseAbs().maxCoeff(),infnorm(g)});
    Mat Hs=H/cost_scale, Awork=A, transform;
    Vec gs=g/cost_scale;
    if(transformed){
      // P = H/s + shift I defines ONLY an invertible coordinate transform.
      // The transformed objective below is T^T (H/s) T, never T^T P T.
      auto ft=Clock::now();
      Mat preconditioner=Hs;
      preconditioner.diagonal().array()+=preconditioner_shift;
      Eigen::LLT<Mat> cf(preconditioner);
      coordinate_factor_ms=elapsed(ft);
      factor_ms+=coordinate_factor_ms;
      if(cf.info()!=Eigen::Success)throw std::invalid_argument("preconditioner is not positive definite");
      auto ct=Clock::now();
      transform=cf.matrixU().solve(Mat::Identity(n,n));
      Hs=(transform.transpose()*Hs*transform).eval();
      Hs=(.5*(Hs+Hs.transpose())).eval();
      gs=(transform.transpose()*gs).eval();
      Awork.noalias()=A*transform.triangularView<Eigen::Upper>();
      x=(cf.matrixU()*x).eval();
      coordinate_transform_ms=elapsed(ct);
    }
    norm_sq.setZero();
    for(Eigen::Index j=0;j<n;++j)norm_sq.array()+=Awork.col(j).array().square();
    for(Eigen::Index i=0;i<m;++i){
      double norm=std::sqrt(norm_sq[i]);
      // Near-zero rows cannot be equilibrated without amplifying roundoff.
      rs[i]=row_scaling && norm>1e-10 ? 1./norm : 1.;
    }
    Mat As=rs.asDiagonal()*Awork;
    Vec ls=rs.cwiseProduct(l),us=rs.cwiseProduct(u);
    Vec ys=physical_y.cwiseQuotient(rs)/cost_scale;
    Vec z=As*x;
    for(Eigen::Index i=0;i<m;++i)z[i]=clipped(z[i]+ys[i]/rho,ls[i],us[i]);
    Mat gram=Mat::Zero(n,n);
    gram.selfadjointView<Eigen::Lower>().rankUpdate(As.transpose());
    gram=gram.selfadjointView<Eigen::Lower>();
    Mat system(n,n);
    Vec rhs(n),new_x(n),ax(m),zold(m),relaxed(m),hx(n),aty(n),work_m(m);
    Vec physical_ax(m),physical_hx(n),physical_aty(n);
    Eigen::LLT<Mat> factor;
    preprocess_ms=elapsed(ts)-coordinate_factor_ms-coordinate_transform_ms;
    auto refactor=[&](){
      auto ft=Clock::now();
      system=Hs+rho*gram;
      system.diagonal().array()+=sigma;
      factor.compute(system);
      factor_ms+=elapsed(ft);++factorizations;
      return factor.info()==Eigen::Success;
    };
    if(!refactor())status="factorization failed";
    else{
      for(int k=1;k<=max_iter;++k){
        auto it=Clock::now();
        work_m=rho*z-ys;
        rhs.noalias()=As.transpose()*work_m;
        rhs+=sigma*x-gs;
        new_x.noalias()=factor.solve(rhs);
        if(!new_x.allFinite()){status="nonfinite iterate";iteration_ms+=elapsed(it);break;}
        x.swap(new_x);
        ax.noalias()=As*x;
        zold=z;
        relaxed=alpha*ax+(1.-alpha)*zold;
        for(Eigen::Index i=0;i<m;++i)z[i]=clipped(relaxed[i]+ys[i]/rho,ls[i],us[i]);
        ys.noalias()+=rho*(relaxed-z);
        iterations=k;
        iteration_ms+=elapsed(it);
        if(k%check_interval==0 || k==max_iter){
          auto ct=Clock::now(); physical_y=cost_scale*rs.cwiseProduct(ys);
          if(transformed)physical_x.noalias()=transform*x;else physical_x=x;
          result=metrics(H,g,A,l,u,physical_x,physical_y,metric_rs,physical_ax,physical_hx,physical_aty);
          ++checks;check_ms+=elapsed(ct);
          if(result.error()<=tol && (primal_abs_tol<0 || result.primal_raw<=primal_abs_tol)){
            converged=true;status="solved";break;
          }
        }
        if(adaptive_interval>0 && k%adaptive_interval==0 && k<max_iter){
          auto ct=Clock::now();
          hx.noalias()=Hs*x;aty.noalias()=As.transpose()*ys;
          double rp=infnorm(ax-z)/std::max({1.,infnorm(ax),infnorm(z)});
          double rd=infnorm(hx+gs+aty)/std::max({1.,infnorm(hx),infnorm(gs),infnorm(aty)});
          double proposed=std::clamp(rho*std::sqrt(std::max(rp,1e-16)/std::max(rd,1e-16)),1e-6,1e6);
          check_ms+=elapsed(ct);
          if(proposed>rho*adaptive_threshold || proposed<rho/adaptive_threshold){
            rho=proposed;++rho_updates;
            if(!refactor()){status="factorization failed";break;}
          }
        }
      }
    }
    // Always return metrics for the actual returned primal/dual, even on failure.
    auto ct=Clock::now();physical_y=cost_scale*rs.cwiseProduct(ys);
    if(transformed)physical_x.noalias()=transform*x;else physical_x=x;
          result=metrics(H,g,A,l,u,physical_x,physical_y,metric_rs,physical_ax,physical_hx,physical_aty);
    check_ms+=elapsed(ct);
  }
  py::dict out;
  out["x"]=physical_x;out["y"]=physical_y;
  out["converged"]=converged;out["status"]=status;
  out["iterations"]=iterations;out["factorization_count"]=factorizations;
  out["rho_updates"]=rho_updates;out["rho_final"]=rho;out["checks"]=checks;
  out["preprocess_ms"]=preprocess_ms;out["factor_ms"]=factor_ms;
  out["coordinate_transform_ms"]=coordinate_transform_ms;out["coordinate_factor_ms"]=coordinate_factor_ms;
  out["coordinate_factorization_count"]=transformed?1:0;out["preconditioner_shift"]=preconditioner_shift;
  out["iteration_ms"]=iteration_ms;out["check_ms"]=check_ms;
  out["objective_scale"]=cost_scale;out["polish_ms"]=0.;
  out["qp_primal_raw"]=result.primal_raw;out["qp_primal_relative"]=result.primal_relative;
  out["qp_dual_raw"]=result.dual_raw;out["qp_dual_relative"]=result.dual_relative;
  out["qp_normal_relative"]=result.normal_relative;out["qp_objective"]=result.objective;
  out["total_ms"]=elapsed(total_start);
  return out;
}
PYBIND11_MODULE(structured_admm_native,m){
#ifdef ADMM_BLAS_ENABLED
  m.attr("blas_enabled")=true;
#else
  m.attr("blas_enabled")=false;
#endif
#ifdef ADMM_NATIVE_ARCH
  m.attr("native_arch_enabled")=true;
#else
  m.attr("native_arch_enabled")=false;
#endif

  m.doc()="Fixed-QP proximal ADMM with cached dense SPD factors and original-coordinate KKT checks.";
  m.def("solve",&solve,py::arg("H"),py::arg("g"),py::arg("A"),py::arg("l"),py::arg("u"),
    py::arg("warm_x")=py::none(),py::arg("warm_y")=py::none(),
    py::arg("tol")=2e-7,py::arg("primal_abs_tol")=-1.,py::arg("max_iter")=20000,
    py::arg("rho")=.1,py::arg("sigma")=1e-6,py::arg("alpha")=1.6,
    py::arg("check_interval")=10,py::arg("adaptive_interval")=50,
    py::arg("adaptive_threshold")=5.,py::arg("objective_scaling")=true,
    py::arg("row_scaling")=true,py::arg("preconditioner_shift")=-1.);
}
