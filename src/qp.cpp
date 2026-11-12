#include "qp.hpp"
#include <stdexcept>
Solver::Solver(const QP& p, double tolerance,int max_iterations):tolerance_(tolerance),max_iterations_(max_iterations) {setup(p);}
void Solver::setup(const QP& p) {
  double t=now();
  Mat upper=p.P.triangularView<Eigen::Upper>();
  P=upper.sparseView(1e-15,1.0); A=p.A.sparseView(1e-15,1.0);
  P.makeCompressed(); A.makeCompressed();
  OSQPCscMatrix cp,ca;
  OSQPCscMatrix_set_data(&cp,P.rows(),P.cols(),P.nonZeros(),P.valuePtr(),P.innerIndexPtr(),P.outerIndexPtr());
  OSQPCscMatrix_set_data(&ca,A.rows(),A.cols(),A.nonZeros(),A.valuePtr(),A.innerIndexPtr(),A.outerIndexPtr());
  OSQPSettings settings; osqp_set_default_settings(&settings);
  settings.verbose=0; settings.eps_abs=tolerance_; settings.eps_rel=tolerance_;
  settings.max_iter=max_iterations_; settings.check_termination=10; settings.polishing=0;
  // Iteration-based adaptation makes results independent of wall-clock timing.
  settings.adaptive_rho=1; settings.adaptive_rho_interval=50;
  int code=osqp_setup(&solver,&cp,p.q.data(),&ca,p.l.data(),p.u.data(),A.rows(),A.cols(),&settings);
  if(code) throw std::runtime_error("OSQP setup failed: "+std::to_string(code));
  setup_ms=now()-t;
}
Solver::~Solver() { if(solver) osqp_cleanup(solver); }
void Solver::budget(int iterations,double tolerance) {
  max_iterations_=iterations;tolerance_=tolerance;
  OSQPSettings settings=*solver->settings;settings.max_iter=iterations;settings.eps_abs=tolerance;settings.eps_rel=tolerance;
  if(osqp_update_settings(solver,&settings)) throw std::runtime_error("OSQP settings update failed");
}
void Solver::warmStart(const Vec& x,const Vec* y) {
  if(x.size()!=P.cols() || !x.allFinite() || (y && (y->size()!=A.rows() || !y->allFinite())))
    throw std::runtime_error("Invalid OSQP warm start");
  if(osqp_warm_start(solver,x.data(),y?y->data():nullptr)) throw std::runtime_error("OSQP warm start failed");
}
void Solver::update(const QP& p) {
  double t=now();Mat upper=p.P.triangularView<Eigen::Upper>();
  Sp nextP=upper.sparseView(1e-15,1.0),nextA=p.A.sparseView(1e-15,1.0);
  nextP.makeCompressed();nextA.makeCompressed();
  auto same=[](const Sp& a,const Sp& b) {
    return a.rows()==b.rows() && a.cols()==b.cols() && a.nonZeros()==b.nonZeros()
      && std::equal(a.outerIndexPtr(),a.outerIndexPtr()+a.outerSize()+1,b.outerIndexPtr())
      && std::equal(a.innerIndexPtr(),a.innerIndexPtr()+a.nonZeros(),b.innerIndexPtr());
  };
  if(same(P,nextP) && same(A,nextA)) {
    P=nextP;A=nextA;
    if(osqp_update_data_mat(solver,P.valuePtr(),nullptr,P.nonZeros(),A.valuePtr(),nullptr,A.nonZeros()) ||
       osqp_update_data_vec(solver,p.q.data(),p.l.data(),p.u.data())) throw std::runtime_error("OSQP data update failed");
  } else {
    // Zero entries can become nonzero as the robot moves. Never silently drop
    // them to preserve the old sparsity pattern. Rebuild and explicitly reseed.
    osqp_cleanup(solver);solver=nullptr;setup(p);
  }
  setup_ms=now()-t;
}
Solution Solver::solve(const Vec* gradient,bool allow_iteration_limit) {
  double t=now();
  if(gradient && osqp_update_data_vec(solver,gradient->data(),nullptr,nullptr))
    throw std::runtime_error("OSQP gradient update failed");
  int code=osqp_solve(solver);
  Solution r;
  r.solve_ms=now()-t; r.setup_ms=setup_ms;
  r.iterations=solver->info->iter;
  r.status=solver->info->status_val;
  r.converged=(code==0 && solver->info->status_val==OSQP_SOLVED);
  if(!r.converged && !(allow_iteration_limit && code==0 &&
      (r.status==OSQP_MAX_ITER_REACHED || r.status==OSQP_SOLVED_INACCURATE)))
    throw std::runtime_error(std::string("OSQP did not solve: ")+solver->info->status);
  r.x=Eigen::Map<Vec>(solver->solution->x,P.cols());
  r.y=Eigen::Map<Vec>(solver->solution->y,A.rows());
  if(!r.x.allFinite() || !r.y.allFinite()) throw std::runtime_error("OSQP nonfinite iterate");
  r.residual=solver->info->prim_res;
  return r;
}
