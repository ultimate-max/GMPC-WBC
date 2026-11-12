#pragma once
#include "common.hpp"
#include <osqp.h>
class Solver {
 public:
  explicit Solver(const QP& p, double tolerance=1e-7,int max_iterations=20000);
  ~Solver();
  Solver(const Solver&)=delete;
  Solver& operator=(const Solver&)=delete;
  Solution solve(const Vec* gradient=nullptr,bool allow_iteration_limit=false);
  void warmStart(const Vec& x,const Vec* y=nullptr);
  void update(const QP& p);
  void budget(int iterations,double tolerance);
  double setup_ms;
 private:
  Sp P,A;
  OSQPSolver* solver=nullptr;
  double tolerance_;int max_iterations_;
  void setup(const QP& p);
};
