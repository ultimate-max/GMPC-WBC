#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <chrono>
#include <string>
#include <vector>
using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;
using Sp = Eigen::SparseMatrix<double, Eigen::ColMajor, long long>;
using Triplet = Eigen::Triplet<double, long long>;
inline double now() {
  return std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline double infnorm(const Vec& v) { return v.lpNorm<Eigen::Infinity>(); }
struct QP { Mat P, A; Vec q, l, u; };
struct Solution { Vec x,y; double setup_ms=0, solve_ms=0, residual=0; int iterations=0,status=0; bool converged=false; };
inline double violation(const QP& p, const Vec& x) {
  const Vec ax = p.A*x;
  return std::max(0.0, std::max((p.l-ax).maxCoeff(), (ax-p.u).maxCoeff()));
}
inline double objective(const QP& p, const Vec& x) { return 0.5*x.dot(p.P*x)+p.q.dot(x); }
