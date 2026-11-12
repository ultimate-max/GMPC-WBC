#pragma once
#include "common.hpp"
#include <memory>
struct Snapshot {
  Mat M, Jfeet, Jtip;
  Vec h, torque_limit, velocity_limit, qmin_delta, qmax_delta, home_delta;
  Eigen::Vector3d tip;
  Vec feet,contact_error,contact_drift;
  double contact_kp=60,contact_kd=16;
};
struct SubsystemModel { Mat M,Jfeet; Vec h; std::vector<int> coordinates; std::string name; };
class Robot {
 public:
  explicit Robot(const std::string& root);
  ~Robot();
  Vec q, v, home;
  Snapshot snapshot();
  Vec integrate(const Vec& position, const Vec& delta);
  Vec difference(const Vec& from, const Vec& to);
  Vec rebaseVelocity(const Vec& old_q,const Vec& delta,const Vec& chart_velocity);
  void verify();
  std::vector<SubsystemModel> subsystems(const std::string& root,bool verbose=true);
  std::vector<std::string> jointNames() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
