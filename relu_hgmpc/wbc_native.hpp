#pragma once
#include "common.hpp"
#include <memory>

struct WState {
  Vec q,v,h,cdot,ce,jdot,qmin,qmax,vmax,taumax;
  Mat M,Jc,J,pose;
};
struct WReference {Vec q,v,a,tau,force,bv,ba,ev,ea;Mat bp,ep;};
struct WTask {Mat C;Vec d;};
struct WResult {
  Vec tau,acc,force,wrench;Mat G;Vec lo,hi;std::vector<WTask> tasks;
  double bound=0,lock=0,kkt=0,dynamics=0,contact=0,interface=0,body=0,arm=0;
  double assembly_ms=0,solve_ms=0,total_ms=0;
};
class NativeWbc {
 public:
  NativeWbc(const std::string& full,const std::string& body,const std::string& arm,
            const std::string& quadprog,Vec home,Vec previous,double dt,bool dual=true);
  ~NativeWbc();
  WResult control(const WState&,const WReference&);
  void setPrevious(const Vec&);
 private:
  struct Impl;std::unique_ptr<Impl> p;
};
