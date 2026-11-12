#pragma once
// Same frozen affine prediction and costs as online_wbc_mpc.py. No model
// approximation changes. Own all inputs before releasing the Python GIL.
py::dict assemble_online_mpc(const py::dict& input) {
  auto mat=[&](const char* key){return input[py::str(key)].cast<Mat>();};
  auto vec=[&](const char* key){return input[py::str(key)].cast<Vec>();};
  int N=input["N"].cast<int>(),n=18*N;
  double dt=input["dt"].cast<double>(),first=input["first"].cast<double>();
  Mat T=mat("T"),F=mat("F"),Aq=mat("Aq"),Av=mat("Av"),Fq=mat("Fq"),Fv=mat("Fv"),cone=mat("cone");
  Mat taskG=mat("taskG"),taskJ=mat("taskJ"),taskE=mat("taskE"),taskV=mat("taskV");
  Mat targetQ=mat("targetQ"),targetA=mat("targetA"),targetTau=mat("targetTau");
  Vec p=vec("p"),v=vec("v"),v0=vec("v0"),a0=vec("a0"),f0=vec("f0"),scale=vec("scale");
  Vec qjoint=vec("qjoint"),qmin=vec("qmin"),qmax=vec("qmax"),vmax=vec("vmax"),taumax=vec("taumax");
  Vec prev=vec("prev"),initialA=vec("initialA");
  Mat D=Mat::Zero(114*N,n),G=Mat::Zero(134*N,n);Vec d(114*N),lo(134*N),hi(134*N);
  {
    py::gil_scoped_release release;
    Mat P=Mat::Zero(24,n),V=P,old=Mat::Zero(18,n),oldA=Mat::Zero(24,n);
    Vec oldb=Vec::Zero(24);int cr=0,br=0;
    auto cost=[&](const Mat& C,const Vec& b,const Vec& weight){
      const int m=b.size();Vec w=weight.array().sqrt();
      D.middleRows(cr,m)=w.asDiagonal()*C;d.segment(cr,m)=w.array()*b.array();cr+=m;
    };
    auto bound=[&](const Mat& C,const Vec& b,const Vec& l,const Vec& u){
      const int m=b.size();G.middleRows(br,m)=C;lo.segment(br,m)=l-b;hi.segment(br,m)=u-b;br+=m;
    };
    Vec accel=Vec::Constant(18,30),rate=Vec::Constant(18,100);accel.head(12).setConstant(50);rate.head(12).setConstant(1000);
    Vec forceLo=Vec::Constant(20,-std::numeric_limits<double>::infinity()),forceHi=Vec::Zero(20);
    for(int f=0;f<4;++f){forceLo[5*f+4]=10;forceHi[5*f+4]=800;}
    Vec posWeight(6),velWeight(6),posture=Vec::Constant(18,40),trustBase(6);
    velWeight.head(3).setConstant(20);velWeight.tail(3).setConstant(8);posture.head(12).setConstant(5);
    trustBase.head(3).setConstant(.1);trustBase.tail(3).setConstant(.3);
    Vec zero=Vec::Zero(18),vlimit=vmax.cwiseMin(Vec::Constant(18,4));
    for(int k=0;k<N;++k){
      Mat U=Mat::Zero(18,n);U.middleCols(18*k,18).setIdentity();Mat tm=scale.asDiagonal()*U;
      Mat A=Aq*P+Av*V;A.middleCols(18*k,18)+=T;
      Vec b=a0+Aq*p+Av*(v-v0);
      Mat B=Fq*P+Fv*V;B.middleCols(18*k,18)+=F;
      Vec f=f0+Fq*p+Fv*(v-v0);
      bound(tm,zero,-.85*taumax,.85*taumax);
      bound(cone*B,cone*f,forceLo,forceHi);
      const double step=k==0?first:dt;
      bound(k==0?tm:Mat(tm-old),k==0?Vec(-prev):zero,-step*rate,step*rate);
      bound(A.bottomRows(18),b.tail(18),-accel,accel);
      P+=dt*V+.5*dt*dt*A;p+=dt*v+.5*dt*dt*b;V+=dt*A;v+=dt*b;
      bound(P.bottomRows(18),qjoint+p.tail(18),qmin.array() + .02,qmax.array() - .02);
      bound(P.bottomRows(18),p.tail(18),Vec::Constant(18,-.15),Vec::Constant(18,.15));
      bound(V.bottomRows(18),v.tail(18),-vlimit,vlimit);
      bound(P.topRows(6),p.head(6),-trustBase,trustBase);
      double terminal=k==N-1?3.:1.;
      for(int frame=0;frame<2;++frame){
        Mat Q=taskG.middleRows(12*k+6*frame,6),J=taskJ.middleRows(6*frame,6);
        posWeight.head(3).setConstant(4000);posWeight.tail(3).setConstant(frame==0?1600:1000);
        cost(Q*P,taskE.row(k).segment(6*frame,6).transpose()+Q*p,terminal*posWeight);
        cost(J*V,J*v-taskV.row(k).segment(6*frame,6).transpose(),velWeight);
      }
      cost(P.bottomRows(18),p.tail(18)-targetQ.row(k).transpose(),posture);
      cost(A.bottomRows(18),b.tail(18)-targetA.row(k).transpose(),Vec::Constant(18,.02));
      cost(k==0?Mat(A.bottomRows(18)):Mat(A.bottomRows(18)-oldA.bottomRows(18)),
        k==0?Vec(b.tail(18)-initialA.tail(18)):Vec(b.tail(18)-oldb.tail(18)),Vec::Constant(18,.02));
      cost(U,-targetTau.row(k).transpose().cwiseQuotient(scale),Vec::Constant(18,.002));
      cost(k==0?U:Mat(scale.cwiseInverse().asDiagonal()*(tm-old)),
        k==0?Vec(-prev.cwiseQuotient(scale)):zero,Vec::Constant(18,.03));
      old=tm;oldA=A;oldb=b;
    }
    if(cr!=D.rows()||br!=G.rows())throw std::runtime_error("Online MPC assembly row mismatch");
  }
  py::dict out;out["D"]=std::move(D);out["d"]=std::move(d);out["G"]=std::move(G);out["lo"]=std::move(lo);out["hi"]=std::move(hi);return out;
}
