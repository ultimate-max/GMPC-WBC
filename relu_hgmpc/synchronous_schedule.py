"""One thread, integer WBC ticks, measured load and explicit frequency floor."""
import math
import numpy as np

class SynchronousSchedule:
    def __init__(self,wbc_hz=200.,initial_hz=50.,min_hz=50.,max_hz=100.,mode='budget',utilization=.8):
        if not 0<min_hz<=initial_hz<=max_hz<=wbc_hz:raise ValueError('Invalid rate ordering')
        self.wbc_hz=wbc_hz;self.dt=1./wbc_hz;self.mode=mode;self.utilization=utilization
        self.minimum_divisor=math.ceil(wbc_hz/max_hz)
        self.maximum_divisor=math.floor(wbc_hz/min_hz)
        self.divisor=round(wbc_hz/initial_hz)
        if abs(wbc_hz/self.divisor-initial_hz)>1e-8:raise ValueError('Initial MPC rate must divide WBC rate')
        self.next=0;self.last=None;self.mpc=[];self.wbc=[];self.next_review=round(.5*wbc_hz)
        self.fast_votes=0;self.events=[];self.over_budget=False;self.requested_divisor=self.divisor
    def due(self,k):return k>=self.next
    def submitted(self,k):self.last=k;self.next=k+self.divisor
    def observe(self,k,wbc_ms,mpc_ms=None):
        self.wbc.append(float(wbc_ms));self.wbc=self.wbc[-200:]
        if mpc_ms is not None:self.mpc.append(float(mpc_ms));self.mpc=self.mpc[-40:]
        if k<self.next_review or len(self.mpc)<3:return
        self.next_review=k+round(.5*self.wbc_hz)
        wbc95=float(np.quantile(self.wbc,.95));mpc95=float(np.quantile(self.mpc,.95))
        available=self.utilization*self.dt*1000-wbc95
        wanted=math.ceil(mpc95/available) if available>0 else self.maximum_divisor+1
        self.requested_divisor=wanted;self.over_budget=wanted>self.maximum_divisor
        candidate=int(np.clip(wanted,self.minimum_divisor,self.maximum_divisor))
        old=self.divisor
        if self.mode=='budget':
            if candidate>old:self.divisor=candidate;self.fast_votes=0
            elif candidate<old:
                self.fast_votes+=1
                if self.fast_votes>=3:self.divisor=old-1;self.fast_votes=0
            else:self.fast_votes=0
        if old!=self.divisor and self.last is not None:self.next=max(k+1,self.last+self.divisor)
        self.events.append(dict(tick=k,time=k*self.dt,old_divisor=old,divisor=self.divisor,
            requested_divisor=wanted,mpc_hz=self.wbc_hz/self.divisor,over_budget=self.over_budget,
            wbc_p95_ms=wbc95,mpc_p95_ms=mpc95))

def self_test():
    a=SynchronousSchedule();ticks=[]
    for k in range(100):
        if a.due(k):ticks.append(k);a.submitted(k)
    assert ticks==list(range(0,100,4))
    for k in range(100,401):a.observe(k,1.,20. if k%4==0 else None)
    assert a.divisor==4 and a.over_budget # respect explicit >=50 Hz requirement
    b=SynchronousSchedule(100.,50.,50.,100.)
    assert b.divisor==2
    c=SynchronousSchedule();ticks=[]
    for k in range(901):
        mpc=None
        if c.due(k):ticks.append(k);c.submitted(k);mpc=3.
        c.observe(k,1.,mpc)
    assert c.divisor==2 and min(np.diff(ticks))>=2
    assert all(e['mpc_hz']>=50 for e in c.events)
    for k in range(901,1201):c.observe(k,1.,20. if k%2==0 else None)
    assert c.divisor==4 and c.over_budget
    print('integer ticks, 50 Hz floor, and WBC-dependent divisor: PASS')
if __name__=='__main__':self_test()
