"""One planner worker owns independent native/Pinocchio data and OSQP state."""
import copy,time,os
import numpy as np
import native
from online_wbc_mpc import OnlineMPC

def clone_segment(seg,plant):
    if seg is None:return None
    result=copy.copy(seg);result.plant=plant;result.q0=seg.q0.copy();result.c=seg.c.copy()
    return result

def clone_plan(plan,plant):
    if plan is None:return None
    result=copy.copy(plan);result.segments=[clone_segment(x,plant) for x in plan.segments]
    result.torques=plan.torques.copy();result.forces=plan.forces.copy()
    return result

def clone_mux(mux,plant,now):
    result=copy.copy(mux);result.plant=plant;result.plan=clone_plan(mux.plan,plant)
    result.bridge=clone_segment(mux.bridge,plant);result.home=tuple(x.copy() for x in mux.home)
    # The main thread has already sampled this tick. Copy only its current
    # stance state; worker-side future propagation never touches main caches.
    key=max(x for x in mux.leg_times if x<=now+1e-10)
    result.leg_times=[key];result.leg_cache={key:mux.leg_cache[key].copy()}
    return result

class PlannerWorker:
    def __init__(self,root,target,backend,cpu=None):
        if cpu is not None:os.sched_setaffinity(0,{cpu})
        self.plant=native.Plant(str(root));self.plant.fast_reference=True
        self.mpc=OnlineMPC(self.plant,root,target,backend=backend,assembly='native')
    def compute(self,s,mux,t,prev_tau,submitted_wall,release_delay=.02):
        wall=time.perf_counter();cpu=time.thread_time()
        mux=clone_mux(mux,self.plant,t)
        anchor=mux.state(t+release_delay)
        plan=self.mpc.solve(s,t+release_delay,prev_tau,anchor[2],anchor=anchor,first_slew_time=release_delay)
        info=dict(self.mpc.last_info)
        info.update(request_time=t,release_time=t+release_delay,
            worker_cpu_ms=1000*(time.thread_time()-cpu),worker_wall_ms=1000*(time.perf_counter()-wall),
            request_latency_ms=1000*(time.perf_counter()-submitted_wall),
            completed_wall=time.perf_counter())
        return plan,info

def snapshot(s,mux,t):
    copied={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in s.items()}
    return copied,clone_mux(mux,None,t)
