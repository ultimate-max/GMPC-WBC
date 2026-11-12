import os,time
import native
from shared_upper import SharedUpper
from threaded_online_mpc import clone_mux

class SharedWorker:
    def __init__(self,root,target,mode,cpu=None):
        if cpu is not None:os.sched_setaffinity(0,{cpu})
        self.plant=native.Plant(str(root));self.plant.fast_reference=True
        self.mpc=SharedUpper(self.plant,root,target,mode)
    def compute(self,s,mux,t,prev_tau,submitted_wall,release_delay=.02):
        wall=time.perf_counter();cpu=time.thread_time();mux=clone_mux(mux,self.plant,t)
        anchor=mux.state(t+release_delay)
        plan=self.mpc.solve(s,t+release_delay,prev_tau,anchor[2],anchor=anchor,first_slew_time=release_delay)
        info=dict(self.mpc.last_info)
        info.update(request_time=t,release_time=t+release_delay,
            worker_cpu_ms=1000*(time.thread_time()-cpu),worker_wall_ms=1000*(time.perf_counter()-wall),
            request_latency_ms=1000*(time.perf_counter()-submitted_wall),completed_wall=time.perf_counter())
        return plan,info
