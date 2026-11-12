"""Consumer independence, immutable transfer and expired/out-of-order policy gates."""
import json,pathlib,time,sys
import numpy as np
import native
from model import equilibrium
from runtime_support import DenseTarget,physical_cores
from strict_gmpc import StrictGMPC
from native_runtime_support import kernel_inputs,publish
root=pathlib.Path(__file__).resolve().parents[1]
p=native.Plant(str(root));p.fast_reference=True;s=p.truth();previous=equilibrium(s)
target=DenseTarget(p,root,'normal');planner=StrictGMPC(p,root,target,joint_damping=5.)
plan=planner.solve(s,0.,previous,np.zeros(24),first_slew_time=.02)
b,a,lib=kernel_inputs(root)
def make():return native.ControlRuntime(str(root),b,a,lib,previous,.0025,True,False)
rt=make();q0=np.array([s.q0 for s in plan.segments]);c=np.vstack([s.c for s in plan.segments])
try:rt.publish(0,0.,.04,q0[:1],c,plan.torques,plan.forces)
except RuntimeError:pass
else:raise AssertionError('Malformed policy accepted')
rt.publish(0,0.,.04,q0,c,plan.torques,plan.forces);q0[:]=np.nan;c[:]=np.nan
obs=rt.observation();obs['q'][:]=np.nan;assert np.isfinite(rt.observation()['q']).all()
rt.start(.2,physical_cores()[0]);before=sys.getswitchinterval();sys.setswitchinterval(1.)
try:
    # Hold the GIL continuously on this thread. A Python WBC consumer could
    # not execute during this interval; the C++ consumer must keep running.
    end=time.perf_counter()+.12
    while time.perf_counter()<end:pass
finally:sys.setswitchinterval(before)
o=rt.observation();assert o['time']>.07,o
publish(rt,plan,-1)
while rt.running():time.sleep(.005)
data=rt.results();assert not data['failure'] and len(data['rows'])==80 and data['rejected']>=1,data['failure']
expired=make();publish(expired,plan,0);expired.start(.5,physical_cores()[0])
while expired.running():time.sleep(.01)
e=expired.results();assert e['failure']=='No unexpired MPC policy',e['failure']
assert .35<=len(e['rows'])*.0025<=.365
result=dict(passed=True,consumer_time_while_gil_held=o['time'],completed_independence_cycles=len(data['rows']),
    rejected_out_of_order=data['rejected'],expiry_stop_seconds=len(e['rows'])*.0025,
    checked=['invalid packet','copy isolation','C++ consumer advances while producer holds GIL','out-of-order rejection','horizon-expiry stop'])
(root/'results/buffered_wbc/native_runtime_tests.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
