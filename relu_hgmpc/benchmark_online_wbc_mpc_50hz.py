"""50 Hz acceptance plus a matched-latency historical 10 Hz ablation."""
import json,pathlib,subprocess,sys
from run_online_wbc_mpc import Target
import native

here=pathlib.Path(__file__).resolve().parent;root=here.parent
out=root/'results/online_wbc_mpc_50hz';out.mkdir(exist_ok=True,parents=True)
protocol=dict(cases=['normal','double','push'],policies=['fixed50','fixed10'],duration=12.,repeats=2,
    wbc_dt=.005,N=10,prediction_dt=.04,release_delay=.02,
    minimum_required_mpc_hz=50,periods={'fixed50':.02,'fixed10':.1},
    historical_ablation='fixed10 is a frequency ablation only and does not meet the requested minimum rate.',
    event_policy='Original 5-20 Hz event policy excluded. With one pending plan and a 20 ms release delay it cannot meaningfully accelerate updates beyond 50 Hz.',
    timing='Full planner wall time including future reference propagation; 20ms deadlines counted explicitly. Also audit 4 WBC cycles plus one MPC solve in each 20ms single-core compute window.',
    unchanged='Same targets, prediction grid/horizon, physical constraints, QP accuracy, quintic references and geometric WBC; only replan frequency differs.',
    scope='Logical serial simulation with 20ms release; not an OS asynchronous hard-real-time test.')
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
for case in ['normal','double']:Target(native.Plant(str(root)),case,12.,out)
for rep in [1,2]:
    for case in protocol['cases']:
        modes=protocol['policies'] if rep==1 else list(reversed(protocol['policies']))
        for mode in modes:
            result=subprocess.run([sys.executable,str(here/'run_online_wbc_mpc.py'),'--case',case,'--policy',mode,
                '--name',f'{case}_{mode}_r{rep}','--delay','.02','--output-dir','results/online_wbc_mpc_50hz'],
                check=False,timeout=240)
            if result.returncode:print('RECORDED FAILED CASE',case,mode,rep,flush=True)
