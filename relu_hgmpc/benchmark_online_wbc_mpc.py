"""Serialized, paired runs. Thresholds fixed before this benchmark."""
import pathlib,subprocess,sys,json
from run_online_wbc_mpc import Target
import native
here=pathlib.Path(__file__).resolve().parent;root=here.parent;out=root/'results/online_wbc_mpc';out.mkdir(exist_ok=True,parents=True)
protocol=dict(cases=['normal','double','push'],policies=['fixed10','fixed5','event'],duration=12.,repeats=2,
    dt=.005,N=10,prediction_dt=.04,release_delay=.04,event_min=.05,event_max=.2,
    thresholds=dict(accel_linear=.5,accel_angular=2.,base_mm=2.,base_deg=.5,ee_mm=5.,ee_deg=2.,joint_rad=.03,margin=.08,trust_margin=.2),
    scope='MPC trajectory generation from committed release-state reference; measured dynamics and WBC feedback; not a full SQP-NMPC or OS real-time asynchronous deployment.')
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
for case in ['normal','double']:Target(native.Plant(str(root)),case,12.,out)
for rep in [1,2]:
    for case in protocol['cases']:
        modes=protocol['policies'] if rep==1 else list(reversed(protocol['policies']))
        for mode in modes:
            result=subprocess.run([sys.executable,str(here/'run_online_wbc_mpc.py'),'--case',case,'--policy',mode,
                '--name',f'{case}_{mode}_r{rep}','--delay','.04','--output-dir','results/online_wbc_mpc'],check=False,timeout=180)
            if result.returncode:print('RECORDED FAILED CASE',case,mode,rep,flush=True)
