import json,pathlib,subprocess,sys
import native
from run_online_wbc_mpc import Target
here=pathlib.Path(__file__).resolve().parent;root=here.parent;out=root/'results/online_wbc_mpc_threads'
protocol=dict(cases=['normal','double','push'],duration=12.,repeats=2,
    modes={'serial_quadprog':('serial','quadprog'),'thread_quadprog':('threaded','quadprog'),
           'thread_osqp':('threaded','osqp_screened')},
    mpc_hz=50,wbc_hz=200,physics_hz=1000,release_delay=.02,prediction_dt=.04,N=10,
    main_and_worker='Actual independent threads, distinct CPU core affinity, 0.1ms Python switch interval; C++ prediction assembly releases GIL.',
    solver='Predict preceding active face via equality-QP SVD; accept only full original KKT <=1e-10 and physical bounds <=1e-9. Otherwise OSQP Hessian whitening, persistent slots, horizon-shift warm start, full-constraint generation; common final original KKT <=2e-7.',
    fairness='All three modes use the same native assembly, task weights, physics, constraints, target and horizon. Python/native assembly equivalence verified separately.',
    timing='Wall-clock paced. Late results really arrive late and are spliced at next control tick. Busy planning slots explicitly skipped and counted. No future.result wait in the live control loop.',
    limitation='WSL/Python simulation; no hard-real-time guarantee. Repeated wall-clock schedules need not be bitwise identical.')
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
for case in ['normal','double']:Target(native.Plant(str(root)),case,12.,out)
for rep in [1,2]:
    for case in protocol['cases']:
        modes=list(protocol['modes'])
        if rep==2:modes.reverse()
        for mode in modes:
            execution,backend=protocol['modes'][mode]
            result=subprocess.run([sys.executable,str(here/'run_threaded_online_mpc.py'),
                '--execution',execution,'--backend',backend,'--case',case,'--name',f'{case}_{mode}_r{rep}'],timeout=90)
            if result.returncode:print('FAILED CASE',case,mode,rep,flush=True)
