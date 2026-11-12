import json,pathlib,subprocess,sys
import native
from run_online_wbc_mpc import Target
here=pathlib.Path(__file__).resolve().parent;root=here.parent;out=root/'results/shared_geometric_wbc'
out.mkdir(parents=True,exist_ok=True)
protocol=dict(cases=['normal','double','push'],duration=12.,repeats=2,upper_modes=['allocation','mpc'],clocks=['fixed','wall'],
    common=dict(upper_request_hz=50,wbc_hz=200,physics_hz=1000,release_delay=.02,blend=.04,
        hierarchy=['physical constraints','base SE3 pose','end-effector SE3 pose','joint posture / force / torque smoothing'],
        targets='Identical complete base, EE, joint posture trajectories; SHA256 verified per case across modes.',
        wbc='Identical CommonWBC/GeometricWBC source, gains, bounds, rolling contact, reference mux and advance_reference PD servo.',
        constraints='Torque .85 rated, slew 1000 leg/100 arm Nm/s, friction .6/sqrt(2), normal10..800N, joint margin .02rad, velocity min(rated,4), acceleration50 leg/30 arm. No extra hard reference-error envelope.',
        solver='Exact least squares + QR task locks; full physical/KKT certification; reduced positive-definite final task uses screened persistent OSQP/active-face certificate if needed. Constrained semidefinite high tasks use quadprog with original KKT gate2e-7.'),
    allocation='Adapted prioritized differential IK: current release target only, body velocity = transported reference velocity -10*log pose error. 18 scaled torque coordinates express contact-eliminated instantaneous velocity allocation over40ms; common physics constraints. Not the original arm-first SNS/activation/manipulability algorithm.',
    mpc='Strict hierarchical geometric frozen-linearization MPC,10 nodes x40ms,180 torque coordinates; exact task-output locks, future targets available.',
    interpretation='Controlled comparison of instantaneous prioritized allocation vs predictive hierarchy with identical execution. Not an exact replication comparison of Xie et al.; original prioritized.py preserved.',
    clock_definitions=dict(fixed='Offline ideal50Hz releases, waits for packet outside CPU timing; compares control at identical logical rate. Never claim real-time feasibility from this arm.',
        wall='Actual wall-paced200Hz loop and nonblocking50Hz requests, skipped slots/late packets counted. Shows implemented system behavior including computation latency.'),
    timing='Sum actual controller thread CPU: truth, reference generation, copies, construction, solve, validation, WBC. Target/cache/preflight separate. Physics integration/logging excluded. Wall thread times not added. PD/physics cost belongs to simulator, identical call.',
    jitter='Actual1kHz joint velocities/applied torques, Welch RMS5..100Hz and10..100Hz over2..12s and8..12s; spectral energy includes tracking and transients, not proof of limit-cycle chatter.',
    repetitions='Runs sequential, second repetition reverses method order. Wall scheduling nondeterminism; two repetitions are not statistical significance.',
    late_physics='MuJoCo still advances5x1ms each WBC tick; no hardware missed-tick torque-hold model.')
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
for case in ['normal','double']:Target(native.Plant(str(root)),case,12.,out)
records=[]
for clock in protocol['clocks']:
    for rep in [1,2]:
        for case in protocol['cases']:
            modes=protocol['upper_modes'] if rep==1 else protocol['upper_modes'][::-1]
            for mode in modes:
                name=f'{clock}_{case}_{mode}_r{rep}'
                command=[sys.executable,str(here/'run_shared_wbc.py'),'--clock',clock,'--backend',mode,'--case',case,'--name',name]
                try:result=subprocess.run(command,timeout=100);code=result.returncode
                except subprocess.TimeoutExpired:code='timeout100s'
                records.append(dict(name=name,returncode=code,command=command))
                (out/'run_ledger.json').write_text(json.dumps(records,indent=2))
                if code:print('FAILED',name,code,flush=True)
