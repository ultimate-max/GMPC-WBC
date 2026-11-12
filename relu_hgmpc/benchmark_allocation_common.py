import json,pathlib,subprocess,sys,shutil
root=pathlib.Path(__file__).resolve().parents[1];here=root/'relu_hgmpc';out=root/'results/allocation_vs_online_mpc';out.mkdir(parents=True,exist_ok=True)
protocol=dict(cases=['normal','double','push'],modes=['allocation_legacy','allocation_rolling','mpc'],repeats=2,duration=12.,
    common='Identical tool pose/body-twist targets and 120N push; same robot/truth/contact geometry, effort and friction/normal-force bounds; 200Hz WBC and 1kHz physics.',
    differences='Complete-architecture comparison. Allocation updates at 200Hz, MPC at50Hz. Allocation internally selects base motion; MPC also receives external base and joint-posture references plus future tool targets. Hierarchies, acceleration/slew constraints and low-level q/v target generation differ.',
    purpose='Do not claim a same-QP solver acceleration, or rank base offsets as common-base tracking errors. Hysteresis baseline retained; original-contact and rolling-contact versions both reported.')
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
for rep in [1,2]:
    for case in protocol['cases']:
        modes=protocol['modes'] if rep==1 else list(reversed(protocol['modes']))
        for mode in modes:
            name=f'{case}_{mode}_r{rep}'
            if mode=='mpc':
                unique='allocation_cmp_'+name
                cmd=[sys.executable,str(here/'run_threaded_online_mpc.py'),'--case',case,'--name',unique]
            else:
                cmd=[sys.executable,str(here/'run_allocation_common.py'),'--case',case,'--contact',mode.removeprefix('allocation_'),'--name',name]
            r=subprocess.run(cmd,timeout=90)
            if mode=='mpc':
                for suffix in ['.json','.csv','_trace.npz']:
                    src=root/'results/online_wbc_mpc_threads'/f'{unique}{suffix}'
                    if src.exists():shutil.copy2(src,out/f'{name}{suffix}')
            if r.returncode:print('FAILED CASE',case,mode,rep,flush=True)
