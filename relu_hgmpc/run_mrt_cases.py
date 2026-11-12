import pathlib,subprocess,sys,json
root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/buffered_wbc';status={}
for case in ['double','hold','push']:
    name='native_mrt400_'+case
    with (out/(name+'.log')).open('w') as f:
        r=subprocess.run([sys.executable,str(root/'relu_hgmpc/run_mrt_wbc.py'),'--duration','12','--case',case,'--name',name],stdout=f,stderr=subprocess.STDOUT)
    status[case]=r.returncode;print(case,r.returncode,flush=True)
print(json.dumps(status))
