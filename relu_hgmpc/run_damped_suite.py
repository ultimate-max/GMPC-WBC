import json,pathlib,subprocess,sys,time
root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/dual_wbc'
runs=[(m,c,f'{m.removeprefix("gmpc_")}_{c}_damped',1.)
      for c in ('normal','double','hold','push') for m in ('gmpc_dual','gmpc_unified')]
runs += [('paper_shared','normal','paper_shared_normal_damped',1.),
         ('gmpc_dual','normal','dual_normal_damped_repeat',1.),
         ('gmpc_unified','normal','unified_normal_damped_repeat',1.),
         ('gmpc_dual','normal','dual_normal_damped_no_soft_smoothing',0.)]
results=[]
for method,case,name,weight in runs:
    if (out/f'{name}.json').exists() and '--rerun' not in sys.argv:continue
    print('START',name,flush=True);start=time.perf_counter()
    with (out/f'{name}.log').open('w') as f:
        ret=subprocess.run([sys.executable,str(root/'relu_hgmpc/run_dual_wbc_comparison.py'),
            '--method',method,'--case',case,'--duration','12','--name',name,
            '--smoothing',str(weight),'--joint-damping','5'],cwd=root,stdout=f,stderr=subprocess.STDOUT)
    result=dict(name=name,exit_code=ret.returncode,elapsed_seconds=time.perf_counter()-start)
    results.append(result);print(result,flush=True)
    (out/'damped_suite_status.json').write_text(json.dumps(results,indent=2))
