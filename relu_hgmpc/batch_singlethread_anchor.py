import pathlib,subprocess,sys,json
root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/anchored_mpc'
jobs=[('single200_paper_shared','paper_shared','normal',200),('single200_paper','paper','normal',200),
      ('single100_normal','anchor','normal',100),('single200_hold','anchor','hold',200),
      ('single200_push','anchor','push',200),('single200_double','anchor','double',200),
      ('single200_repeat','anchor','normal',200)]
for name,method,case,rate in jobs:
    with (out/f'{name}.log').open('w') as log:
        result=subprocess.run([sys.executable,str(pathlib.Path(__file__).with_name('run_singlethread_anchor.py')),
            '--method',method,'--case',case,'--duration','12','--wbc-hz',str(rate),
            '--schedule','ratio' if method!='anchor' else 'budget','--name',name],stdout=log,stderr=subprocess.STDOUT)
    path=out/f'{name}.json'
    if path.exists():
        d=json.loads(path.read_text());print(name,result.returncode,d['completed_seconds'],d['failure']['reason'] if d['failure'] else 'complete',flush=True)
    else:print(name,result.returncode,'no result',flush=True)
