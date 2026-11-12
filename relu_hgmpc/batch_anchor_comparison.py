import argparse,pathlib,subprocess,sys,json
ap=argparse.ArgumentParser();ap.add_argument('--duration',type=float,default=12.)
ap.add_argument('--cases',nargs='+',default=['normal','double','hold','push'])
ap.add_argument('--methods',nargs='+',default=['anchor','paper_shared','paper','direct'])
ap.add_argument('--tag',default='v3');args=ap.parse_args()
root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/anchored_mpc';out.mkdir(exist_ok=True,parents=True)
for case in args.cases:
    for method in args.methods:
        name=f'{args.tag}_{case}_{method}'
        with (out/f'{name}.log').open('w') as log:
            r=subprocess.run([sys.executable,str(pathlib.Path(__file__).with_name('run_anchor_comparison.py')),
                '--method',method,'--case',case,'--duration',str(args.duration),'--name',name],stdout=log,stderr=subprocess.STDOUT)
        path=out/f'{name}.json'
        if path.exists():
            d=json.loads(path.read_text());print(name,r.returncode,d['completed_seconds'],d['failure']['reason'] if d['failure'] else 'complete',flush=True)
        else:print(name,r.returncode,'missing result',flush=True)
