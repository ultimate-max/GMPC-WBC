"""Serial timing runs, alternate method order; stop on any failed run."""
import pathlib,subprocess,sys
here=pathlib.Path(__file__).resolve().parent
def run(*args):subprocess.run([sys.executable,str(here/args[0]),*args[1:]],check=True)
for case in ['normal','double','offset']:
    run('common_wbc_reference.py','--case',case)
for rep in [1,2]:
    for case in ['normal','double','offset']:
        for mode in (['product','se3'] if rep==1 else ['se3','product']):
            run('run_geometric_wbc.py','--case',case,'--mode',mode,'--name',f'{case}_{mode}_r{rep}')
