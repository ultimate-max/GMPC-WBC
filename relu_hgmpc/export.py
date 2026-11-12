from pathlib import Path
import sys,shutil
root=Path(__file__).resolve().parents[1];src=root/'results'/'relu_hgmpc'
dst=Path(sys.argv[1]);dst.mkdir(parents=True,exist_ok=True)
names=['RESULTS.md','tracking.png','tracking.pdf','timing.png','timing.pdf','verification.json','solver_verification.json','reproducibility.json']
for name in names:shutil.copy2(src/name,dst/name)
for path in src.glob('*_final_12s.*'):shutil.copy2(path,dst/path.name)
for name in ['README.md','FORMULATION.md']:shutil.copy2(root/'relu_hgmpc'/name,dst/name)
print('Exported verified experiment artifacts to',dst)
