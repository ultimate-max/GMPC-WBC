"""Shared target samples, CPU selection and timing statistics for active runners."""
import hashlib,os,pathlib
import numpy as np
from common_wbc_reference import CommonReference

class DenseTarget:
    def __init__(self,plant,root,case):
        out=root/'results/buffered_wbc';out.mkdir(parents=True,exist_ok=True)
        kind='double' if case=='double' else 'normal';path=out/f'target_{kind}_400hz.npz'
        self.dt=.0025
        if not path.exists():
            ref=CommonReference(plant,kind,12.5)
            samples=[ref.sample(k*self.dt) for k in range(5001)]
            np.savez_compressed(path,**{key:np.array([s[key] for s in samples]) for key in samples[0]})
        with np.load(path) as source:values={key:source[key] for key in source.files}
        if case=='hold':
            values['ee_pose'][:]=values['ee_pose'][0];values['ee_twist'][:]=0;values['ee_acc'][:]=0
        digest=hashlib.sha256()
        for key in sorted(values):digest.update(values[key].tobytes());values[key].flags.writeable=False
        self.hash=digest.hexdigest();self.samples=[{k:v[i] for k,v in values.items()} for i in range(5001)]
    def at(self,t):
        idx=round(t/self.dt)
        if abs(idx*self.dt-t)>1e-8 or not 0<=idx<len(self.samples):raise ValueError(f'Target off grid/range: {t}')
        return self.samples[idx]


def physical_cores():
    cores={}
    for cpu in sorted(os.sched_getaffinity(0)):
        path=pathlib.Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key=tuple((path/f).read_text().strip() for f in ('physical_package_id','core_id'))
        cores.setdefault(key,cpu)
    return list(cores.values())


def stats(x):
    x=np.asarray(x);return dict(mean=float(x.mean()),p95=float(np.quantile(x,.95)),p99=float(np.quantile(x,.99)),max=float(x.max())) if len(x) else None

