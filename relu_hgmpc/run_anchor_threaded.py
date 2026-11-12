"""Use the existing two-core scheduler with the final anchored controller."""
import pathlib,os,json,hashlib
import pinocchio as pin
import native
import run_shared_wbc as driver
from shared_worker import SharedWorker
from shared_wbc import CommonWBC
from anchored_mpc import AnchoredMPC
from anchored_geometry import mount_transform,error_dynamics

class Worker(SharedWorker):
    def __init__(self,root,target,mode,cpu=None):
        if mode!='mpc':raise ValueError('This entry point tests anchored MPC only')
        if cpu is not None:os.sched_setaffinity(0,{cpu})
        self.plant=native.Plant(str(root));self.plant.fast_reference=True
        self.mpc=AnchoredMPC(self.plant,root,target)

class Execution(CommonWBC):
    def __init__(self,plant,mode):
        super().__init__(plant,'se3');self.ee_first=True
        root=pathlib.Path(__file__).resolve().parents[1]
        model=pin.buildModelFromUrdf(str(root/'models/b2_z1.urdf'),pin.JointModelFreeFlyer())
        self.C=mount_transform(model,plant.truth()['q'])
    def anchored_error(self,s,D,Vd,Ad):return error_dynamics(s,self.C,D,Vd,Ad)

if __name__=='__main__':
    driver.PlannerWorker=Worker;driver.FeedbackWBC=Execution
    try:driver.main()
    finally:
        import sys
        name=sys.argv[sys.argv.index('--name')+1]
        root=pathlib.Path(__file__).resolve().parents[1]
        path=root/'results/shared_geometric_wbc'/f'{name}.json'
        if path.exists():
            data=json.loads(path.read_text());data['actual_method']='final anchored MPC + EE-first anchored WBC'
            for name in ('run_anchor_threaded.py','anchored_mpc.py','anchored_geometry.py','anchor_qp.py','body_arm_tasks.py'):
                data['source_sha256'][name]=hashlib.sha256((root/'relu_hgmpc'/name).read_bytes()).hexdigest()
            path.write_text(json.dumps(data,indent=2))
