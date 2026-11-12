import pathlib,cProfile,pstats
import native
from shared_upper import SharedUpper
from run_online_wbc_mpc import Target
from online_wbc_mpc import ReferenceMux,FeedbackWBC
root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/shared_geometric_wbc'
p=native.Plant(str(root));p.fast_reference=True;s=p.truth();mux=ReferenceMux(p,s)
target=Target(p,'normal',2.,out);mpc=SharedUpper(p,root,target,'mpc');wbc=FeedbackWBC(p,'se3')
pr=cProfile.Profile();pr.enable()
for k in range(30):
    anchor=mux.state(.02)
    mpc.solve(s,.02,wbc.prev_tau,anchor[2],anchor=anchor,first_slew_time=.02)
pr.disable();pstats.Stats(pr).sort_stats('tottime').print_stats(20)
