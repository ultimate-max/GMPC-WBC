import sys, importlib.util
sys.path.append('/home/ganhe/miniconda3/envs/frictionest/lib/python3.11/site-packages')
import numpy, torch
print('numpy',numpy.__version__,'torch',torch.__version__,'cuda',torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
x=torch.ones(3,device='cuda',dtype=torch.float64); print((x@x).item())
print('pin',importlib.util.find_spec('pinocchio'))
print('pybind headers',torch.__path__[0]+'/include')
