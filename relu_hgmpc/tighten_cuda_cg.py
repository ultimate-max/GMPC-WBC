"""Documented FP64 precision and serial multi-workspace lifetime fixes."""
from pathlib import Path
import subprocess
root=Path('/home/ganhe/project/b2z1_mpc/third_party/osqp_core_cuda100')
p=root/'include/public/osqp_api_constants.h'
s=p.read_text()
old='# define OSQP_CG_TOL_MIN    (1E-7)'
new='# define OSQP_CG_TOL_MIN    (1E-12)'
assert s.count(old)==1 or s.count(new)==1
p.write_text(s.replace(old,new))
p=root/'algebra/cuda/algebra_libs.cu';s=p.read_text()
if 'CUDA_handle_users' not in s:
    s=s.replace('CUDA_Handle_t *CUDA_handle = OSQP_NULL;',
        'CUDA_Handle_t *CUDA_handle = OSQP_NULL;\n/* Shared by serial workspaces; release only after the last owner. */\nstatic OSQPInt CUDA_handle_users = 0;')
    s=s.replace('if (CUDA_handle) return 0;', 'if (CUDA_handle) { ++CUDA_handle_users; return 0; }')
    s=s.replace('if (!CUDA_handle) return 1;\n  return 0;', 'if (!CUDA_handle) return 1;\n  CUDA_handle_users = 1;\n  return 0;')
    s=s.replace('if (!CUDA_handle) return;\n\n  cuda_free_libs', 'if (!CUDA_handle) return;\n  if (--CUDA_handle_users > 0) return;\n\n  cuda_free_libs')
    assert s.count('CUDA_handle_users')==4
    p.write_text(s)
diff=subprocess.check_output(['git','diff','--','include/public/osqp_api_constants.h','algebra/cuda/algebra_libs.cu'],cwd=root,text=True)
out=Path('/home/ganhe/project/b2z1_mpc/relu_hgmpc/cuda_fp64_precision.patch')
out.write_text(diff)
print(diff)
