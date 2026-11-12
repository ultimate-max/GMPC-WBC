import json,pathlib,sys
import numpy as np
root=pathlib.Path(__file__).resolve().parents[1]/'results/online_wbc_mpc_threads'
for name in sys.argv[1:]:
    s=json.loads((root/f'{name}.json').read_text());a=np.genfromtxt(root/f'{name}.csv',delimiter=',',names=True)
    print(name,s['failed'],'skipped',s['skipped_requests'],'completed',s['simulated_cycles'])
    for key in ['worker_cpu_ms','worker_wall_ms','request_latency_ms','planner_ms','construction_ms','solve_ms','osqp_total_ms','osqp_rounds','osqp_constraints']:
        if key in s['plans'][0]:
            x=np.array([p[key] for p in s['plans']]);print(key,'mean',x.mean(),'p99',np.quantile(x,.99),'max',x.max())
    for key in ['main_wall_ms','main_cpu_ms','wbc_ms','start_lateness_ms','control_response_ms','base_position_error_mm','ee_position_error_mm']:
        print(key,'mean',a[key].mean(),'p99',np.quantile(a[key],.99),'max',a[key].max())
    print('deadline_misses',int(a['wbc_deadline_miss'].sum()))
