import json,pathlib,sys,collections
root=pathlib.Path(__file__).resolve().parents[1]
for name in sys.argv[1:]:
    r=json.loads((root/'results/buffered_wbc'/f'{name}.json').read_text())
    keys=('failure','completed_seconds','wall_seconds','achieved_wbc_hz','deadline_miss_fraction',
          'ee_position_error_mm','ee_rotation_error_deg','base_position_error_mm','controller_cpu_seconds',
          'control_wall_ms','wbc_ms','reference_ms','planner_wall_ms','installed_per_wall_second',
          'minimum_contacts','torque_clips','constraint_violation','task_lock_max',
          'leg_velocity','arm_velocity','leg_torque','arm_torque')
    print(name,json.dumps({k:r.get(k) for k in keys},indent=2))
    print('packet dispositions',collections.Counter(p['disposition'] for p in r['packets']))
