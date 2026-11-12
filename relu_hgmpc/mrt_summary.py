import json,pathlib,sys
root=pathlib.Path(__file__).resolve().parents[1]/'results/buffered_wbc'
for name in sys.argv[1:]:
    r=json.loads((root/(name+'.json')).read_text())
    keys=['failure','completed_seconds','achieved_wbc_hz','installed_per_wall_second','minimum_contacts','torque_clips','full_deadline_miss_fraction','planner_total_ms','full_loop_ms','ee_position_error_mm','ee_rotation_error_deg','base_position_error_mm','successful_plans','failed_plans','leg_velocity','arm_velocity','leg_torque','arm_torque']
    print(name,json.dumps({k:r.get(k) for k in keys},indent=2))
