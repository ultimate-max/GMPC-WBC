import pathlib,json
root=pathlib.Path(__file__).resolve().parents[1]
for file in sorted((root/'results/split_gmpc').glob('*.json')):
    s=json.loads(file.read_text())
    if 'cycles' not in s:continue
    fields=['failed','cycles','position_rmse_mm','rotation_rmse_deg','min_contacts','missing_contact_ms','max_slip_path_mm','torque_clips']
    d={k:s.get(k) for k in fields}
    for k in ['controller_ms','interface_scaled','dual_residual','full_dynamics_max','wrench_force_max','first_acceleration_defect']:
        d[k]=s.get(k)
    print(file.stem,json.dumps(d))
