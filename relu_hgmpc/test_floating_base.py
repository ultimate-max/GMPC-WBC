"""Check that additional instrumentation is passive and clocks/frames agree."""
import json
import pathlib
import numpy as np
from scipy.spatial.transform import Rotation
import native
from model import equilibrium

root = pathlib.Path(__file__).resolve().parents[1]
out = root / 'results/floating_base'
plants = [native.Plant(str(root)) for _ in range(2)]
plants[1].instrument = True
plants[1].instrument_base = True
snapshots = []
for k in range(10):
    a, b = [p.truth() for p in plants]
    assert np.array_equal(a['q'], b['q'])
    assert np.array_equal(a['v'], b['v'])
    snapshots.append(b)
    tau = equilibrium(a)
    for p in plants:
        p.advance(tau, np.zeros(24), 20)
assert np.array_equal(plants[0].truth()['q'], plants[1].truth()['q'])
assert np.array_equal(plants[0].truth()['v'], plants[1].truth()['v'])
base = plants[1].take_base_trace()
trace = plants[1].take_trace()
assert base.shape == (200, 20) and trace.shape == (200, 129)
assert np.array_equal(base[:, 0], trace[:, 0])
assert np.array_equal(base[::20, 1:8], np.array([s['q'][:7] for s in snapshots]))
assert np.array_equal(base[::20, 8:14], np.array([s['v'][:6] for s in snapshots]))
R = Rotation.from_quat(base[:, 4:8]).as_matrix()
for src, dst in [(slice(8, 11), slice(14, 17)), (slice(11, 14), slice(17, 20))]:
    assert np.max(np.abs(np.einsum('nij,nj->ni', R, base[:, src]) - base[:, dst])) < 1e-12
assert plants[0].take_base_trace().shape == (0, 20)
assert plants[1].take_base_trace().shape == (0, 20)
verification = dict(passive_instrumentation_bitwise=True, samples=200,
    independent_state_channel_exact=True, world_twist_transform_verified=True,
    timestamps_exact=True, legacy_129_column_trace_preserved=True)
smoke = root / 'results/floating_base_smoke/smoke_hgmpc_trace.npz'
old = root / 'results/hierarchical_gmpc/hierarchy_null_scale1_r1_trace.npz'
if smoke.exists() and old.exists():
    fresh, previous = np.load(smoke), np.load(old)
    n = len(fresh['q'])
    diffs = {key: float(np.max(np.abs(fresh[key] - previous[key][:n]))) for key in ['q', 'v', 'tau', 'acc']}
    assert max(diffs.values()) < 1e-9, diffs
    verification['existing_gmpc_replay_max_diff'] = diffs
(out / 'verification.json').write_text(json.dumps(verification, indent=2))
print(json.dumps(verification, indent=2))
