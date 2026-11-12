"""Read-only timing audit of explicit SE(3) primitives during QP assembly.

Replays stored states; changes no controller or simulation settings. Timed
wrappers include the Python/native call boundary. This is a microbenchmark,
not an end-to-end Cartesian-versus-geometric controller comparison.
"""
import hashlib
import json
import pathlib
import sys
import time

import numpy as np
import pinocchio as pin

root = pathlib.Path('/home/ganhe/project/b2z1_mpc')
sys.path.insert(0, str(root / 'relu_hgmpc'))
import native
from model import Reference
from split_gmpc import SplitGMPC

plant = native.Plant(str(root))
initial = plant.truth()
ref = Reference(plant, initial['q'], initial['pose'])
controller = SplitGMPC(plant, ref, root, mode='central')
trace = np.load(root / 'results/split_gmpc/distributed30_12s_trace.npz')
indices = np.linspace(0, len(trace['q']) - 1, 60, dtype=int)
states = [(plant.evaluate(trace['q'][i], trace['v'][i]), i * .02)
          for i in indices]

for state, t in states[:10]:
    controller.build(state, t)

originals = {name: getattr(pin, name) for name in ('log6', 'Jlog6', 'exp6')}
stats = {name: {'seconds': 0., 'calls': 0} for name in originals}

def wrapper(name, function):
    def measured(*args, **kwargs):
        start = time.perf_counter()
        result = function(*args, **kwargs)
        elapsed = time.perf_counter() - start
        stats[name]['seconds'] += elapsed
        stats[name]['calls'] += 1
        return result
    return measured

rows = []
try:
    for name, function in originals.items():
        setattr(pin, name, wrapper(name, function))
    for state, t in states:
        start = time.perf_counter()
        controller.build(state, t)
        rows.append(1000 * (time.perf_counter() - start))
finally:
    for name, function in originals.items():
        setattr(pin, name, function)
    controller.close()

result = dict(
    samples=len(states), nodes=controller.N,
    source_sha256=hashlib.sha256((root / 'relu_hgmpc/split_gmpc.py').read_bytes()).hexdigest(),
    build_mean_ms=float(np.mean(rows)), build_median_ms=float(np.median(rows)),
    primitives={name: dict(mean_ms_per_build=1000 * data['seconds'] / len(states),
                          calls_per_build=data['calls'] / len(states))
                for name, data in stats.items()},
    note='Replay of 60 stored closed-loop states, 10-node QP assembly only; '
         'native state evaluation, solver setup, solve and diagnostics excluded. '
         'Primitive timings include Python/native boundary but not all SE3 '
         'object operations or associated matrix products. Timing wrappers add overhead.')
result['explicit_primitives_total_ms_per_build'] = sum(
    value['mean_ms_per_build'] for value in result['primitives'].values())
result['explicit_primitives_fraction_of_build'] = (
    result['explicit_primitives_total_ms_per_build'] / result['build_mean_ms'])
output = root / 'results/chatter_cause/geometry_timing.json'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
