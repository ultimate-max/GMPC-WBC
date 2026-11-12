# Shared geometric WBC comparison

This experiment isolates the upper controller more carefully than the earlier
`allocation_vs_online_mpc` experiment. Results and limitations are in
`../results/shared_geometric_wbc/RESULTS.md`.

Both upper controllers feed the **same** `CommonWBC` (SE(3), rigid rolling
contact, 200 Hz), `ReferenceMux` (C2 splice, nonlinear stance reconstruction),
and native `advance_reference` servo (MuJoCo 1 kHz). Priorities at both levels
are physical constraints > base pose > tool pose > posture/input smoothing.
The complete base, tool, posture targets and all numerical physics limits are
shared. The source and target hashes are stored per run.

The controlled allocation baseline is an adaptation: instantaneous prioritized
differential IK, with velocity command `transported Vd - 10 log(Td^-1 T)`.
Contact-eliminated dynamics express the feasible velocity increment using 18
torque variables. It reads only the release-time target; its 40 ms interval is
a command interval, not future task preview. It does **not** preserve the
original paper's arm-first SNS switching or manipulability third task. Original
`prioritized.py` and its WBC remain unchanged. Do not label this comparison an
exact replication or a direct superiority result over Xie et al.

The GMPC upper layer uses 10 x 40 ms future nodes, 180 torque variables and
strict lexicographic task locks. It uses the same frozen affine dynamics and
first-order geometric error construction as the earlier online MPC, with its
weighted base/tool objective replaced by actual priority levels.

`shared_hierarchy.py` uses exact least squares when all physical constraints and
KKT certify the solution. Pivoted QR gives task nullspaces. Constrained high
tasks use a regularized active-set candidate, accepted only against the original
unregularized KKT, with a feasible active-set correction when needed. The final
positive-definite reduced task uses `OnlineOSQP` structure reuse, constraint
generation and certified active-face prediction. It is a hybrid solver, not an
all-OSQP benchmark. Reduced primal coordinates rotate each update and are not
incorrectly reused as torque blocks.

Inherited near-zero constraint rows are checked in original physical units
before removal. Bound roundoff below 2e-7 is snapped only for the numerical
subproblem. Original KKT, full physical bounds, and all previous task outputs
are checked before releasing a plan. No error tolerance is increased to make a
run pass.

## Reproduce

From `/home/ganhe/project/b2z1_mpc`:

```bash
source relu_hgmpc/env.sh
python relu_hgmpc/test_shared_wbc.py
python relu_hgmpc/benchmark_shared_wbc.py
python relu_hgmpc/report_shared_wbc.py
```

To inspect one run, use `run_shared_wbc.py --backend allocation|mpc --clock
fixed|wall --case normal|double|push --name NAME`. Each experiment is 12 seconds
by default. Do not run timing experiments concurrently. Benchmark order is
reversed in repeat 2. The benchmark writes a ledger and enforces a 100-second
wall timeout per run.

`fixed` is an **offline** exact-50-Hz release experiment. It waits for a worker
packet outside controller CPU timing. `wall` is genuinely wall-paced with a
nonblocking worker, 50 Hz requested and skipped/late updates explicitly logged.
Only the latter evaluates implemented scheduling. Neither certifies hard real
time or emulates hardware torque holding after missed deadlines.

Total controller CPU sums the main and worker threads, including model/matrix
construction, reference propagation, copy, solve, validation and WBC. Simulator
integration/servo and logging are excluded; initialization and target caching
are separate. Jitter metrics use actual 1 kHz joint velocities/applied torques,
not optimized commands. High-frequency energy also includes transients.

Three tasks are normal, double-amplitude and a 120 N world-x push for 150 ms at
6 seconds. Incomplete runs are excluded from full-duration rankings. There is
no fallback controller after an infeasible/unaccepted problem; failures describe
this implementation and are not proofs that the original task is impossible.

`diagnose_shared_push.py` replays fixed-clock failures and checks the captured
linear feasible set independently. Diagnostic and pre-fix batches are retained
separately. Do not mix their hashes/results with the final batch.
