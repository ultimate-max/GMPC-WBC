# Paper-aligned B2 + Z1 SQP-RTI benchmark

**Current status: damped distributed closed loop validated for the recorded simulation task, single-full-step paper reproduction still NOT VERIFIED.** See [the interface/ADMM audit](INTERFACE_AUDIT.md) and [current results](../results/interface_audit/RESULTS.md). The stable profile passes three 12-second trials, a serial control, and a 60-second run. The earlier 0.66-second failure report in `results/paper_repro` is retained as historical evidence, not the current stable profile.

The distributed default is now `--profile stable`: 10 coordination rounds per control cycle, primal step 0.2, dual step 0.05. `--profile single_step` preserves one unrelaxed update and still fails. Both use acados SQP-RTI for each local call. Do not describe the stable profile as a reproduction of the paper's one-update speedup.

This is a new nonlinear benchmark; `src/closed_loop.cpp` remains the earlier frozen-linearization OSQP prototype. The two are not equivalent performance reproductions.

Reference: Amatucci et al., [Accelerating Model Predictive Control for Legged Robots through Distributed Optimization](https://arxiv.org/html/2403.11742v3), Section III and IV. Public code: [DWMPC](https://github.com/iit-DLSLab/DWMPC).

## Reproduction scope

Implemented target: 50 shooting intervals, 10 ms prediction step, 50 Hz control, semi-implicit integration, one SQP-RTI update per local solver call, parallel front/rear/arm blocks, lagged remote trajectories, eliminated interface wrench, scaled consensus duals, proximal state regularization. Centralized uses one call per cycle; the stable distributed profile uses ten coordination rounds per cycle. Centralized and distributed formulations share robot dynamics, task weights, contact geometry, torque/joint/friction bounds and the same MuJoCo execution layer.

Intentional differences from the paper: B2+Z1 rather than its robot; Pinocchio rather than ADAM; standing end-effector tracking rather than the paper's locomotion suite; world-expressed base twists rather than an unspecified interface convention; documented cost weights and proximal/consensus tuning because the complete original arm configuration is unavailable. This is a paper-aligned algorithm reproduction on a different platform, not an exact reproduction of its reported 4x result.

The public `generate_ocp.py` at the downloaded revision uses 15 nodes with nonuniform steps, a reduced block of the whole-body inertia, auxiliary contact force decisions, and softened constraints. It is not identical to the paper's published 50-node formulation. We follow the paper's stated physical subsystem formulation instead of silently treating the current repository as the exact experimental version.

## Dynamics and interface elimination

Each physical subsystem is partitioned into base and joint acceleration:

```
[A B; B' D] [a_base; a_joint] = [g_base + F; g_joint]
H = A - B solve(D, B')
b = g_base - B solve(D, g_joint)
H a_base = b + F
```

All twists and interface wrenches are expressed in the world frame at the common base origin. `partition.py` assigns one third of the shared `base_link` inertia to each virtual subsystem. Articulated branches are included exactly once; other shared fixed attachments retain the earlier front/rear split. No mass is duplicated. The canonical full robot and MuJoCo model are unchanged. This allocation is a documented implementation choice, not a recovered author parameter.

Eliminating joint acceleration and the internal wrenches with `sum(F)=0` gives the common next interface velocity

```
v_base_next = solve(sum_i H_i, sum_i (H_i v_base_i + dt b_i))
a_joint_i = solve(D_i, g_joint_i - B_i' (v_base_next-v_base_i)/dt).
```

The local solver differentiates its own terms. Other subsystems' `H` and momentum terms are numerical parameters evaluated on the previous iterate. This is a signed, dimensionally explicit reconstruction of the interface elimination underlying Eqs. (7)-(10), not a copy of unpublished original code. It avoids separately inverting each free-floating subsystem inertia. At a common state and controls, the resulting acceleration equals the complete Pinocchio forward dynamics; `verify.py` checks this at nonzero poses/velocities, not just at rest. Agreement at consensus does not establish convergence or equivalence away from consensus.

State is `[p_world, quaternion_xyzw, joints, v_world, omega_world, joint_rates]`. Pinocchio uses body twists internally, so both inertia and bias are transformed, including the derivative of the velocity transform. Quaternion integration is multiplicative; the position step is world translation. The body posture cost uses the SO(3) logarithm.

## Optimization

Central stage dimensions: 49 state + 30 input = 79, excluding contact slacks. Each leg block: 25 + 12 = 37. Arm: 25 + 6 = 31. There are 12 total no-slip slacks centrally and 6 per leg subsystem. There are no interface-wrench decision variables.

acados uses Gauss-Newton SQP-RTI, HPIPM with 50 retained blocks (no horizon reduction), a 50-iteration QP cap, 1e-6 QP tolerances, and 1e-6 Levenberg-Marquardt regularization. Both controllers use the same options. The generated solver initially specifies 10 blocks; the C++ runtime explicitly changes this to the recorded 50-block benchmark setting before solving. Each solver is single-threaded; the three local solves use OpenMP with independent capsules. Python is used only for offline model/code generation and reporting. The control loop and physical simulation run in C++.

The sum of local task costs equals the central task cost at consensus. Body cost is divided by three; joint/contact/tool costs belong to their respective subsystem. Control regularization is centered on a common feasible static feed-forward torque/contact-force reference. The default distributed extra stage cost is `0.5*200*||v_i-(v_j_old+v_k_old)/2+d_i||² + 0.5*100*||x-x_previous||²`. The aggregate edge multiplier updates as `d_i += dual_step*(v_i_new-(v_j_new+v_k_new)/2)`; 200 is the aggregate weight, twice the individual edge weight. acados scales running costs by 0.01; terminal costs are not time-scaled. At the terminal node only the state proximal weight 100 remains; the consensus weight is explicitly zero, consistent with the running sum in Eq. (12). These weights have not been recovered from the original experiment; the effective trajectory proximal matrix weights running states by 1 and terminal states by 100. Local base trajectories remain separate between coordination rounds.

No-slip velocity constraints use quadratic and linear slack penalties; foot-center ground-height equality is enforced at future running nodes, omitted at the measured initial node to avoid imposing exact rigid ground on a compliant simulator. Terminal contact constraints are omitted in both formulations. Coulomb friction uses the same conservative inscribed pyramid, coefficient 0.6, and normal-force cap 800 N. Joint bounds and effort limits come from the URDF. These choices are shared between controllers and must be distinguished from undisclosed original settings.

Warm starts shift state/input predictions and consensus duals by two prediction nodes per 20 ms control update. State/control tails are held; new running-node duals and the unconstrained terminal dual are zeroed. The previous implementation incorrectly updated/held terminal multipliers. The default now uses pairwise interface residuals via an algebraically equivalent aggregated edge multiplier, retaining separate local base trajectories. See the audit for the target and dual formulas; `--admm-form consensus` retains the old averaging variant for diagnosis. Initialization stops when interface L2 is below 1e-3 and full dynamics defect L2 below 1e-4, subject to an iteration cap; its timing is reported separately.

The validated stable profile selects `--relax 0.2 --dual-step 0.05 --outer 10`. Other diagnostic options remain disabled: `--input-prox` adds a scaled-input proximal term, `--coupling 0` uses continuous acceleration elimination instead of velocity projection, `--rollout 1` rebuilds the warm trajectory through complete dynamics, and `--height-tolerance` relaxes the foot-height equality. Those alternatives are experiments, not validated fixes or original-paper features. Earlier `/tmp` logs copied into `diagnostics/` are development records and must not be pooled with the frozen three-trial benchmark.

## Metrics and fairness

- `max_local_solve_ms`: acados `time_tot`, maximum across local solvers per coordination round, summed across all rounds, including RTI preparation and feedback. Centralized uses its single solver's same timer.
- `parallel_wall_ms`: wall time of the parallel solve region including solution extraction and scheduling.
- `controller_ms`: wall time for horizon shift, remote dynamics evaluation, reference/parameter update, solver execution, dual update and plan merging. Ground-truth reading, diagnostics, CSV I/O, physics stepping and the common 1 kHz servo are excluded for both controllers.
- `interface_l2`: square root of the sum of squared world-twist differences over all three unordered subsystem pairs and all 51 state nodes. No scaling, infinity norm, or division by node count is applied. `interface_max` gives the maximum individual pair/node norm. The paper does not fully specify its aggregation convention, so equality of numerical thresholds cannot be assumed.
- `full_defect_l2/max`: recompute the complete nonlinear whole-body step on the merged distributed trajectory or the central trajectory, then compare with the next predicted state. Both controllers use exactly the same function. This mixed-unit state norm complements, and does not replace, interface feasibility.
- Tracking error, stance contacts, base tilt, foot displacement, solver status and torque clipping are logged alongside timing. A successful RTI status is not proof of converged nonlinear KKT conditions.

MuJoCo uses true configuration and velocity. Both controllers share the same joint servo and model-based Z1 damping/friction compensation. Simulated time advances independently of compute time, so deadline misses are reported; this is not a hardware real-time or latency-injected validation.

## Commands

All commands run in WSL under `/home/ganhe/project/b2z1_mpc`:

```bash
bash paper_repro/setup.sh
bash paper_repro/build.sh
source paper_repro/env.sh
paper_repro/build/paper_rti "$PWD" --mode central --duration 6 --name central_trial
paper_repro/build/paper_rti "$PWD" --mode distributed --duration 6 --name distributed_trial --threads 3
```

Raw logs are in `results/paper_repro`. Keep the dependency revision manifest with benchmark outputs. Timing is environment-sensitive and must be measured on the same machine with no concurrent builds. Neither a speedup nor numerical agreement with the original paper is assumed in advance.

Run the historical single-step experiment with `bash paper_repro/benchmark.sh`. It retains per-run exit codes even when a controller fails. `summary.json` reports `completed`, `tracking_valid`, abort events, and explicitly invalidates speedup when either controller fails. A completed benchmark script is not evidence that both controllers passed.

The older `benchmark.sh` explicitly selects `single_step` to avoid silently substituting the stable configuration. Current stability verification is `bash paper_repro/validate_stability.sh`; after that finishes, run `bash paper_repro/compare_stable.sh` alone for timing and `python paper_repro/report_stability.py` for the current report. Total timing sums all coordination rounds. Centralized uses one worker; distributed uses three independent solver workers. The stable profile has finite prediction defects and different tracking accuracy from centralized RTI; timing is a configuration comparison, not a matched-optimality solver benchmark.
