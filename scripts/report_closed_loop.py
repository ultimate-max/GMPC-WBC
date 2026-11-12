"""Summarize measured closed-loop traces; do not infer missing/failed runs."""
from pathlib import Path
import csv
import json
import platform
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
out = root / "results/closed_loop"
labels = {"central":"Central OSQP (500)","admm5":"ADMM (5 x 100)",
          "admm10":"ADMM (10 x 100)","admm30":"ADMM (30 x 100)"}
colors = {"central":"#2463b4","admm5":"#21a079","admm10":"#e28c21","admm30":"#9a5bab"}
summary = {"environment":{"platform":platform.platform(),"horizon_nodes":8,"dt_s":0.04,
            "physics_dt_s":0.001,"state_feedback":"MuJoCo qpos/qvel ground truth; no contact-force oracle",
            "contact":"elliptic cone, mu=0.6, noslip_iterations=10, solref=[0.005,1], solimp=[0.99,0.999,0.0001,...]",
            "execution":"offline fixed simulation steps; computation overruns are recorded, not injected into plant",
            "timing":"model, horizon shift, QP assembly/update and solve; excludes bootstrap, diagnostics, logging and physics/1 kHz common servo"},"methods":{}}
records = {}
for key in labels:
    runs = []
    for rep in range(1,4):
        name=f"bench_{key}_{rep}"
        meta=out/f"{name}.json"
        if not meta.exists():
            continue
        with (out/f"{name}.csv").open() as f:
            rows=list(csv.DictReader(f))
        data={col:np.array([float(row[col]) for row in rows]) for col in rows[0]}
        assert len(rows)==150, f"Incomplete trace: {name}"
        runs.append((json.loads(meta.read_text()),data))
    if not runs:
        summary["methods"][key]={"completed_runs":0,"status":"failed_or_missing"}
        continue
    records[key]=runs
    get=lambda col:np.concatenate([r[1][col] for r in runs])
    ms=get("controller_ms")
    summary["methods"][key]={"completed_runs":len(runs),"cycles":len(ms),
        "mean_ms":float(ms.mean()),"p50_ms":float(np.quantile(ms,.5)),"p95_ms":float(np.quantile(ms,.95)),
        "max_ms":float(ms.max()),"over_40ms_fraction":float(np.mean(ms>40)),
        "tracking_rmse_mm":float(1000*np.sqrt(np.mean(get("error_m")**2))),
        "tracking_max_mm":float(1000*get("error_m").max()),
        "foot_drift_max_mm":float(1000*get("foot_drift_m").max()),
        "base_tilt_max_deg":float(np.rad2deg(get("tilt_rad")).max()),
        "all_contacts_fraction":float(np.mean(get("contacts")==4)),
        "torque_clip_count":int(get("torque_clips").sum()),
        "qp_violation_p95":float(np.quantile(get("qp_violation"),.95)),
        "qp_violation_max":float(get("qp_violation").max()),
        "consensus_primal_p95":float(np.quantile(get("primal"),.95)),
        "not_converged_fraction":float(np.mean(get("converged")==0)),
        "bootstrap_mean_ms":float(np.mean([r[0]["bootstrap_ms"] for r in runs])),
        "model_mean_ms":float(get("model_ms").mean()),"update_mean_ms":float(get("update_ms").mean()),
        "solve_mean_ms":float(get("solve_ms").mean())}
(out/"summary.json").write_text(json.dumps(summary,indent=2))
fig, axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for key,runs in records.items():
    d=runs[0][1];label=labels[key];color=colors[key]
    axes[0,0].plot(d["time"],1000*d["error_m"],label=label,color=color)
    ms=np.concatenate([r[1]["controller_ms"] for r in runs]);ordered=np.sort(ms)
    axes[0,1].plot(ordered,np.arange(1,len(ordered)+1)/len(ordered),label=label,color=color)
    axes[1,0].semilogy(d["time"],np.maximum(d["qp_violation"],1e-9),label=label,color=color)
    axes[1,1].plot(d["time"],1000*d["foot_drift_m"],label=label,color=color)
axes[0,0].set(title="Actual end-effector tracking",xlabel="Simulation time (s)",ylabel="Position error (mm)")
axes[0,1].set(title="Controller latency: 3 sequential runs",xlabel="Wall time per control cycle (ms)",ylabel="Empirical CDF")
axes[0,1].axvline(40,color="black",ls="--",lw=1,label="40 ms control period")
axes[1,0].set(title="Full-body QP constraint violation",xlabel="Simulation time (s)",ylabel="Infinity norm (scaled mixed units)")
axes[1,1].set(title="Standing foot-centre displacement",xlabel="Simulation time (s)",ylabel="Maximum displacement (mm)")
for ax in axes.flat:
    ax.grid(True,alpha=.25);ax.legend(fontsize=8)
fig.suptitle("B2 + Z1: warm-started, iteration-limited MPC in MuJoCo",fontsize=15)
fig.savefig(out/"comparison.png",dpi=180)
fig.savefig(out/"comparison.pdf")
fig2,axs=plt.subplots(3,1,figsize=(10,8),sharex=True,constrained_layout=True)
first=next(iter(records.values()))[0][1]
for i,axis in enumerate("xyz"):
    axs[i].plot(first["time"],first[f"target_{axis}"],"k--",label="Reference")
    for key,runs in records.items():
        d=runs[0][1];axs[i].plot(d["time"],d[f"tip_{axis}"],label=labels[key],color=colors[key])
    axs[i].set_ylabel(f"World {axis} (m)");axs[i].grid(alpha=.25)
axs[0].legend(ncol=3,fontsize=8);axs[0].set_title("End-effector positions from executed closed loops")
axs[-1].set_xlabel("Simulation time (s)");fig2.savefig(out/"tracking.png",dpi=180)
print(json.dumps(summary["methods"],indent=2))
