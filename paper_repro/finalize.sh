#!/usr/bin/env bash
set -euo pipefail
source /home/ganhe/project/b2z1_mpc/paper_repro/env.sh
mkdir -p results/paper_repro/diagnostics
for log in /tmp/b2z1_smoke_*.log; do
  [ ! -f "$log" ] || cp "$log" results/paper_repro/diagnostics/
done
python paper_repro/report.py > results/paper_repro/report_stdout.json
python paper_repro/audit_report.py
python - <<'PY'
from pathlib import Path
p=Path('README.md');text=p.read_text();marker='## acados/SQP-RTI reproduction status'
if marker not in text:
    text=marker+'\n\nThe new nonlinear paper-aligned implementation is in [paper_repro](paper_repro/README.md). Centralized closed-loop tests pass, but distributed reproduction is **not verified** and no speedup claim is valid. Read [the experiment report](results/paper_repro/REPORT.md). The OSQP prototype below is retained separately.\n\n'+text
    p.write_text(text)
PY
export_dir=/mnt/c/Users/whutj/Documents/Codex/2026-11-07/zotero-plugin-zotero-openai-curated-remote/outputs/b2z1_paper_reproduction
mkdir -p "$export_dir/source" "$export_dir/raw"
cp results/paper_repro/REPORT.md results/paper_repro/summary.json results/paper_repro/config.json results/paper_repro/environment.json results/paper_repro/source_sha256.json results/paper_repro/model_verification.log results/paper_repro/comparison.png results/paper_repro/comparison.pdf "$export_dir/"
cp results/paper_repro/*trial*.* "$export_dir/raw/"
cp paper_repro/*.py paper_repro/*.cpp paper_repro/*.hpp paper_repro/*.sh paper_repro/CMakeLists.txt paper_repro/README.md "$export_dir/source/"
