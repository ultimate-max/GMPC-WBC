#!/usr/bin/env bash
set -euo pipefail
source /home/ganhe/project/b2z1_mpc/paper_repro/env.sh
mkdir -p results/interface_audit
python paper_repro/verify.py > results/interface_audit/model_tests.log
python paper_repro/verify_interfaces.py > results/interface_audit/interface_tests.json
paper_repro/build/paper_rti "$PWD" --mode distributed --self-test 1 > results/interface_audit/runtime_tests.log
for run in 1 2 3; do
  name=stable_repeat${run}
  paper_repro/build/paper_rti "$PWD" --mode distributed --profile stable --duration 12 --name "$name" > "results/interface_audit/$name.log" 2>&1
  cp "results/paper_repro/$name.csv" results/interface_audit/
  tail -n 1 "results/interface_audit/$name.log"
done
name=stable_serial
paper_repro/build/paper_rti "$PWD" --mode distributed --profile stable --threads 1 --duration 6 --name "$name" > "results/interface_audit/$name.log" 2>&1
cp "results/paper_repro/$name.csv" results/interface_audit/
tail -n 1 "results/interface_audit/$name.log"
# A failing single-full-step experiment is retained, never replaced by a fallback.
status=0
paper_repro/build/paper_rti "$PWD" --mode distributed --profile single_step --duration 6 --name audited_single_step > results/interface_audit/audited_single_step.log 2>&1 || status=$?
printf '%s\n' "$status" > results/interface_audit/audited_single_step.exit
cp results/paper_repro/audited_single_step.csv results/interface_audit/
