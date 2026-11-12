#!/usr/bin/env bash
set -euo pipefail
source /home/ganhe/project/b2z1_mpc/paper_repro/env.sh
mkdir -p results/paper_repro
python paper_repro/verify.py | tee results/paper_repro/model_verification.log
rho=${RHO:-200}
sigma=${SIGMA:-100}
input_prox=${INPUT_PROX:-0}
coupling=${COUPLING:-1}
printf '{"N":50,"prediction_dt":0.01,"control_dt":0.02,"duration":6,"repetitions":3,"condensing":50,"rho":%s,"sigma":%s,"input_prox":%s,"coupling":%s}\n' "$rho" "$sigma" "$input_prox" "$coupling" > results/paper_repro/config.json
for run in 1 2 3; do
  if [ "$run" = 2 ]; then order='distributed central'; else order='central distributed'; fi
  for mode in $order; do
    name=${mode}_trial${run}
    status=0
    paper_repro/build/paper_rti "$PWD" --mode "$mode" --profile single_step --duration 6 --name "$name" --condensing 50 --rho "$rho" --sigma "$sigma" --input-prox "$input_prox" --coupling "$coupling" > "results/paper_repro/$name.log" 2>&1 || status=$?
    printf '%s\n' "$status" > "results/paper_repro/$name.exit"
    tail -n 1 "results/paper_repro/$name.log"
  done
done
status=0
paper_repro/build/paper_rti "$PWD" --mode distributed --profile single_step --duration 6 --name distributed_serial_trial1 --threads 1 --condensing 50 --rho "$rho" --sigma "$sigma" --input-prox "$input_prox" --coupling "$coupling" > results/paper_repro/distributed_serial_trial1.log 2>&1 || status=$?
printf '%s\n' "$status" > results/paper_repro/distributed_serial_trial1.exit
python paper_repro/report.py > results/paper_repro/report_stdout.json
