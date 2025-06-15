#!/bin/bash

# ====================================
# Usage:
# nohup bash distill-sap/scripts/eval_distill.sh > run.log 2>&1 &
# ====================================

cd /mnt/opsstorage/xushuai/SAP
export PYTHONPATH=/mnt/opsstorage/xushuai/SAP

# Eval SAP-Distill
## Fight against AI Bot
bots=("randomAI" "randomBiasedAI" "guidedRojoA3N" "passiveAI" "workerRushAI" "lightRushAI" "coacAI" "mixedBot" "rojo" "izanagi" "tiamat" "droplet" "naiveMCTSAI" "mayari")
for opponent in "${bots[@]}"; do
    /mnt/opsstorage/xushuai/SAP/.venv/bin/python "/mnt/opsstorage/xushuai/SAP/distill-sap/eval_distill.py" --opponent "$opponent" --episodes 10 --max_steps 2000
done
