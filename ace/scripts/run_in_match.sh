#!/bin/bash

export PYTHONPATH="/root/desc/skill-rts/"


# Fight against 50 strategies
for i in {1..50}; do
    opponent="ace/data/strategies/strategy_${i}.json"
    python "ace/in_match/run.py" --opponent "$opponent"
done

# Fight against AI Bot
# bots=("randomAI" "randomBiasedAI" "guidedRojoA3N" "passiveAI" "workerRushAI" "lightRushAI" "coacAI" "mixedBot" "rojo" "izanagi" "tiamat" "droplet" "naiveMCTSAI" "mayari")

# for bot in "${bots[@]}"; do
#     python "ace/in_match/run.py" --opponent "$bot" --temperature 0
# done
