#!/bin/bash

export PYTHONPATH="/root/desc/skill-rts/"

# Step 1 (Offline): Generate strategies
# python ace/offline/gen_strategies.py

# Step 2 (Offline): Battle
# json_files=$(find ace/data/train -type f -name "*.json")
# for json_file in $json_files; do
#     filename=$(basename "$json_file")
#     number=$(echo "$filename" | grep -oP '\d+(?=\.json)')
#     numbers+=("$number")
# done
# strategy_indices=($(echo "${numbers[@]}" | tr ' ' '\n' | sort -n))

# for i in "${strategy_indices[@]}"; do
#     for j in "${strategy_indices[@]}"; do
#         if [ "$i" -gt "$j" ]; then
#             strategy="ace/data/train/strategy_${i}.json"
#             opponent="ace/data/train/strategy_${j}.json"
#             python "ace/offline/battle.py" --opponent "$opponent" --strategy "$strategy"
#         fi
#     done
# done

# Step 3 (Offline): Payoff Network Training
# python ace/offline/payoff_net.py

# Step 4 (Online): Fight against 50 strategies
for i in {1..50}; do
    opponent="ace/data/strategies/strategy_${i}.json"
    python "ace/experiment/eval_ace.py" --opponent "$opponent"
done

# Step 5 (Online): Fight against AI Bot
bots=("randomAI" "randomBiasedAI" "guidedRojoA3N" "passiveAI" "workerRushAI" "lightRushAI" "coacAI" "mixedBot" "rojo" "izanagi" "tiamat" "droplet" "naiveMCTSAI" "mayari")

for opponent in "${bots[@]}"; do
    python "ace/experiment/eval_ace.py" --opponent "$opponent"
done

# Step 6 (Online): Fight against no strategy llm
for opponent in "Vanilla" "CoT" "PLAP"; do
    python "ace/experiment/eval_ace.py" --opponent "$opponent"
done
