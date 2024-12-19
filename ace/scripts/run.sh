#!/bin/bash

export PYTHONPATH="/root/desc/skill-rts/"

# Offline: Generate strategies
# python ace/offline/gen_strategies.py

# Offline: Battle
json_files=$(find ace/data/train -type f -name "*.json")
for json_file in $json_files; do
    filename=$(basename "$json_file")
    number=$(echo "$filename" | grep -oP '\d+(?=\.json)')
    numbers+=("$number")
done
strategy_indices=($(echo "${numbers[@]}" | tr ' ' '\n' | sort -n))

for i in "${strategy_indices[@]}"; do
    for j in "${strategy_indices[@]}"; do
        if [ "$i" -gt "$j" ]; then
            strategy="ace/data/train/strategy_${i}.json"
            opponent="ace/data/train/strategy_${j}.json"
            python "ace/offline/battle.py" --opponent "$opponent" --strategy "$strategy"
        fi
    done
done

# Offline: Payoff Network Training
# python ace/offline/payoff_net.py

# Online: Fight against 50 strategies
# for i in {1..50}; do
#     opponent="ace/data/strategies/strategy_${i}.json"
#     python "ace/main.py" --opponent "$opponent"
# done

# Online: Fight against AI Bot
# bots=("randomAI" "randomBiasedAI" "guidedRojoA3N" "passiveAI" "workerRushAI" "lightRushAI" "coacAI" "mixedBot" "rojo" "izanagi" "tiamat" "droplet" "naiveMCTSAI" "mayari")

# for bot in "${bots[@]}"; do
#     python "ace/main.py" --opponent "$bot" --temperature 0
# done
