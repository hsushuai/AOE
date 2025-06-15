#!/bin/bash

export PYTHONPATH="/root/desc/skill-rts/"

# Generate strategies
# python sap/offline/gen_strategies.py

# Step 2 (Offline): Battle
# json_files=$(find sap/data/train -type f -name "*.json")
# for json_file in $json_files; do
#     filename=$(basename "$json_file")
#     number=$(echo "$filename" | grep -oP '\d+(?=\.json)')
#     numbers+=("$number")
# done
# strategy_indices=($(echo "${numbers[@]}" | tr ' ' '\n' | sort -n))

# for i in "${strategy_indices[@]}"; do
#     for j in "${strategy_indices[@]}"; do
#         if [ "$i" -gt "$j" ]; then
#             strategy="sap/data/train/strategy_${i}.json"
#             opponent="sap/data/train/strategy_${j}.json"
#             python "sap/offline/battle.py" --opponent "$opponent" --strategy "$strategy"
#         fi
#     done
# done

# Payoff Network Training
# python sap/offline/payoff_net.py

# # SAP  Fight against 50 strategies
# for i in {1..50}; do
#     opponent="sap/data/strategies/strategy_${i}.json"
#     python "sap/experiment/eval_ace.py" --opponent "$opponent"
# done

# # SAP Fight against AI Bot
# bots=("randomAI" "randomBiasedAI" "guidedRojoA3N" "passiveAI" "workerRushAI" "lightRushAI" "coacAI" "mixedBot" "rojo" "izanagi" "tiamat" "droplet" "naiveMCTSAI" "mayari")

# for opponent in "${bots[@]}"; do
#     python "sap/experiment/eval_ace.py" --opponent "$opponent"
# done

# # SAP Fight against no strategy llm
# for opponent in "Vanilla" "CoT" "PLAP"; do
#     python "sap/experiment/eval_ace.py" --opponent "$opponent"
# done

# Eval baselines
# python sap/experiment/eval_baseline.py  # strategy_27_vs_strategy_34

# Ablation

# # no greedy
# for i in {1..50}; do
#     opponent="sap/data/strategies/strategy_${i}.json"
#     python "sap/experiments/eval_no_greedy.py" --opponent "$opponent"
# done

# # no SEN
# for i in {1..50}; do
#     opponent="sap/data/strategies/strategy_${i}.json"
#     python "sap/experiments/eval_no_sen.py" --opponent "$opponent"
# done

# # fixed strategy
# for i in {1..50}; do
#     opponent="sap/data/strategies/strategy_${i}.json"
#     python "sap/experiments/eval_fixed.py" --opponent "$opponent"
# done

# # no tips
# for i in {1..50}; do
#     opponent="sap/data/strategies/strategy_${i}.json"
#     python "sap/experiments/eval_no_tips.py" --opponent "$opponent"
# done

# 16x16 offline battle
# json_files=$(find sap/data/train -type f -name "*.json")
# for json_file in $json_files; do
#     filename=$(basename "$json_file")
#     number=$(echo "$filename" | grep -oP '\d+(?=\.json)')
#     numbers+=("$number")
# done
# strategy_indices=($(echo "${numbers[@]}" | tr ' ' '\n' | sort -n))

# for i in "${strategy_indices[@]}"; do
#     for j in "${strategy_indices[@]}"; do
#         if [ "$i" -gt "$j" ]; then
#             strategy="sap/data/train/strategy_${i}.json"
#             opponent="sap/data/train/strategy_${j}.json"
#             python "sap/offline/battle.py" --opponent "$opponent" --strategy "$strategy"
#         fi
#     done
# done

# SAP Fight against 50 strategies
# for i in {1..50}; do
#     opponent="sap/data/strategies/strategy_${i}.json"
#     python "sap/experiments/eval_map_scaling.py" --opponent "$opponent" --max_steps 4000
# done

# # SAP Fight against AI Bot
# bots=("randomAI" "randomBiasedAI" "guidedRojoA3N" "passiveAI" "workerRushAI" "lightRushAI" "coacAI" "mixedBot" "rojo" "izanagi" "tiamat" "droplet" "naiveMCTSAI" "mayari")

# for opponent in "${bots[@]}"; do
#     python "sap/experiments/eval_map_scaling.py" --opponent "$opponent" --episodes 10 --max_steps 4000
# done

# SAP Fight against no strategy llm
# for opponent in "Vanilla" "CoT" "PLAP"; do
#     python "sap/experiments/eval_map_scaling.py" --opponent "$opponent" --max_steps 4000
# done

