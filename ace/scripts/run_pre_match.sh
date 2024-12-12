#!/bin/bash

export PYTHONPATH="/root/desc/skill-rts/"

# Generate strategies
# python ace/pre_match/gen_strategies.py

# Battle
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
            python "ace/pre_match/battle.py" --opponent "$opponent" --strategy "$strategy"
        fi
    done
done
