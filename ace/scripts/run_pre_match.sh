#!/bin/bash

# Generate opponent
# python ace/pre_match/gen_opponent.py

# Battle
json_files=$(find ace/data/train/strategies -type f -name "*.json")
for json_file in $json_files; do
    filename=$(basename "$json_file")
    number=$(echo "$filename" | grep -oP '\d+(?=\.json)')
    numbers+=("$number")
done
strategy_indices=($(echo "${numbers[@]}" | tr ' ' '\n' | sort -n))

for i in "${strategy_indices[@]}"; do
    for j in "${strategy_indices[@]}"; do
        if [ "$i" -gt "$j" ]; then
            strategy="ace/data/train/strategies/strategy_${i}.json"
            opponent="ace/data/train/opponents/strategy_${j}.json"
            python "ace/pre_match/battle.py" --opponent "$opponent" --strategy "$strategy"
        fi
    done
done


# json_files=$(find ace/data/train/strategies -type f -name "*.json")

# numbers=()
# for json_file in $json_files; do
#     filename=$(basename "$json_file")
#     number=$(echo "$filename" | grep -oP '\d+(?=\.json)')
#     numbers+=("$number")
# done

# strategy_indices=($(echo "${numbers[@]}" | tr ' ' '\n' | sort -n))

# # Maximum parallel jobs
# max_parallel=8
# current_jobs=0

# for i in "${strategy_indices[@]}"; do
#     for j in "${strategy_indices[@]}"; do
#         if [ "$i" -gt "$j" ]; then
#             strategy="ace/data/train/strategies/strategy_${i}.json"
#             opponent="ace/data/train/opponents/strategy_${j}.json"
#             python ace/pre_match/battle.py --opponent "$opponent" --strategy "$strategy" &
            
#             ((current_jobs++))
            
#             # Wait if max_parallel jobs are running
#             if [ "$current_jobs" -ge "$max_parallel" ]; then
#                 wait -n # Wait for any job to finish
#                 ((current_jobs--))
#             fi
#         fi
#     done
# done

# # Wait for all remaining jobs to finish
# wait