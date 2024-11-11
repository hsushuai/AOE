#!/bin/bash

# Generate opponent
# python ace/pre_match/gen_opponent.py

# Generate response strategies
for i in {1..50}; do
    opponent_file="ace/data/opponent/opponent_strategy_${i}.json"
    python ace/pre_match/gen_response.py --opponent_strategy "$opponent_file"
done
