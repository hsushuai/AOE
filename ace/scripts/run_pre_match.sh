#!/bin/bash

# Generate opponent
# python ace/pre_match/gen_opponent.py

# Battle
for i in {1..100}; do
    for j in {1..100}; do
        if [ "$i" -ge "$j" ]; then
            strategy="ace/data/strategies/strategy_${i}.json"
            opponent="ace/data/opponents/strategy_${j}.json"
            python "ace/pre_match/battle.py" --opponent "$opponent" --strategy "$strategy"
        fi
    done
done
