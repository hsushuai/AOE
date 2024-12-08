#!/bin/bash

export PYTHONPATH="/root/desc/skill-rts/"


# Run in match
for i in {1..50}; do
    opponent="ace/data/opponents/strategy_${i}.json"
    python "ace/in_match/run.py" --opponent "$opponent"
done
