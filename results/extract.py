import os
import pandas as pd
import json


def extract_pre_match_runs_results():
    runs_dir = "runs/pre_match_runs"
    df = pd.DataFrame()
    runs = os.listdir(runs_dir)
    runs = sorted(runs, key=lambda x: int(x.split("_")[0]) * 100 + int(x.split("_")[1]))
    for run_name in runs:
        strategy = run_name.split("_")[0]
        opponent = run_name.split("_")[1]
        run_dir = os.path.join(runs_dir, run_name)
        with open(os.path.join(run_dir, "metric.json")) as f:
            metric = json.load(f)
        payoffs = list(
            map(
                lambda win_loss,
                damage_dealt,
                resource_spent,
                resource_harvested: win_loss * 10
                + damage_dealt * 0.1
                + (resource_spent - resource_harvested) * 0.1,
                metric["win_loss"],
                metric["damage_dealt"],
                metric["resource_spent"],
                metric["resource_harvested"],
            )
        )
        df.loc[strategy, opponent] = payoffs[0]
        df.loc[opponent, strategy] = payoffs[1]
    df = df.fillna(0)
    df = df.sort_index(key=lambda x: x.astype(int))
    df.to_csv("ace/data/payoff/payoff_matrix.csv")
