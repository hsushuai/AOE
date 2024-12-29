import os
import json
from ace.experiments.plots.plot_main_result import calculate_payoffs


def get_results():
    runs_dir = {
        "no_greedy": "runs/eval_no_greedy",
        "no_sen": "runs/eval_no_sen",
        "fixed": "runs/eval_fixed"
    }
    results = {}
    for run_name, run_dir in runs_dir.items():
        results[run_name] = {}
        for opponent in os.listdir(run_dir):
            avg_payoffs = []
            win_loss = [0, 0, 0]  # loss, draw, win
            for run in os.listdir(f"{run_dir}/{opponent}"):
                with open(f"{run_dir}/{opponent}/{run}/metric.json") as f:
                    metric = json.load(f)
                payoffs = calculate_payoffs(metric)
                if metric["win_loss"][0] < 0:
                    win_loss[0] += 1
                elif metric["win_loss"][0] == 0:
                    win_loss[1] += 1
                else:
                    win_loss[2] += 1
                avg_payoffs.append(payoffs)
            results[run_name][opponent] = {
                "payoffs": [sum(d) / len(avg_payoffs) for d in zip(*avg_payoffs)],
                "win_rate": win_loss[2] / len(avg_payoffs)
            }
    
    stat = {}
    seen = [filename.split(".")[0] for filename in os.listdir("ace/data/train")]
    unseen = [filename.split(".")[0] for filename in os.listdir("ace/data/test")]
    for player in results.keys():
        seen_payoffs = []
        seen_win_rate = []
        unseen_payoffs = []
        unseen_win_rate = []
        for opponent in results[player].keys():
            if opponent in seen:
                seen_payoffs.append(results[player][opponent]["payoffs"][0])
                seen_win_rate.append(results[player][opponent]["win_rate"])
            elif opponent in unseen:
                unseen_payoffs.append(results[player][opponent]["payoffs"][0])
                unseen_win_rate.append(results[player][opponent]["win_rate"])
            else:
                raise ValueError(f"Unknown opponent: {opponent}")
        stat[player] = {
            "seen payoff": sum(seen_payoffs) / len(seen_payoffs),
            "seen win rate": sum(seen_win_rate) / len(seen_win_rate),
            "unseen payoff": sum(unseen_payoffs) / len(unseen_payoffs),
            "unseen win rate": sum(unseen_win_rate) / len(unseen_win_rate)
        }
    print(stat)

if __name__ == "__main__":
    get_results()
