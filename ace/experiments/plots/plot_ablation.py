import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ace.experiments.plots.plot_main_result import calculate_payoffs


def get_results():
    runs_dir = {
        "ace": "runs/eval_ace",
        "no_greedy": "runs/eval_no_greedy",
        "no_sen": "runs/eval_no_sen",
        "fixed": "runs/eval_fixed"
    }
    results = {}
    for run_name, run_dir in runs_dir.items():
        results[run_name] = {}
        for i in range(1, 31):
            opponent = f"strategy_{i}"
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
    df = pd.DataFrame(stat).T
    df.to_csv("ace/experiments/plots/ablation.csv")


def plot():
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    # Data
    seen = [0.882, 0.843, 0.824, 0.941]
    unseen = [1.000, 0.949, 0.974, 0.923]

    # Convert to percentages
    seen_percentage = [val * 100 for val in seen]
    unseen_percentage = [val * 100 for val in unseen]

    x = np.arange(2)  # the label locations (Seen and Unseen)
    width = 0.1  # the width of the bars

    # Plot
    fig, ax = plt.subplots()
    colors = ["#d86c50", "#0ac9bf", "#a39aef", "#f4cc71"]

    # Plot bars for each category
    ax.bar(x - 1.5 * width, [seen_percentage[0], unseen_percentage[0]], width, label="ACE", color=colors[0], edgecolor="black", hatch="/")
    ax.bar(x - 0.5 * width, [seen_percentage[1], unseen_percentage[1]], width, label="ACE w/o greedy", color=colors[1], edgecolor="black", hatch="x")
    ax.bar(x + 0.5 * width, [seen_percentage[2], unseen_percentage[2]], width, label="ACE w/o SEN", color=colors[2], edgecolor="black", hatch="\\")
    ax.bar(x + 1.5 * width, [seen_percentage[3], unseen_percentage[3]], width, label="ACE fixed", color=colors[3], edgecolor="black", hatch="-")

    # Add labels, title, and legend
    ax.set_ylabel("Win Rate (%)")
    ax.set_ylim(70, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(["Seen", "Unseen"])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=4, fontsize="small")

    # Save plot
    plt.tight_layout()
    plt.savefig("ace/experiments/plots/ablation.pdf")


if __name__ == "__main__":
    get_results()
    plot()
