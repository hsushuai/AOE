import os
import pandas as pd
import json
import matplotlib.pyplot as plt


def extract_offline_runs_results():
    runs_dir = "runs/offline_runs"
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


def extract_online_results(runs_dir, output):
    match_results = {}
    for opponent in os.listdir(runs_dir):
        win_loss = [0, 0, 0]  # [win, draw, loss]
        for run in os.listdir(f"{runs_dir}/{opponent}"):
            filename = f"{runs_dir}/{opponent}/{run}/metric.json"
            if not os.path.exists(filename):
                continue
            with open(filename) as f:
                metric = json.load(f)
            if metric["win_loss"][0] == 1:
                win_loss[0] += 1  # win
            elif metric["win_loss"][0] == 0:
                win_loss[1] += 1  # draw
            else:
                win_loss[2] += 1  # loss
        match_results[opponent] = win_loss

    # Plot win-loss matrix
    os.environ["QT_QPA_PLATFORM"] = "offscreen"  # run without GUI
    
    num_opponents = len(match_results)
    cols = 5
    rows = (num_opponents // cols) + (1 if num_opponents % cols != 0 else 0)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    axes = axes.flatten()
    
    for idx, (opponent, win_loss) in enumerate(match_results.items()):
        wins, draws, losses = win_loss
        total = wins + draws + losses
        
        win_ratio = wins / total if total > 0 else 0
        draw_ratio = draws / total if total > 0 else 0
        loss_ratio = losses / total if total > 0 else 0
        
        ax = axes[idx]
        
        ax.set_ylim(0, 1)
        ax.bar(['Win', 'Draw', 'Loss'], [win_ratio, draw_ratio, loss_ratio], color=['green', 'yellow', 'red'])
        
        ax.set_title(f'{opponent}')
    
    for j in range(num_opponents, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig(output)
    print(f"Results saved to {output}")


if __name__ == "__main__":
    extract_online_results("runs/temperature1_vs_bots", "results/ace_vs_bots_temp1.pdf")
            