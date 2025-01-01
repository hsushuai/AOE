import os
import json
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from skill_rts.agents import bot_ais


def get_ace_result(runs_dir):
    results = {}
    for bot in bot_ais.keys():
        result = [0, 0, 0]  # win, draw, loss
        for run in os.listdir(f"{runs_dir}/{bot}"):
            with open(f"{runs_dir}/{bot}/{run}/metric.json") as f:
                metric = json.load(f)
            if metric["win_loss"][0] == 1:
                result[2] += 1
            elif metric["win_loss"][0] == 0:
                result[1] += 1
            else:
                result[0] += 1
        results[bot] = [n / sum(result) for n in result]
    return results


def plot():
    ace_results = get_ace_result("runs/eval_ace")
    gridnet_results = {
        "guidedRojoA3N": [0.1, 0.1, 0.8],
        "randomBiasedAI": [0, 0, 1],
        "randomAI": [0, 0, 1],
        "passiveAI": [0, 0.0, 1],
        "workerRushAI": [0, 0, 1],
        "lightRushAI": [0, 0, 1],
        "coacAI": [1, 0, 0],
        "naiveMCTSAI": [0.1, 0.2, 0.7],
        "mixedBot": [0, 0, 1],
        "rojo": [0, 0, 1],
        "izanagi": [0.2, 0.4, 0.4],
        "tiamat": [0, 0, 1],
        "droplet": [0.1, 0, 0.9],
    }
    transformer_results = {
        "guidedRojoA3N": [0.2, 0.2, 0.6],
        "randomBiasedAI": [0, 0, 1],
        "randomAI": [0, 0, 1],
        "passiveAI": [0, 0, 1],
        "workerRushAI": [0, 0, 1],
        "lightRushAI": [0, 0, 1],
        "coacAI": [0, 0, 1],
        "naiveMCTSAI": [0.2, 0.1, 0.7],
        "mixedBot": [0.1, 0, 0.9],
        "rojo": [0, 0, 1],
        "izanagi": [0.2, 0, 0.8],
        "tiamat": [0, 0, 1],
        "droplet": [0.1, 0, 0.9],
    }

    ai_names = [
        "coacAI",
        "workerRushAI",
        "droplet",
        "mixedBot",
        "izanagi",
        "tiamat",
        "lightRushAI",
        "rojo",
        "guidedRojoA3N",
        "naiveMCTSAI",
        "randomBiasedAI",
        "passiveAI",
        # "randomAI",
    ]
    n_rows, n_cols = 4, 3
    plt.rcParams.update({"font.size": 17})
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 14), sharex=True, sharey=True)

    colors = ["#d86c50", "#0ac9bf", "#a39aef"]
    bar_width = 0.25

    for i in range(len(ai_names)):
        var_name = ai_names[i]
        ax = axes.flatten()[i]
        
        ace_result = ace_results.get(var_name, [0, 0, 0])
        gridnet_result = gridnet_results.get(var_name, [0, 0, 0])
        transformer_result = transformer_results.get(var_name, [0, 0, 0])

        x_pos = [0, 1, 2]
        
        ax.bar([x - bar_width for x in x_pos], ace_result, color=colors[0], width=bar_width, align="center", label="ACE", edgecolor="black", hatch="/")
        ax.bar([x for x in x_pos], gridnet_result, color=colors[1], width=bar_width, align="center", label="GridNet", edgecolor="black", hatch="\\")
        ax.bar([x + bar_width for x in x_pos], transformer_result, color=colors[2], width=bar_width, align="center", label="Transformer", edgecolor="black", hatch="x")
        
        ax.set_title(var_name)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(["loss", "tie", "win"])
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    
    plt.legend()
    fig.tight_layout()
    fig.savefig("ace/experiments/plots/against_bots.pdf")


if __name__ =="__main__":
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    plot()