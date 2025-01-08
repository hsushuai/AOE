import matplotlib.pyplot as plt
import numpy as np
import os
import json
from tqdm.rich import tqdm
from skill_rts.game import Trajectory
import matplotlib.patches as mpatches

os.environ["QT_QPA_PLATFORM"] = "offscreen"
plt.rcParams.update({"font.size": 17})
actions = ["attack", "harvest", "return", "produce"]

def get_action_distribution():
    data = {
        "SAP": [],
        "Vanilla": [],
        "CoT": [],
        "PLAP": []
    }

    # Vanilla, CoT, PLAP
    baseline = ["Vanilla", "CoT", "PLAP"]
    for method in baseline:
        for match in tqdm(os.listdir("runs/eval_baseline"), desc=method):
            if method in match:
                player_id = 0 if method == match.split("_vs_")[0] else 1
                traj = Trajectory.load(f"runs/eval_baseline/{match}/traj.json")
                action_stat = np.zeros(len(actions))
                for gs in traj:
                    for unit in gs:
                        if unit is not None and unit.owner == player_id and unit.action in actions:
                            action_stat[actions.index(unit.action)] += 1
                # action_stat /= action_stat.sum()
                action_stat /= traj.get_gametime()
                data[method].append(action_stat.tolist())
        
        # against ACE
        for run in os.listdir(f"runs/eval_ace/{method}"):
            action_stat = np.zeros(len(actions))
            traj = Trajectory.load(f"runs/eval_ace/{method}/{run}/traj.json")
            for gs in traj:
                for unit in gs:
                    if unit is not None and unit.owner == 1 and unit.action in actions:
                        action_stat[actions.index(unit.action)] += 1
            # action_stat /= action_stat.sum()
            action_stat /= traj.get_gametime()
            data[method].append(action_stat.tolist())
    
    # ACE
    for match in tqdm(os.listdir("runs/eval_ace"), desc="SAP"):
        if match not in baseline and "strategy" not in match:
            continue
        for run in os.listdir(f"runs/eval_ace/{match}"):
            traj = Trajectory.load(f"runs/eval_ace/{match}/{run}/traj.json")
            action_stat = np.zeros(len(actions))
            for gs in traj:
                for unit in gs:
                    if unit is not None and unit.owner == 0 and unit.action in actions:
                        action_stat[actions.index(unit.action)] += 1
            # action_stat /= action_stat.sum()
            action_stat /= traj.get_gametime()
            data["SAP"].append(action_stat.tolist())
    
    with open("ace/experiments/plots/action_distribution.json", "w") as f:
        json.dump(data, f)


def plot():
    with open("ace/experiments/plots/action_distribution.json", "r") as f:
        data = json.load(f)

    plt.figure()
    colors = ["#d86c50", "#0ac9bf", "#a39aef", "#f4cc71"]
    methods = list(data.keys())
    num_methods = len(methods)

    for j, action in enumerate(actions, start=1):
        positions = [j + (i - num_methods / 2) * 0.2 for i in range(num_methods)]
        stats = [[data[method][trial][j - 1] * 100 for trial in range(len(data[method]))] for method in methods]
        
        bplot = plt.boxplot(stats, positions=positions, widths=0.15, patch_artist=True, showfliers=True)
        
        for patch, color in zip(bplot["boxes"], colors):
            patch.set_facecolor(color)
        
        for median in bplot["medians"]:
            median.set_linestyle(":")
            median.set_color("black")
    
    legend_patches = [mpatches.Patch(facecolor=colors[i], label=method, edgecolor="black") for i, method in enumerate(methods)]
    plt.legend(handles=legend_patches, loc="upper right")

    plt.ylabel("Action Distribution")
    # plt.yticks(np.arange(0, 120, 20))
    plt.xticks(ticks=range(1, len(actions) + 1), labels=actions)
    
    plt.tight_layout()
    plt.savefig("ace/experiments/plots/action_distribution.pdf")


def get_metric_data():
    data = {
        "SAP": [],
        "Vanilla": [],
        "CoT": [],
        "PLAP": []
    }

    # Vanilla, CoT, PLAP
    baseline = ["Vanilla", "CoT", "PLAP"]
    for method in baseline:
        for match in os.listdir("runs/eval_baseline"):
            if method in match:
                player_id = 0 if method == match.split("_vs_")[0] else 1
                with open(f"runs/eval_baseline/{match}/metric.json", "r") as f:
                    metric = json.load(f)
                resource_harvested = metric["resource_harvested"][player_id] / (metric["game_time"])
                unit_produced = sum(metric["unit_produced"][player_id].values()) / (metric["game_time"])
                damage_dealt = metric["damage_dealt"][player_id] / (metric["game_time"])
                damage_taken = metric["damage_taken"][player_id] / (metric["game_time"])  
                data[method].append([damage_dealt, damage_taken, resource_harvested, unit_produced])
        
        # against ACE
        for run in os.listdir(f"runs/eval_ace/{method}"):
            with open(f"runs/eval_ace/{method}/{run}/metric.json", "r") as f:
                metric = json.load(f)
            resource_harvested = metric["resource_harvested"][1] / (metric["game_time"])
            unit_produced = sum(metric["unit_produced"][1].values()) /( metric["game_time"])
            damage_dealt = metric["damage_dealt"][1] / (metric["game_time"])
            damage_taken = metric["damage_taken"][1] / (metric["game_time"])            
            data[method].append([damage_dealt, damage_taken, resource_harvested, unit_produced])
    
    # ACE
    for match in os.listdir("runs/eval_ace"):
        if match not in baseline and "strategy" not in match:
            continue
        for run in os.listdir(f"runs/eval_ace/{match}"):
            with open(f"runs/eval_ace/{match}/{run}/metric.json", "r") as f:
                metric = json.load(f)
            resource_harvested = metric["resource_harvested"][0] / (metric["game_time"])
            unit_produced = sum(metric["unit_produced"][0].values()) /( metric["game_time"])
            damage_dealt = metric["damage_dealt"][0] / (metric["game_time"])
            damage_taken = metric["damage_taken"][0] / (metric["game_time"])
            data["SAP"].append([damage_dealt, damage_taken, resource_harvested, unit_produced])

    with open("ace/experiments/plots/metric_data.json", "w") as f:
        json.dump(data, f)


def plot_metric():
    with open("ace/experiments/plots/metric_data.json", "r") as f:
        data = json.load(f)

    plt.figure()
    colors = ["#d86c50", "#0ac9bf", "#a39aef", "#f4cc71"]
    methods = list(data.keys())
    metrics = ["DD", "DT", "RH", "UP"]

    bar_width = 0.2  # Width of each bar
    x = np.arange(len(metrics))  # Positions for metrics

    for i, method in enumerate(methods):
        # Collect data for each metric
        means = []
        stds = []
        for j in range(len(metrics)):
            metric_data = [data[method][trial][j] * 100 for trial in range(len(data[method]))]
            means.append(np.mean(metric_data))
            stds.append(np.std(metric_data))

        # Shift x positions for this method
        positions = x + (i - len(methods) / 2) * bar_width
        
        # Plot bars with error bars
        plt.bar(positions, means, yerr=stds, width=bar_width, color=colors[i], label=method, capsize=5, edgecolor="black")

    # Customize legend
    plt.legend(loc="upper right")

    # Label axes and set ticks
    plt.ylabel("Metric")
    plt.xticks(ticks=x, labels=metrics)

    plt.tight_layout()
    plt.savefig("ace/experiments/plots/metric.pdf")


if __name__ == "__main__":
    # get_action_distribution()
    plot()
    # get_metric_data()
    plot_metric()