import json
import os
import pandas as pd


def get_opponents():
    data_dir = "ace/data"
    seen = [filename.split(".")[0] for filename in os.listdir(f"{data_dir}/train") if filename.endswith(".json")]
    unseen = [filename.split(".")[0] for filename in os.listdir(f"{data_dir}/test") if filename.endswith(".json")]
    llm_based = ["Vanilla", "CoT", "PLAP"]
    return {
        "seen": seen,
        "unseen": unseen,
        "llm_based": llm_based
    }


def calculate_payoffs(metric: dict, win_loss_only=False) -> list:
    if win_loss_only:
        if metric["win_loss"] == [0, 0]:
            return [0, 0]
        elif metric["win_loss"] == [1, -1]:
            return [1, 0]
        else:
            return [0, 1]
    gamma = 1 - (metric["game_time"] / 2000)
    win_loss = [p * 10 * gamma * 4 for p in metric["win_loss"]]
    harvest = [p * 1 for p in metric["resource_harvested"]]
    attack = [sum(p.values()) * 1 for p in metric["unit_killed"]]
    build = [(p["base"] + p["barracks"]) * 0.2 for p in metric["unit_produced"]]
    produce_worker = [p["worker"] * 1 for p in metric["unit_produced"]]
    produce_army = [(p["heavy"] + p["light"] + p["ranged"]) * 4 for p in metric["unit_produced"]]
    payoffs = [sum(p) for p in zip(win_loss, harvest, attack, build, produce_worker, produce_army)]
    return payoffs


def get_ace_result(runs_dir, win_loss_only):
    opponents = get_opponents()
    results = {}
    # seen
    avg_payoffs = []
    for filename in opponents["seen"]:
        payoffs_list = []
        for run in os.listdir(f"{runs_dir}/{filename}"):
            with open(f"{runs_dir}/{filename}/{run}/metric.json") as f:
                metric = json.load(f)
            payoffs = calculate_payoffs(metric, win_loss_only)
            payoffs_list.append(payoffs)
        avg_payoffs.append([sum(p) / len(p) for p in zip(*payoffs_list)])
    results["Seen"] = [sum(p) / len(p) for p in zip(*avg_payoffs)]
    # unseen
    avg_payoffs = []
    for filename in opponents["unseen"]:
        payoffs_list = []
        for run in os.listdir(f"{runs_dir}/{filename}"):
            with open(f"{runs_dir}/{filename}/{run}/metric.json") as f:
                metric = json.load(f)
            payoffs = calculate_payoffs(metric, win_loss_only)
            payoffs_list.append(payoffs)
        avg_payoffs.append([sum(p) / len(p) for p in zip(*payoffs_list)])
    results["Unseen"] = [sum(p) / len(p) for p in zip(*avg_payoffs)]
    # llm_based
    for opponent in opponents["llm_based"]:
        avg_payoffs = []
        for run in os.listdir(f"{runs_dir}/{opponent}"):
            with open(f"{runs_dir}/{opponent}/{run}/metric.json") as f:
                metric = json.load(f)
            payoffs = calculate_payoffs(metric, win_loss_only)
            avg_payoffs.append(payoffs)
        results[opponent] = [sum(p) / len(p) for p in zip(*avg_payoffs)]
    return results


def get_baselines_result(runs_dir, win_loss_only):
    results = {}
    # unseen vs seen
    strategies = get_opponents()
    avg_payoffs = []
    for player in strategies["unseen"]:
        for opponent in strategies["seen"]:
            with open(f"{runs_dir}/{player}_vs_{opponent}/metric.json") as f:
                metric = json.load(f)
            payoffs = calculate_payoffs(metric, win_loss_only)
            avg_payoffs.append(payoffs)
    results["Unseen_vs_Seen"] = [sum(p) / len(p) for p in zip(*avg_payoffs)]
    
    # CoT vs Vanilla
    with open(f"{runs_dir}/{player}_vs_{opponent}/metric.json") as f:
        metric = json.load(f)
    results["CoT_vs_Vanilla"] = calculate_payoffs(metric, win_loss_only)

    # PLAP vs Vanilla, CoT
    for opponent in ["Vanilla", "CoT"]:
        with open(f"{runs_dir}/PLAP_vs_{opponent}/metric.json") as f:
            metric = json.load(f)
        results[f"PLAP_vs_{opponent}"] = calculate_payoffs(metric, win_loss_only)
    
    # seen vs llm_based
    for opponent in strategies["llm_based"]:
        avg_payoffs = []
        for player in strategies["seen"]:
            with open(f"{runs_dir}/{player}_vs_{opponent}/metric.json") as f:
                metric = json.load(f)
            payoffs = calculate_payoffs(metric, win_loss_only)
            avg_payoffs.append(payoffs)
        results[f"Seen_vs_{opponent}"] = [sum(p) / len(p) for p in zip(*avg_payoffs)]
    
    # unseen vs llm_based
    for opponent in strategies["llm_based"]:
        avg_payoffs = []
        for player in strategies["unseen"]:
            with open(f"{runs_dir}/{player}_vs_{opponent}/metric.json") as f:
                metric = json.load(f)
            payoffs = calculate_payoffs(metric, win_loss_only)
            avg_payoffs.append(payoffs)
        results[f"Unseen_vs_{opponent}"] = [sum(p) / len(p) for p in zip(*avg_payoffs)]
    
    return results


def get_main_result(win_loss_only):
    ace_results = get_ace_result("runs/eval_ace", win_loss_only)
    baselines_results = get_baselines_result("runs/eval_baseline", win_loss_only)
    df = pd.DataFrame()
    for match, payoffs in baselines_results.items():
        player, opponent = match.split("_vs_")
        df.loc[player, opponent] = payoffs[0]
        df.loc[opponent, player] = payoffs[1]
    for opponent, payoffs in ace_results.items():
        df.loc["ACE", opponent] = payoffs[0]
        df.loc[opponent, "ACE"] = payoffs[1]
    
    df = df.fillna(0)
    order = ["Vanilla", "CoT", "PLAP", "Seen", "Unseen", "ACE"]
    df = df.reindex(index=order, columns=order, fill_value=0)
    df.to_csv("ace/experiments/plots/main_results.csv")


def get_scaling_result():
    results = get_ace_result("runs/scaling_runs", win_loss_only=True)
    df = pd.DataFrame()
    for opponent, payoffs in results.items():
        df.loc["ACE", opponent] = payoffs[0]
        df.loc[opponent, "ACE"] = payoffs[1]
    df = df.fillna(0)
    order = ["Vanilla", "CoT", "PLAP", "Seen", "Unseen", "ACE"]
    df = df.reindex(index=order, columns=order, fill_value=0)
    df.to_csv("ace/experiments/plots/scaling_results.csv")


if __name__ == "__main__":
    # get_main_result(win_loss_only=False)
    get_scaling_result()
