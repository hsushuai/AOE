import json
import os


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


def calculate_payoffs(metric: dict) -> list:
    win_loss = [p * 10 for p in metric["win_loss"]]
    harvest = [p * 1 for p in metric["resource_harvested"]]
    attack = [sum(p.values()) * 1 for p in metric["unit_killed"]]
    build = [(p["base"] + p["barracks"]) * 0.2 for p in metric["unit_produced"]]
    produce_worker = [p["worker"] * 1 for p in metric["unit_produced"]]
    produce_army = [(p["heavy"] + p["light"] + p["ranged"]) * 4 for p in metric["unit_produced"]]
    payoffs = [sum(p) for p in zip(win_loss, harvest, attack, build, produce_worker, produce_army)]
    return payoffs

def get_ace_result(runs_dir):
    opponents = get_opponents()
    results = {}
    # seen
    avg_payoffs = []
    for filename in opponents["seen"]:
        payoffs_list = []
        for run in os.listdir(f"{runs_dir}/{filename}"):
            with open(f"{runs_dir}/{filename}/{run}/metric.json") as f:
                metric = json.load(f)
            payoffs = calculate_payoffs(metric)
            payoffs_list.append(payoffs)
        avg_payoffs.append([sum(p) / len(p) for p in zip(*payoffs_list)])
    results["seen"] = [sum(p) / len(p) for p in zip(*avg_payoffs)]
    # unseen
    avg_payoffs = []
    for filename in opponents["unseen"]:
        payoffs_list = []
        for run in os.listdir(f"{runs_dir}/{filename}"):
            with open(f"{runs_dir}/{filename}/{run}/metric.json") as f:
                metric = json.load(f)
            payoffs = calculate_payoffs(metric)
            payoffs_list.append(payoffs)
        avg_payoffs.append([sum(p) / len(p) for p in zip(*payoffs_list)])
    results["unseen"] = [sum(p) / len(p) for p in zip(*avg_payoffs)]
    # llm_based
    for opponent in opponents["llm_based"]:
        avg_payoffs = []
        for run in os.listdir(f"{runs_dir}/{opponent}"):
            with open(f"{runs_dir}/{opponent}/{run}/metric.json") as f:
                metric = json.load(f)
            payoffs = calculate_payoffs(metric)
            avg_payoffs.append(payoffs)
        results[opponent] = [sum(p) / len(p) for p in zip(*avg_payoffs)]
    return results


def get_baselines_result(runs_dir):
    results = {}
    # unseen vs seen
    strategies = get_opponents()
    avg_payoffs = []
    for player in strategies["unseen"]:
        for opponent in strategies["seen"]:
            with open(f"{runs_dir}/{player}_vs_{opponent}/metric.json") as f:
                metric = json.load(f)
            payoffs = calculate_payoffs(metric)
            avg_payoffs.append(payoffs)
    results["unseen_vs_seen"] = [sum(p) / len(p) for p in zip(*avg_payoffs)]
    
    # seen vs llm_based
    for opponent in strategies["llm_based"]:
        avg_payoffs = []
        for player in strategies["seen"]:
            with open(f"{runs_dir}/{player}_vs_{opponent}/metric.json") as f:
                metric = json.load(f)
            payoffs = calculate_payoffs(metric)
            avg_payoffs.append(payoffs)
        results[f"seen_vs_{opponent}"] = [sum(p) / len(p) for p in zip(*avg_payoffs)]
    
    # unseen vs llm_based
    for opponent in strategies["llm_based"]:
        avg_payoffs = []
        for player in strategies["unseen"]:
            with open(f"{runs_dir}/{player}_vs_{opponent}/metric.json") as f:
                metric = json.load(f)
            payoffs = calculate_payoffs(metric)
            avg_payoffs.append(payoffs)
        results[f"unseen_vs_{opponent}"] = [sum(p) / len(p) for p in zip(*avg_payoffs)]
    
    # CoT vs Vanilla
    with open(f"{runs_dir}/{player}_vs_{opponent}/metric.json") as f:
        metric = json.load(f)
    results["CoT_vs_Vanilla"] = calculate_payoffs(metric)

    # PLAP vs Vanilla, CoT
    for opponent in ["Vanilla", "CoT"]:
        with open(f"{runs_dir}/PLAP_vs_{opponent}/metric.json") as f:
            metric = json.load(f)
        results[f"PLAP_vs_{opponent}"] = calculate_payoffs(metric)
    
    return results



if __name__ == "__main__":
    # results = get_ace_result("runs/eval_ace")
    results = get_baselines_result("runs/eval_baseline")
    print(results)
