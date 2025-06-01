import json
from omegaconf import OmegaConf
import os


templates = OmegaConf.load("distill-sap/template/prompt.yaml")


def extract_strategy_obs_to_plan(run_dir):
    """
    Extracts the strategy and observation as agent inputs and the plan as agent output.
    If the strategy-base agent not win in the run, it returns None.
    """
    with open(f"{run_dir}/metric.json") as f:
        metric = json.load(f)
    with open(f"{run_dir}/plans.json") as f:
        trajectories = json.load(f)
    strategy_based_agents_ids = [i for i, p in enumerate(trajectories[0]["players"]) if "strategy" in p]
    if not any(metric["win_loss"][i] == 1 for i in strategy_based_agents_ids):
        return None

    winner_id = metric["win_loss"].index(1)
    data = []
    for traj in trajectories:
        d = traj["players"][winner_id]
        prompt = templates["strategy_obs_to_plan"].format(
            strategy=d["strategy"],
            observation=d["obs"],
            player_id=d["id"]
        )
        data.append([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": d["plan"]}
        ])
    return data


def get_all_run_dirs(run_dir):
    run_dirs = []
    for root, dirs, files in os.walk(run_dir):
        if "plans.json" in files and not dirs:
            run_dirs.append(root)
    return run_dirs 


def extract_all_strategy_obs_to_plan(output):
    run_dirs = get_all_run_dirs("runs")
    all_data = []
    for run_dir in run_dirs:
        print(f"Extracting data from {run_dir}")
        data = extract_strategy_obs_to_plan(run_dir)
        if data is not None:
            all_data += data

    with open(output, "w") as f:
        for data in all_data:
            f.write(json.dumps({"messages": data}) + "\n")
    
    print(f"Extracted {len(all_data)} data points to {output}")


if __name__ == "__main__":
    extract_all_strategy_obs_to_plan("distill-sap/data/strategy_obs_to_plan.jsonl")