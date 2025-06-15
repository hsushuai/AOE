import json
from omegaconf import OmegaConf
import os
from sap.traj_feat import TrajectoryFeature
from skill_rts.game import Trajectory
import concurrent.futures


def extract_strategy_obs_to_plan(run_dir, qwen3=False):
    """
    Extracts the strategy and observation as agent inputs and the plan as agent output.
    If the strategy-base agent not win in the run, it returns None.
    """
    print(f"Extracting data from {run_dir}")
    with open(f"{run_dir}/metric.json") as f:
        metric = json.load(f)
    with open(f"{run_dir}/plans.json") as f:
        trajectories = json.load(f)
    strategy_based_agents_ids = [i for i, p in enumerate(trajectories[0]["players"]) if "strategy" in p]
    if not any(metric["win_loss"][i] == 1 for i in strategy_based_agents_ids):
        return None

    winner_id = metric["win_loss"].index(1)
    data = []
    templates = OmegaConf.load("sap/templates/planner.yaml")
    usr_template = templates["USER"].strip()
    sys_prompt = templates["SYSTEM"].strip()
    for traj in trajectories:
        d = traj["players"][winner_id]
        usr_prompt = usr_template.format(
            strategy=d["strategy"],
            observation=d["obs"],
            player_id=d["id"]
        )
        data.append([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": usr_prompt},
            {"role": "assistant", "content": d["plan"]}
        ])
        if qwen3:
            data[-1][0]["content"] += " /no_think"
            data[-1][1]["content"] = "<think>\n\n</think>\n\n" + data[-1][1]["content"]
    return data


def extract_traj_to_strategy(run_dir, qwen3=False):
    print(f"Extracting data from {run_dir}")
    with open(f"{run_dir}/plans.json") as f:
        plans = json.load(f)
    if len(plans[0]["players"]) != 2 or "strategy" not in plans[0]["players"][1]:
        return None
    traj = Trajectory.load(f"{run_dir}/traj.json")
    traj_feat = TrajectoryFeature(traj)
    templates = OmegaConf.load("sap/templates/recognizer.yaml")
    usr_template = templates["USER"].strip()
    sys_prompt = templates["SYSTEM"].strip()
    data = []
    for i, p in enumerate(plans[1:]):
        d = plans[i - 1]["players"][1]
        usr_prompt = usr_template.format(trajectory=traj_feat.to_string(end=p["time"]))
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": usr_prompt},
            {"role": "assistant", "content": d["strategy"]}
        ]
        if qwen3:
            messages[0]["content"] += " /no_think"
            messages[1]["content"] = "<think>\n\n</think>\n\n" + messages[1]["content"]
        data.append(messages)
    # add the end of the trajectory
    d = plans[-1]["players"][1]
    usr_prompt = usr_template.format(trajectory=traj_feat.to_string())
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": usr_prompt},
        {"role": "assistant", "content": d["strategy"]}
    ]
    if qwen3:
        messages[0]["content"] += " /no_think"
        messages[1]["content"] = "<think>\n\n</think>\n\n" + messages[1]["content"]
    data.append(messages)
    return data


def get_all_run_dirs(run_dir, seen_only=False):
    run_dirs = []
    for root, dirs, files in os.walk(run_dir):
        if root == "sap_distill":
            continue
        if "plans.json" in files and not dirs:
            run_dirs.append(root)
    return run_dirs


def extract_all_strategy_obs_to_plan(output, qwen3=False):
    run_dirs = get_all_run_dirs("runs")
    all_data = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=80) as executor:
        futures = [executor.submit(extract_strategy_obs_to_plan, run_dir, qwen3) for run_dir in run_dirs]
        for future in concurrent.futures.as_completed(futures):
            data = future.result()
            if data is not None:
                all_data += data
    with open(output, "w") as f:
        for data in all_data:
            f.write(json.dumps({"messages": data}) + "\n")
    
    print(f"Extracted {len(all_data)} data points to {output}")


def extract_all_traj_to_strategy(output, qwen3=False):
    run_dirs = get_all_run_dirs("runs")
    all_data = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=80) as executor:
        futures = [executor.submit(extract_traj_to_strategy, run_dir, qwen3) for run_dir in run_dirs]
        for future in concurrent.futures.as_completed(futures):
            data = future.result()
            if data is not None:
                all_data += data
    with open(output, "w") as f:
        for data in all_data:
            f.write(json.dumps({"messages": data}) + "\n")
    print(f"Extracted {len(all_data)} data points to {output}")


def merge_data(data_dir):
    input_files = [f for f in os.listdir(data_dir) if f.endswith(".jsonl")]
    all_data = []
    for input_file in input_files:
        if input_file == "train.jsonl":
            continue
        data = []
        with open(f"{data_dir}/{input_file}", "r") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        all_data += data
        print(f"Loaded {len(data)} data points from {input_file}")
    output_file = f"{data_dir}/train.jsonl"
    with open(output_file, "w") as f:
        for data in all_data:
            f.write(json.dumps(data) + "\n")
    print(f"Merged {len(all_data)} data points to {output_file}")


if __name__ == "__main__":
    data_version=1
    extract_all_strategy_obs_to_plan(f"distill-sap/data/data-{data_version}/planner.jsonl", qwen3=False)
    extract_all_traj_to_strategy(f"distill-sap/data/data-{data_version}/recognizer.jsonl", qwen3=False)
    merge_data("distill-sap/data/data-1")
