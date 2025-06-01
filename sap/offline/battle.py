import argparse
from omegaconf import OmegaConf
import os
import re
from skill_rts.envs.wrappers import MicroRTSLLMEnv
from skill_rts import logger
from sap.agent  import Planner
import json
import pandas as pd
import time

DATA_DIR = "sap/data"


def parse_args(config_path: str = "sap/configs/battle.yaml"):

    cfg = OmegaConf.load(config_path)

    parser = argparse.ArgumentParser()
    parser.add_argument("--map_path", type=str, help="Path to the map file")
    parser.add_argument("--max_steps", type=int, help="Maximum steps for the environment")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--temperature", type=float, help="Temperature for LLM")
    parser.add_argument("--max_tokens", type=int, help="Maximum tokens for LLM")
    parser.add_argument("--num_generations", type=int, help="Number of generations for LLM")
    parser.add_argument("--opponent", type=str, help="Strategy for opponent")
    parser.add_argument("--strategy", type=str, help="Strategy for agent")

    args = parser.parse_args()

    if args.map_path is not None:
        cfg.env.map_path = args.map_path
    if args.model is not None:
        cfg.agents[0].model = args.model
        cfg.agents[1].model = args.model
    if args.temperature is not None:
        cfg.agents[0].temperature = args.temperature
        cfg.agents[1].temperature = args.temperature
    if args.max_tokens is not None:
        cfg.agents[0].max_tokens = args.max_tokens
        cfg.agents[1].max_tokens = args.max_tokens
    if args.opponent is not None:
        cfg.agents[1].strategy = args.opponent
    if args.strategy is not None:
        cfg.agents[0].strategy = args.strategy

    cfg.env.map_path = "maps/16x16/basesWorkers16x16.xml"
    
    return cfg


def train_test_split(train_size=0.6):
    """Train for seen opponents, test for unseen opponents"""
    import numpy as np
    import shutil

    num_strategies = len([_ for _ in os.listdir(f"{DATA_DIR}/strategies") if _.endswith(".json")])
    print(f"Number of strategies: {num_strategies}")
    np.random.seed(520)
    train_indices = np.random.choice(range(1, num_strategies + 1), size=int(num_strategies * train_size), replace=False)
    test_indices = np.setdiff1d(np.arange(1, num_strategies + 1), train_indices)
    print(f"Number of training strategies: {len(train_indices)}")
    print(f"Number of testing strategies: {len(test_indices)}")

    train_dir = f"{DATA_DIR}/train"
    test_dir = f"{DATA_DIR}/test"
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    for i in train_indices:
        shutil.copy(f"{DATA_DIR}/strategies/strategy_{i}.json", f"{train_dir}/strategy_{i}.json")

    for i in test_indices:
        shutil.copy(f"{DATA_DIR}/strategies/strategy_{i}.json", f"{test_dir}/strategy_{i}.json")


def extract_battle_results():
    runs_dir = "runs/offline_runs_16x16"
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
    df.to_csv("sap/data/payoff/payoff_matrix_16x16.csv")


def run():
    # Initialize
    cfg = parse_args()
    map_name = cfg.env.map_path.split("/")[-1].split(".")[0]
    strategy_name = re.search(r"strategy_(\d+)", cfg.agents[0].strategy).group(1)
    opponent_name = re.search(r"strategy_(\d+)", cfg.agents[1].strategy).group(1)
    run_dir = f"runs/offline_runs_16x16/{strategy_name}_{opponent_name}"
    logger.set_level(logger.DEBUG)

    # Run the game
    agents = []
    for agent_cfg in cfg.agents:
        agents.append(Planner(**agent_cfg, map_name=map_name))

    env = MicroRTSLLMEnv(agents, **cfg.env, run_dir=run_dir)
    start_time = time.time()
    try:
        payoffs, trajectory = env.run()
    except Exception as e:
        print(f"Error: {e}")
        env.close()
        return
    end_time = time.time()
    metric = env.metric

    # Save the results
    OmegaConf.save(cfg, f"{run_dir}/config.yaml")
    trajectory.to_json(f"{run_dir}/raw_traj.json")
    metric.to_json(f"{run_dir}/metric.json")
    print(f"{strategy_name} vs {opponent_name} | Payoffs: {payoffs} | Runtime: {(end_time - start_time)/60:.2f}min, {env.time} steps")


if __name__ == "__main__":
    # train_test_split()
    # run()
    extract_battle_results()
