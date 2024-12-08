import argparse
from omegaconf import OmegaConf
import os
import re
from skill_rts.envs.wrappers import MicroRTSLLMEnv
from ace.agent  import Planner
from ace.strategy import Strategy
import json
import pandas as pd
import time

DATA_DIR = "ace/data"


def parse_args(config_path: str = "ace/pre_match/config/battle.yaml"):

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
    
    return cfg


def gen_opponent(map_name: str) -> str:
    """Convert strategies for opponent
    
    Convert existing strategies to opponent strategies.
    Specifically, convert the defensive strategy's defensive 
    area to the area corresponding to player 1.
    """
    width, height = re.search(r"(\d+)x(\d+)", map_name).groups()
    width, height = int(width) - 1, int(height) - 1

    for i in range(1, int(1e9)):
        filename = f"{DATA_DIR}/strategies/strategy_{i}.json"
        if not os.path.exists(filename):
            break
        strategy = Strategy.load_from_json(filename)
        if not strategy.aggression:
            old_defense = strategy.defense
            strategy.defense = [(width - loc[0], height - loc[1]) for loc in strategy.defense]
            for old_loc, new_loc in zip(old_defense, strategy.defense):
                try:
                    strategy.strategy = strategy.strategy.replace(str(old_loc), str(new_loc))
                    strategy.description = strategy.description.replace(str(old_loc), str(new_loc))
                except Exception as e:
                    print(f"Failed to convert strategy {i}:\n{e}")
        print(f"Converted strategy {i}")
        strategy.to_json(f"{DATA_DIR}/opponents/strategy_{i}.json", map_name)


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

    train_strategy_dir = f"{DATA_DIR}/train/strategies"
    train_opponent_dir = f"{DATA_DIR}/train/opponents"
    test_strategy_dir = f"{DATA_DIR}/test/strategies"
    test_opponent_dir = f"{DATA_DIR}/test/opponents"
    if not os.path.exists(train_strategy_dir):
        os.makedirs(train_strategy_dir)
    if not os.path.exists(train_opponent_dir):
        os.makedirs(train_opponent_dir)
    if not os.path.exists(test_strategy_dir):
        os.makedirs(test_strategy_dir)
    if not os.path.exists(test_opponent_dir):
        os.makedirs(test_opponent_dir)
    
    for i in train_indices:
        shutil.copy(f"{DATA_DIR}/strategies/strategy_{i}.json", f"{train_strategy_dir}/strategy_{i}.json")
        shutil.copy(f"{DATA_DIR}/opponents/strategy_{i}.json", f"{train_opponent_dir}/strategy_{i}.json")

    for i in test_indices:
        shutil.copy(f"{DATA_DIR}/strategies/strategy_{i}.json", f"{test_strategy_dir}/strategy_{i}.json")
        shutil.copy(f"{DATA_DIR}/opponents/strategy_{i}.json", f"{test_opponent_dir}/strategy_{i}.json")


def extract_battle_results():
    runs_dir = "runs/pre_match_runs"
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


def run():
    # Initialize
    cfg = parse_args()
    map_name = cfg.env.map_path.split("/")[-1].split(".")[0]
    strategy_name = re.search(r"strategy_(\d+)", cfg.agents[0].strategy).group(1)
    opponent_name = re.search(r"strategy_(\d+)", cfg.agents[1].strategy).group(1)
    run_dir = f"pre_match_runs/{strategy_name}_{opponent_name}"

    # Run the game
    agents = []
    for agent_cfg in cfg.agents:
        agents.append(Planner(**agent_cfg, map_name=map_name))

    env = MicroRTSLLMEnv(agents, **cfg.env, run_dir=run_dir)
    start_time = time.time()
    payoffs, trajectory = env.run()
    end_time = time.time()
    metric = env.metric

    # Save the results
    trajectory.to_json(f"{run_dir}/raw_traj.json")
    metric.to_json(f"{run_dir}/metric.json")
    print(f"{strategy_name} vs {opponent_name} | Payoffs: {payoffs} | Runtime: {(end_time - start_time)/60:.2f}min, {env.time} steps")


if __name__ == "__main__":
    # gen_opponent("basesWorkers8x8")
    # train_test_split()
    # run()
    extract_battle_results()
