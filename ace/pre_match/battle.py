import argparse
from omegaconf import OmegaConf
import os
import re
from skill_rts.envs.wrappers import MicroRTSLLMEnv
from ace.agent  import AceAgent
from ace.strategy import Strategy
import time

BASE_DIR = "ace/data/strategies"


def parse_args(config_path: str = "ace/pre_match/configs/battle.yaml"):

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
        filename = f"{BASE_DIR}/strategy_{i}.json"
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
        strategy.to_json(os.path.join(os.path.dirname(BASE_DIR), "opponents", f"strategy_{i}.json"))


def train_test_split(train_size=0.6):
    import numpy as np
    import shutil

    num_strategies = len([_ for _ in os.listdir(BASE_DIR) if _.endswith(".json")])
    print(f"Number of strategies: {num_strategies}")
    train_indices = np.random.choice(range(1, num_strategies + 1), size=int(num_strategies * train_size), replace=False)
    test_indices = np.setdiff1d(np.arange(1, num_strategies + 1), train_indices)
    print(f"Number of training strategies: {len(train_indices)}")
    print(f"Number of testing strategies: {len(test_indices)}")

    opponents_dir = os.path.join(os.path.dirname(BASE_DIR), "opponents")
    train_strategy_dir = "ace/data/train/strategies"
    train_opponent_dir = "ace/data/train/opponents"
    test_dir = "ace/data/test"
    if not os.path.exists(train_strategy_dir):
        os.makedirs(train_strategy_dir)
    if not os.path.exists(train_opponent_dir):
        os.makedirs(train_opponent_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    for i in train_indices:
        shutil.copy(f"{BASE_DIR}/strategy_{i}.json", f"{train_strategy_dir}/strategy_{i}.json")
        shutil.copy(f"{opponents_dir}/strategy_{i}.json", f"{train_opponent_dir}/strategy_{i}.json")

    for i in test_indices:
        shutil.copy(f"{opponents_dir}/strategy_{i}.json", f"{test_dir}/strategy_{i}.json")


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
        agents.append(AceAgent(**agent_cfg, map_name=map_name))

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
    run()
