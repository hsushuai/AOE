import argparse
from omegaconf import OmegaConf
import os
import re
from skill_rts.envs.wrappers import MicroRTSLLMEnv
from ace.agent  import AceAgent
from ace.strategy import Strategy
import numpy as np

BASE_DIR = "ace/data/strategies"


def parse_args(config_path: str = "ace/configs/pre_match/battle.yaml"):

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
        cfg.agents[1].strategy = args.opponent_strategy
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


def main():
    # Initialize
    cfg = parse_args()
    map_name = cfg.env.map_path.split("/")[-1].split(".")[0]
    strategy_name = re.search(r"strategy_(\d+)", cfg.agents[0].strategy).group(1)
    opponent_name = re.search(r"strategy_(\d+)", cfg.agents[1].strategy).group(1)
    runs_dir = f"pre_match_runs/{strategy_name}_{opponent_name}"

    # Run the game
    agents = []
    for agent_cfg in cfg.agents:
        agents.append(AceAgent(**agent_cfg, map_name=map_name))
    
    avg_payoffs = []
    for i in range(cfg.best_of_n):
        run_dir = f"{runs_dir}/match_{i + 1}"
        env = MicroRTSLLMEnv(agents, **cfg.env, run_dir=run_dir)
        payoffs, trajectory = env.run()
        metric = env.metric

        avg_payoffs.append(payoffs)

        # Save the results
        trajectory.to_json(f"{run_dir}/raw_traj.json")
        metric.to_json(f"{run_dir}/metric.json")
        print(f"{strategy_name} vs {opponent_name} | Match {i + 1} | Payoffs: {payoffs}")
        break


if __name__ == "__main__":
    # gen_opponent("basesWorkers8x8")
    main()
