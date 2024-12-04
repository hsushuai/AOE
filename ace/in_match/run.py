import argparse
from omegaconf import OmegaConf
import os
from skill_rts.envs.wrappers import MicroRTSLLMEnv
from ace.agent  import Planner, Recognizer
from ace.traj_feat import TrajectoryFeature
from ace.strategy  import Strategy
from skill_rts.agents import bot_agent


def parse_args(config_path: str = "ace/configs/in_match/run.yaml"):

    cfg = OmegaConf.load(config_path)

    parser = argparse.ArgumentParser()
    parser.add_argument("--map_path", type=str, help="Path to the map file")
    parser.add_argument("--max_steps", type=int, help="Maximum steps for the environment")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--temperature", type=float, help="Temperature for LLM")
    parser.add_argument("--max_tokens", type=int, help="Maximum tokens for LLM")
    parser.add_argument("--num_generations", type=int, help="Number of generations for LLM")
    parser.add_argument("--opponent_strategy", type=str, help="Strategies for opponent")

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
    if args.opponent_strategy is not None:
        cfg.agents[1].strategy = args.opponent_strategy
    
    return cfg


def main():
    # ====================
    #      Initialize
    # ====================
    cfg = parse_args()
    map_name = cfg.env.map_path.split("/")[-1].split(".")[0]

    if cfg.agents[1].model in bot_agent.ALL_AIS:
        opponent_agent = bot_agent.ALL_AIS[cfg.agents[1].model]
        opponent_name = cfg.agents[1].model
    else:
        opponent_agent = Planner(**cfg.agents[1], player_id=1, map_name=map_name)
        opponent_name = cfg.agents[1].strategy.split('/')[-1].split('.')[0]
    
    run_dir = f"in_match_runs/{opponent_name}"
    os.makedirs(run_dir, exist_ok=True)

    # Run the game
    agent = Planner(
        player_id=0,
        map_name=map_name,
        strategy="none",
        prompt="few-shot-w-strategy",
        **cfg.agents[0]
    )
    env = MicroRTSLLMEnv([agent, opponent_agent], **cfg.env, run_dir=run_dir)
    payoffs, trajectory = env.run()
    metric = env.metric

    # Save the results
    trajectory.to_json(f"{run_dir}/traj.json")
    metric.to_json(f"{run_dir}/metric.json")
    print(f"Opponent {opponent_name} |  Payoffs: {payoffs} | Game Time: {env.time}")


if __name__ == "__main__":
    # main()
    strategy = Strategy.load_from_json("ace/data/strategies/strategy_2.json")
    print(strategy.feats)
    print(strategy.one_hot_feats)
    decoded_strategy = Strategy.decode(strategy.one_hot_feats)
    print(decoded_strategy.feats)
    print(decoded_strategy.one_hot_feats)
