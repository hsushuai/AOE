import argparse
from omegaconf import OmegaConf
import json
from skill_rts.envs.wrappers import MicroRTSLLMEnv
from ace.agent import Planner, AceAgent
from ace.strategy import Strategy
from skill_rts.agents import bot_agent
from skill_rts import logger
import time
import os


def parse_args():

    cfg = OmegaConf.load("ace/configs/ace.yaml")

    parser = argparse.ArgumentParser()
    parser.add_argument("--map_path", type=str, help="Path to the map file")
    parser.add_argument("--max_steps", type=int, help="Maximum steps for the environment")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--temperature", type=float, help="Temperature for LLM")
    parser.add_argument("--max_tokens", type=int, help="Maximum tokens for LLM")
    parser.add_argument("--num_generations", type=int, help="Number of generations for LLM")
    parser.add_argument("--opponent", type=str, help="Strategy for opponent")
    parser.add_argument("--interval", type=int, help="Interval for update plan")

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
    if args.interval is not None:
        cfg.env.interval = args.interval
    
    # do not update strategy during the match
    cfg.agents[0].strategy_interval = cfg.env.max_steps
    cfg.episodes = 5
    return cfg


def main():
    # Initialize
    cfg = parse_args()
    map_name = cfg.env.map_path.split("/")[-1].split(".")[0]

    if cfg.agents[1].strategy in bot_agent.ALL_AIS:
        opponent_agent = bot_agent.ALL_AIS[cfg.agents[1].strategy]
        opponent_name = cfg.agents[1].strategy
    else:
        opponent_agent = Planner(**cfg.agents[1], player_id=1, map_name=map_name)
        opponent_name = cfg.agents[1].strategy.split('/')[-1].split('.')[0]
    
    runs_dir = f"runs/eval_adaption/{opponent_name}"
    logger.set_level(logger.DEBUG)

    ace = AceAgent(player_id=0, map_name=map_name, **cfg.agents[0])
    ace.meta_strategy = Strategy.load_from_json("ace/data/strategies/strategy_10.json")
    ace.strategy = ace.meta_strategy.to_string()
    ace.planner.strategy = ace.strategy
    env = MicroRTSLLMEnv([ace, opponent_agent], **cfg.env)
    os.makedirs(runs_dir, exist_ok=True)
    adaption_log = open(f"{runs_dir}/adaption.log", "w")
    
    # Run the episodes
    for episode in range(cfg.episodes):
        run_dir = f"{runs_dir}/run_{episode}"
        env.set_dir(run_dir)
        start_time = time.time()
        try:
            payoffs, trajectory = env.run()
        except Exception as e:
            print(f"Error in episode {episode}: {e}")
            env.close()
            break
        
        # Update the strategy
        logger.set_stream(adaption_log)
        ace.update_strategy(trajectory)

        # Save the results
        OmegaConf.save(cfg, f"{run_dir}/config.yaml")
        trajectory.to_json(f"{run_dir}/traj.json")
        env.metric.to_json(f"{run_dir}/metric.json")
        with open(f"{run_dir}/plans.json", "w") as f:
            json.dump(env.plans, f, indent=4)
        print(f"Match {episode} | Opponent {opponent_name} | Payoffs: {payoffs} | Runtime: {(time.time() - start_time) / 60:.2f}min, {env.time}steps")
    adaption_log.close()


if __name__ == "__main__":
    main()
