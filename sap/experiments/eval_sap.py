import argparse
from omegaconf import OmegaConf
import json
from skill_rts.envs import MicroRTSLLMEnv
from sap.agent  import Planner, SAPAgent
from skill_rts.agents import bot_ais, VanillaAgent, CoTAgent, PLAPAgent
from skill_rts import logger
import time


llm_based_baselines = {
    "Vanilla": VanillaAgent,
    "CoT": CoTAgent,
    "PLAP": PLAPAgent
}


def parse_args():

    cfg = OmegaConf.load("sap/configs/sap.yaml")

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
    
    return cfg


def main():
    # Initialize
    cfg = parse_args()
    map_name = cfg.env.map_path.split("/")[-1].split(".")[0]

    if cfg.agents[1].strategy in bot_ais:
        opponent_agent = bot_ais[cfg.agents[1].strategy]
        opponent_name = cfg.agents[1].strategy
    elif "json" in cfg.agents[1].strategy:
        opponent_agent = Planner(**cfg.agents[1], player_id=1, map_name=map_name)
        opponent_name = cfg.agents[1].strategy.split('/')[-1].split('.')[0]
    elif cfg.agents[1].strategy in llm_based_baselines:
        opponent_agent = llm_based_baselines[cfg.agents[1].strategy](
            cfg.agents[1].model,
            cfg.agents[1].temperature,
            cfg.agents[1].max_tokens,
            player_id=1
        )
        opponent_name = cfg.agents[1].strategy
    else:
        raise ValueError(f"Unknown opponent strategy: {cfg.agents[1].strategy}")
    
    runs_dir = f"runs/main_runs/{opponent_name}"
    logger.set_level(logger.DEBUG)

    agent = SAPAgent(player_id=0, map_name=map_name, **cfg.agents[0])
    env = MicroRTSLLMEnv([agent, opponent_agent], **cfg.env)
    # reviewer = Reviewer(**cfg.agents[0])
    
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
            continue
        
        # Post-match review
        # if payoffs[0] < payoffs[1]:
        #     for d in env.plans:
        #         player = d["players"][0]
        #         reviewer.reflect_planner(player["strategy"], player["obs"], player["plan"])
        #     reviewer.reflect_meta_strategy(trajectory)
        #     env.agents[0].meta_strategy = reviewer.meta_strategy
        #     env.agents[0].planner.tips = reviewer.planner_tips
        # else:
        #     reviewer.recognize_strategy(trajectory)

        # Save the results
        OmegaConf.save(cfg, f"{run_dir}/config.yaml")
        trajectory.to_json(f"{run_dir}/traj.json")
        env.metric.to_json(f"{run_dir}/metric.json")
        with open(f"{run_dir}/plans.json", "w") as f:
            json.dump(env.plans, f, indent=4)
        print(f"Match {episode} | Opponent {opponent_name} | Payoffs: {payoffs} | Runtime: {(time.time() - start_time) / 60:.2f}min, {env.time}steps")


if __name__ == "__main__":
    main()
