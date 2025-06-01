import argparse
from omegaconf import OmegaConf
import json
from skill_rts.envs import MicroRTSLLMEnv
from sap.agent  import Planner
from skill_rts.agents import  VanillaAgent, CoTAgent, PLAPAgent
from skill_rts import logger
import time
import os


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
    parser.add_argument("--player", type=str, help="Strategy for player")
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
    if args.player is not None:
        cfg.agents[0].strategy = args.player
    if args.interval is not None:
        cfg.env.interval = args.interval
    
    return cfg


def run(cfg):
    # Initialize agents
    map_name = cfg.env.map_path.split("/")[-1].split(".")[0]
    agents = {}
    for i, agent_cfg in enumerate(cfg.agents):
        if agent_cfg.strategy in llm_based_baselines:
            agent = llm_based_baselines[agent_cfg.strategy](
                agent_cfg.model,
                agent_cfg.temperature,
                agent_cfg.max_tokens,
                player_id=i
            )
            agents[agent_cfg.strategy] = agent
        elif "json" in agent_cfg.strategy:
            agent = Planner(
                model=agent_cfg.model,
                temperature=agent_cfg.temperature,
                max_tokens=agent_cfg.max_tokens,
                prompt="few-shot-w-strategy", 
                player_id=i, 
                map_name=map_name,
                strategy=agent_cfg.strategy
            )
            agents[agent_cfg.strategy.split("/")[-1].split(".")[0]] = agent
        else:
            raise ValueError(f"Unknown agent strategy: {agent_cfg.strategy}")

    pair = list(agents.keys())
    match_name = f"{pair[0]}_vs_{pair[1]}"
    run_dir = f"runs/eval_baseline/{match_name}"
    logger.set_level(logger.DEBUG)

    env = MicroRTSLLMEnv(list(agents.values()), **cfg.env)
    
    # Run the episodes
    print(f"Match {match_name} | ", end="", flush=True)
    env.set_dir(run_dir)
    start_time = time.time()
    try:
        payoffs, trajectory = env.run()
    except Exception as e:
        print(f"Error in match {match_name}: {e}")
        env.close()
        return

    # Save the results
    OmegaConf.save(cfg, f"{run_dir}/config.yaml")
    trajectory.to_json(f"{run_dir}/traj.json")
    env.metric.to_json(f"{run_dir}/metric.json")
    with open(f"{run_dir}/plans.json", "w") as f:
        json.dump(env.plans, f, indent=4)
    print(f"Payoffs: {payoffs} | Runtime: {(time.time() - start_time) / 60:.2f}min, {env.time}steps")


def main():
    cfg = parse_args()
    seen = [f"sap/data/train/{filename}" for filename in os.listdir("sap/data/train") if filename.endswith(".json")]
    unseen = [f"sap/data/test/{filename}" for filename in os.listdir("sap/data/test") if filename.endswith(".json")]
    seen_battle = []
    for filename in seen:
        seen_battle.append([(filename, "Vanilla"), (filename, "CoT"), (filename, "PLAP")])
    unseen_battle = []
    for filename in unseen:
        unseen_battle.append([(filename, "Vanilla"), (filename, "CoT"), (filename, "PLAP")])
    unseen_vs_seen = []
    for filename in unseen:
        unseen_vs_seen.append([(filename, seen_filename) for seen_filename in seen])
            
    battle_matrix = [
        [("CoT", "Vanilla")],
        [("PLAP", "Vanilla"), ("PLAP", "CoT")]
    ] + seen_battle + unseen_battle + unseen_vs_seen
    num_battles = 0
    for battles in battle_matrix:
        num_battles += len(battles)
    i = 0
    for battles in battle_matrix:
        for battle in battles:
            i += 1
            cfg.agents[0].strategy = battle[0]
            cfg.agents[1].strategy = battle[1]
            print(f"Runs {i}/{num_battles} | ", end="", flush=True)
            run(cfg)


if __name__ == "__main__":
    main()
