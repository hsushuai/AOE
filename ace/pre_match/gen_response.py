import argparse
from omegaconf import OmegaConf
import yaml
import json
import os
from skill_rts.envs.wrappers import MicroRTSLLMEnv
from skill_rts.agents.llm_clients import Qwen
from ace.agent  import Planner
from ace.traj_feat import TrajectoryFeature
from skill_rts import logger

logger.set_level(logger.INFO)

MAX_GENERATIONS = int(1e9)


def parse_args(config_path: str = "ace/config/pre_match/gen_response.yaml"):

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


def get_prompt_template(map_name):
    with open("ace/config/pre_match/template.yaml") as f:
        template = yaml.safe_load(f)
    RESPONSE_INSTRUCTION = template["RESPONSE_INSTRUCTION"] + "\n"
    MANUAL = template["MANUAL"] + "\n"
    STRATEGY_SPACE = template["STRATEGY_SPACE"] + "\n"
    OPPONENT = template["OPPONENT"] + "\n"
    EXIST_STRATEGY = template["EXIST_STRATEGY"] + "\n"
    METRICS = template["METRICS"] + "\n"
    TRAJECTORY = template["TRAJECTORY"] + "\n"
    MAP = template[f"{map_name}_MAP"] + "\n"
    EXAMPLES = template["EXAMPLES"] + "\n"
    RESPONSE_TIPS = template["RESPONSE_TIPS"] + "\n"
    RESPONSE_START = template["RESPONSE_START"]
    return (
        RESPONSE_INSTRUCTION
        + MANUAL
        + STRATEGY_SPACE
        + MAP
        + OPPONENT
        + EXIST_STRATEGY
        + METRICS
        + TRAJECTORY
        + EXAMPLES
        + RESPONSE_TIPS
        + RESPONSE_START
    )


def get_agents(cfg):
    map_name = cfg.env.map_path.split("/")[-1].split(".")[0]
    agents = [
        Planner(**cfg.agents[0], player_id=0, map_name=map_name), 
        Planner(**cfg.agents[1], player_id=1, map_name=map_name)
    ]
    return agents


def save_strategy(response, file_dir):
    strategy = response.split("## Description")[0]
    description = "## Description" + response.split("## Description")[1]
    structure = {
        "strategy": strategy,
        "description": description,
        "raw_response": response
    }

    if not os.path.exists(file_dir):
        os.makedirs(file_dir, exist_ok=True)

    file_path = f"{file_dir}/response_strategy.json"

    with open(file_path, "w") as f:
        json.dump(structure, f, indent=4)
    
    return strategy, file_path


def main():
    # Initialize
    cfg = parse_args()
    map_name = cfg.env.map_path.split("/")[-1].split(".")[0]
    opponent_name = cfg.agents[1].strategy.split('/')[-1].split('.')[0]
    runs_dir = f"pre_match_runs/{opponent_name}"

    llm_client = Qwen(**cfg.agents[0])
    prompt_template = get_prompt_template(map_name)
    with open(cfg.agents[1].strategy) as f:
        opponent_strategy = json.load(f)["strategy"]
    opponent_agent = Planner(**cfg.agents[1], player_id=1, map_name=map_name)

    payoffs = [-1, 1]
    trajectory, metric = None, None
    strategy = ""
    i = 0
    
    # Generate response strategy until winning
    while payoffs[0] <= 0 and cfg.max_iterations > i:
        run_dir = f"{runs_dir}/iter_{i}"

        # Generate response strategy
        traj_feats = TrajectoryFeature(trajectory) if trajectory is not None else None
        trajectory = traj_feats.to_string() if hasattr(traj_feats, "to_string") else ""
        metric = metric.to_string() if hasattr(metric, "to_string") else ""
        prompt = prompt_template.format(
            opponent=opponent_strategy,
            trajectory=trajectory,
            metric=metric,
            exist_strategy=strategy
        )
        response = llm_client(prompt)
        strategy, strategy_path = save_strategy(response, run_dir)

        # Run the game
        agent = Planner(
            player_id=0,
            map_name=map_name,
            strategy=strategy_path,
            prompt="few-shot-w-strategy",
            **cfg.agents[0]
        )
        env = MicroRTSLLMEnv([agent, opponent_agent], **cfg.env, run_dir=run_dir)
        payoffs, trajectory = env.run()
        metric = env.metric

        # Save the results
        trajectory.to_json(f"{run_dir}/raw_traj.json")
        metric.to_json(f"{run_dir}/metric.json")
        print("\r" + " " * 50 + "\r", end="", flush=True)
        print(f"Opponent {opponent_name} | Iteration {i} | Payoffs: {payoffs}")
        i += 1


if __name__ == "__main__":
    main()
