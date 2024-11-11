import argparse
from omegaconf import OmegaConf
import yaml
import json
import os
from skill_rts.envs.wrappers import MicroRTSLLMEnv
from skill_rts.agents.llm_clients import Qwen
from ace.agent  import AceAgent
from ace.traj_feat import TrajectoryFeature
from skill_rts.agents import bot_agent
from tqdm.rich import tqdm


def parse_args(config_path: str = "ace/configs/pre_match/gen_response.yaml"):

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
    template = OmegaConf.load("ace/configs/in_match/template.yaml")
    recognize_template = template["RECOGNIZE_TEMPLATE"]
    response_template = template["RESPONSE_TEMPLATE"]
    return recognize_template, response_template


def get_agents(cfg):
    map_name = cfg.env.map_path.split("/")[-1].split(".")[0]
    agents = [
        AceAgent(**cfg.agents[0], player_id=0, map_name=map_name), 
        AceAgent(**cfg.agents[1], player_id=1, map_name=map_name)
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
    # opponent_name = cfg.agents[1].strategy.split('/')[-1].split('.')[0]
    opponent_name = "WorkerRush"
    runs_dir = f"in_match_runs/{opponent_name}"

    llm_client = Qwen(**cfg.agents[0])
    recognize_template, response_template = get_prompt_template(map_name)
    opponent_agent = AceAgent(**cfg.agents[1], player_id=1, map_name=map_name)

    payoffs = [-1, 1]
    trajectory, metric = None, None
    i = 0
    
    # Generate response strategy until winning
    while payoffs[0] <= 0 and cfg.max_iterations > i:
        run_dir = f"{runs_dir}/iter_{i}"
        os.makedirs(run_dir, exist_ok=True)
        debug_file = open(f"{run_dir}/debug.log", "w")

        # Recognize opponent strategy
        print("\rRecognizing opponent strategy", end="", flush=True)
        traj_feats = TrajectoryFeature(trajectory) if trajectory is not None else None
        trajectory = traj_feats.to_string() if hasattr(traj_feats, "to_string") else ""
        metric = metric.to_string() if hasattr(metric, "to_string") else ""
        recognize_prompt = recognize_template.format(trajectory=trajectory)
        opponent_strategy = llm_client(recognize_prompt)
        
        print(f"Recognized prompt:\n{recognize_prompt}", file=debug_file, flush=True)
        print(f"Recognized Opponent strategy:\n{opponent_strategy}", file=debug_file, flush=True)
        
        # Generate response strategy
        print("\r" + " " * 30 + "\r", end="", flush=True)
        print("\rGenerating response strategy", end="", flush=True)
        response_prompt = response_template.format(opponent=opponent_strategy)
        response = llm_client(response_prompt)
        _, strategy_path = save_strategy(response, run_dir)
        print(f"Generated prompt:\n{response_prompt}", file=debug_file, flush=True)
        print(f"Generated Response strategy:\n{response}", file=debug_file, flush=True)

        # Run the game
        print("\r" + " " * 30 + "\r", end="", flush=True)
        print("\rRunning the game", end="", flush=True)
        agent = AceAgent(
            player_id=0,
            map_name=map_name,
            strategy=strategy_path,
            prompt="few-shot-w-strategy",
            **cfg.agents[0]
        )
        env = MicroRTSLLMEnv([agent, bot_agent.workerRushAI], **cfg.env, run_dir=run_dir)
        payoffs, trajectory = env.run()
        metric = env.metric

        # Save the results
        print("\r" + " " * 30 + "\r", end="", flush=True)
        trajectory.to_json(f"{run_dir}/raw_traj.json")
        metric.to_json(f"{run_dir}/metric.json")
        print(f"Opponent {opponent_name} | Iteration {i} | Payoffs: {payoffs}")
        i += 1
        debug_file.close()


if __name__ == "__main__":
    main()