import argparse
from omegaconf import OmegaConf
import yaml
import json
import os
from skill_rts.agents.llm_clients import Qwen
from skill_rts import logger

logger.set_level(logger.INFO)

MAX_GENERATIONS = int(1e9)

def parse_args(config_path: str = "/root/desc/skill-rts/ACE/pre_match/config.yaml"):

    cfg = OmegaConf.load(config_path)

    parser = argparse.ArgumentParser()
    parser.add_argument("--map_path", type=str, help="Path to the map file")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--temperature", type=float, help="Temperature for LLM")
    parser.add_argument("--max_tokens", type=int, help="Maximum tokens for LLM")
    parser.add_argument("--num_generations", type=int, help="Number of generations for LLM")

    args = parser.parse_args()

    if args.map_path is not None:
        cfg.env.map_path = args.map_path
    if args.model is not None:
        cfg.llm.model = args.model
    if args.temperature is not None:
        cfg.llm.temperature = args.temperature
    if args.max_tokens is not None:
        cfg.llm.max_tokens = args.max_tokens
    if args.num_generations is not None:
        cfg.num_generations = args.num_generations
    
    return cfg


def get_prompt_template(map):
    with open("ACE/pre_match/opponent_gen.yaml") as f:
        template = yaml.safe_load(f)
    
    OPPONENT_INSTRUCTION = template["OPPONENT_INSTRUCTION"] + "\n"
    MANUAL = template["MANUAL"] + "\n"
    STRATEGY_SPACE = template["STRATEGY_SPACE"] + "\n"
    EXIST_STRATEGY = template["EXIST_STRATEGY"] + "\n"
    MAP = template[f"{map}_MAP"] + "\n"
    EXAMPLES = template["EXAMPLES"] + "\n"
    OPPONENT_START = template["OPPONENT_START"]
    return OPPONENT_INSTRUCTION + MANUAL + STRATEGY_SPACE + EXIST_STRATEGY + MAP + EXAMPLES + OPPONENT_START


def save_opponent(response):
    strategy = response.split("## Description")[0]
    description = "## Description" + response.split("## Description")[1]
    structure = {
        "strategy": strategy,
        "description": description,
        "raw_response": response
    }

    for i in range(1, MAX_GENERATIONS):
        filename = f"ACE/data/opponent_strategy_{i}.json"
        if not os.path.exists(filename):
            break
    
    with open(filename, "w") as f:
        json.dump(structure, f, indent=4)


def load_existing_strategy():
    strategies = []
    for i in range(1, MAX_GENERATIONS):
        filename = f"ACE/data/opponent_strategy_{i}.json"
        if not os.path.exists(filename):
            break
        with open(filename, "r") as f:
            strategies.append(json.load(f)["strategy"])
    return strategies


def display_existing_strategy():
    strategies = []
    for i in range(1, MAX_GENERATIONS):
        filename = f"ACE/data/opponent_strategy_{i}.json"
        if not os.path.exists(filename):
            break
        with open(filename, "r") as f:
            strategies.append(json.load(f)["raw_response"])
    
    logger.info(f"Loaded {len(strategies)} existing strategies")
    with open("temp/strategy.txt", "w") as f:
        f.write("\n\n".join(strategies))


def main():
    # Initialize
    cfg = parse_args()
    map_name = cfg.env.map_path.split("/")[-1].split(".")[0]
    prompt_template = get_prompt_template(map_name)
    llm_client = Qwen(cfg.llm.model, cfg.llm.temperature, cfg.llm.max_tokens)

    # Generate strategies
    for i in range(cfg.num_generations):
        exist_strategies = load_existing_strategy()
        prompt = prompt_template.format(exist_strategy="\n".join(exist_strategies))
        response = llm_client(prompt)
        if response is not None:
            save_opponent(response)
            logger.info(f"Saved opponent strategy {i + 1}")


if __name__ == "__main__":
    # logger.info("Generating opponent strategy...")
    # main()
    # logger.info("🥳 Done!")
    display_existing_strategy()
