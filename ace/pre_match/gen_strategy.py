import argparse
from omegaconf import OmegaConf
import yaml
import os
from skill_rts.agents.llm_clients import Qwen
from skill_rts import logger
from ace.strategy import Strategy
import numpy as np

logger.set_level(logger.INFO)

MAX_GENERATIONS = int(1e9)
BASE_DIR = "ace/data/strategies"


def parse_args(config_path: str = "ace/pre_match/config/gen_strategy.yaml"):

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
    with open("ace/pre_match/config/template.yaml") as f:
        template = yaml.safe_load(f)
    
    INSTRUCTION = template["INSTRUCTION"] + "\n"
    MANUAL = template["MANUAL"] + "\n"
    STRATEGY_SPACE = template["STRATEGY_SPACE"] + "\n"
    EXIST_STRATEGY = template["EXIST_STRATEGY"] + "\n"
    MAP = template[f"{map}_MAP"] + "\n"
    EXAMPLES = template["EXAMPLES"] + "\n"
    TIPS = template["TIPS"] + "\n"
    START = template["START"]
    return INSTRUCTION + MANUAL + STRATEGY_SPACE + EXIST_STRATEGY + MAP + EXAMPLES + TIPS + START


def save_strategy(strategy: Strategy, map_name: str):
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR, exist_ok=True)
    
    for i in range(1, MAX_GENERATIONS):
        filename = f"{BASE_DIR}/strategy_{i}.json"
        if not os.path.exists(filename):
            break
    strategy.to_json(filename, map_name)


def extract_strategies_to_csv():
    import pandas as pd

    strategies = []
    for i in range(1, MAX_GENERATIONS):
        filename = f"{BASE_DIR}/strategy_{i}.json"
        if not os.path.exists(filename):
            break
        strategy = Strategy.load_from_json(filename)
        
        strategies.append({
            "index": i,
            "economic": strategy.economic,
            "barracks": strategy.barracks,
            "military": strategy.military,
            "aggression": strategy.aggression,
            "attack": strategy.attack,
            "defense": strategy.defense,
            "strategy": strategy.strategy,
            "description": strategy.description
        })
    
    df = pd.DataFrame(strategies)
    
    output_filename = f"{BASE_DIR}/strategies.csv"
    df.to_csv(output_filename, index=False)
    
    logger.info(f"Extracted {i - 1} strategies to {output_filename}")


def analyze_diversity():
    from scipy.spatial.distance import pdist
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import seaborn as sns

    # Load data
    batch_feats = []
    batch_feats = []
    for i in range(int(1e9)):
        filename = f"{BASE_DIR}/strategy_{i + 1}.json"
        if not os.path.exists(filename):
            break
        strategy = Strategy.load_from_json(filename)
        
        batch_feats.append(strategy.feats)
    batch_feats = np.vstack(np.array(batch_feats))
    
    logger.info(f"Loaded {batch_feats.shape[0]} strategies")
    logger.info(f"Dimension of encoded features: {batch_feats.shape[1]}")

    # Calculate average distance
    ham_dists = pdist(batch_feats, metric='hamming')
    eu_dists = pdist(batch_feats, metric='euclidean')

    avg_ham_dist = np.mean(ham_dists)
    avg_eu_dis = np.mean(eu_dists)
    max_ham_dist = np.max(ham_dists)
    max_eu_dis = np.max(eu_dists)
    min_ham_dist = np.min(ham_dists)
    min_eu_dis = np.min(eu_dists)

    logger.info(f"Average Hamming distance: {avg_ham_dist}")
    logger.info(f"Average Euclidean distance: {avg_eu_dis}")
    logger.info(f"Max Hamming distance: {max_ham_dist}")
    logger.info(f"Max Euclidean distance: {max_eu_dis}")
    logger.info(f"Min Hamming distance: {min_ham_dist}")
    logger.info(f"Min Euclidean distance: {min_eu_dis}")

    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    
    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=0)
    combined_feats = np.vstack([batch_feats, Strategy.feat_space()])

    combined_2d_tsne = tsne.fit_transform(combined_feats)

    strategy_2d_tsne = combined_2d_tsne[:len(batch_feats), :]
    all_2d_tsne = combined_2d_tsne[len(batch_feats):, :]

    # Plot the strategy distribution map
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    sns.histplot(eu_dists, bins=100, kde=True, ax=axes[0], alpha=0.7)
    axes[0].set_title("Strategy Euclidean Distance Distribution")
    axes[0].set_xlabel("Distance")
    axes[0].set_ylabel("Frequency")

    axes[1].scatter(strategy_2d_tsne[:, 0], strategy_2d_tsne[:, 1], alpha=0.7, color="blue")
    axes[1].scatter(all_2d_tsne[:, 0], all_2d_tsne[:, 1], alpha=0.1, color="red")
    axes[1].set_title("Strategy Distribution Map (t-SNE Dimensionality Reduction)")
    axes[1].set_xlabel("t-SNE Dimension 1")
    axes[1].set_ylabel("t-SNE Dimension 2")

    plt.tight_layout()
    # plt.grid(True)
    filename = "results/strategy_distribution_map.png"
    plt.savefig(filename, dpi=300)
    logger.info(f"Strategy distribution map saved to {filename}")


def gen_strategies():
    logger.info("Generating strategy...")
    # Initialize
    cfg = parse_args()
    map_name = cfg.env.map_path.split("/")[-1].split(".")[0]
    prompt_template = get_prompt_template(map_name)
    llm_client = Qwen(cfg.llm.model, cfg.llm.temperature, cfg.llm.max_tokens)
    exist_strategies = []

    # Generate strategies
    i = 1
    while i <= cfg.num_generations:
        exist_strategies_text = [strategy.strategy for strategy in exist_strategies]
        prompt = prompt_template.format(exist_strategy="\n".join(exist_strategies_text))
        response = llm_client(prompt)
        strategy = Strategy.load_from_raw(response)
        if strategy is not None and strategy not in exist_strategies:
            save_strategy(strategy, map_name)
            exist_strategies.append(strategy)
            logger.info(f"Saved strategy {i}")
            i += 1
    logger.info("🥳 Done!")


if __name__ == "__main__":
    # gen_strategies()
    analyze_diversity()
    # extract_strategies_to_csv()
