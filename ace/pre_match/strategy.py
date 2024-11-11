import numpy as np


class Strategy:

    UNIT2IDX = {
        "worker": 0,
        "heavy": 1,
        "light": 2,
        "ranged": 3,
        "base": 4,
        "barracks": 5,
    }
    map_size: int = (8, 8)  # default is  8x8 map

    def __init__(self, raw_strategy: str):
        self.raw = raw_strategy
        self.parse_feats()
    
    def parse_feats(self):
        self.economic = self.raw.split("Economic Feature: ")[1].split("\n")[0].strip()
        self.barracks = self.raw.split("Barracks Feature: ")[1].split("\n")[0].strip()
        self.military = self.raw.split("Military Feature: ")[1].split("\n")[0].strip()
        self.aggression = self.raw.split("Aggression Feature: ")[1].split("\n")[0].strip()
        self.attack = self.raw.split("Attack Feature: ")[1].split("\n")[0].strip()
        self.defense = self.raw.split("Defense Feature: ")[1].split("\n")[0].strip()
    
    def encode(self) -> np.ndarray:
        feats = []
        
        # economic
        feats.append(int(self.economic))
        
        # barracks
        if "resource" in self.barracks:
            feats.append(int(self.barracks.split(">= ")[1]))
        else:
            feats.append(-1)
        
        # military
        military_feat = [self.UNIT2IDX[unit_type.lower()] for unit_type in self.military.split(" and ")]
        military_feat.extend([-1] * (4 - len(military_feat)))
        feats.extend(military_feat)
        
        # aggression
        aggression_feat = 1 if self.aggression == "True" else 0
        feats.append(aggression_feat)
        
        # attack
        if ">" in self.attack:
            attack_feat = [self.UNIT2IDX[unit_type.lower()] for unit_type in self.attack.split(" > ")]
        else:
            attack_feat = []
        attack_feat.extend([-1] * (6 - len(attack_feat)))
        feats.extend(attack_feat)

        # defense
        if self.defense != "None":
            locs = eval(self.defense)
            defense_feat = list(map(lambda loc: loc[0] * self.map_size[0] + loc[1], locs))
        else:
            defense_feat = [-1, -1]
        feats.extend(defense_feat)

        return np.array(feats)
    
    @property
    def feats(self) -> np.ndarray:
        return self.encode()
    
    @property
    def one_hot_feats(self) -> np.ndarray:
        feats_size = 1 + 6 + 4 + 1 + 6 * 6 + self.map_size[0] * self.map_size[1]
        one_hot_feats = np.zeros((feats_size), dtype=int)

        # economic
        one_hot_feats[0] = self.feats[0] - 1

        # barracks
        if self.feats[1] != -1:
            one_hot_feats[self.feats[1] - 5 + 1] = 1
        
        # military
        military_feat = np.sort(self.feats[2:6])
        military_feat = military_feat[np.where(military_feat != -1)]
        one_hot_feats[military_feat + 7] = 1

        # aggression
        one_hot_feats[11] = self.feats[6]

        # attack
        attack_feat = self.feats[7:13]
        attack_feat = attack_feat[np.where(attack_feat != -1)]
        for i, feat in enumerate(attack_feat):
            one_hot_feats[feat + 12 + i * 6] = 1
        
        # defense
        defense_feat = self.feats[13:]
        left_upper, right_lower = tuple(map(lambda x: (x // self.map_size[0], x % self.map_size[0]), defense_feat))
        for x in range(left_upper[0], right_lower[0] + 1):
            for y in range(left_upper[1], right_lower[1] + 1):
                one_hot_feats[x * self.map_size[0] + y + 48] = 1

        return one_hot_feats


def extract_strategies_to_csv():
    import os
    import json

    text = "index,economic,barracks,military,aggression,attack,defense\n"
    for i in range(int(1e9)):
        filename = f"ace/data/opponent_strategy_{i + 1}.json"
        if not os.path.exists(filename):
            break
        with open(filename) as f:
            raw_strategy = json.load(f)["strategy"]
        strategy = Strategy(raw_strategy)

        text += f"{i}, "
        text += f"{strategy.economic},"
        text += f"{strategy.barracks},"
        text += f"{strategy.military},"
        text += f"{strategy.aggression},"
        text += f"{strategy.attack},"
        text += f'"{strategy.defense}"\n'
    with open("temp/strategy.csv", "w") as f:
        f.write(text)


def analyze_diversity():
    import json
    from skill_rts import logger
    from scipy.spatial.distance import pdist
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import os

    # Load data
    batch_feats = []
    batch_one_hot_feats = []
    for i in range(int(1e9)):
        filename = f"ace/data/opponent_strategy_{i + 1}.json"
        if not os.path.exists(filename):
            break
        with open(filename) as f:
            raw_strategy = json.load(f)["strategy"]
        strategy = Strategy(raw_strategy)
        
        batch_feats.append(strategy.feats)
        batch_one_hot_feats.append(strategy.one_hot_feats)
   
    batch_feats = np.array(batch_feats)
    batch_one_hot_feats = np.array(batch_one_hot_feats)

    logger.info(f"Loaded {batch_feats.shape[0]} strategies")
    logger.info(f"Dimension of encoded features: {batch_feats.shape[1]}")
    logger.info(f"Dimension of one-hot features: {batch_one_hot_feats.shape[1]}")

    # Calculate average distance
    ham_dists = pdist(batch_one_hot_feats, metric='hamming')
    eu_dists = pdist(batch_one_hot_feats, metric='euclidean')

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

    # PCA
    pca = PCA(n_components=2)
    strategy_2d_pca = pca.fit_transform(batch_one_hot_feats)

    # or t-SNE (suitable for high-dimensional data)
    tsne = TSNE(n_components=2, perplexity=30, random_state=0)
    strategy_2d_tsne = tsne.fit_transform(batch_one_hot_feats)

    # Plot the strategy distribution map
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    axes[0].scatter(strategy_2d_pca[:, 0], strategy_2d_pca[:, 1], alpha=0.7)
    axes[0].set_title("Strategy Distribution Map (PCA Dimensionality Reduction)")
    axes[0].set_xlabel("PCA Dimension 1")
    axes[0].set_ylabel("PCA Dimension 2")

    axes[1].scatter(strategy_2d_tsne[:, 0], strategy_2d_tsne[:, 1], alpha=0.7)
    axes[1].set_title("Strategy Distribution Map (t-SNE Dimensionality Reduction)")
    axes[1].set_xlabel("t-SNE Dimension 1")
    axes[1].set_ylabel("t-SNE Dimension 2")

    plt.tight_layout()
    # plt.grid(True)
    plt.savefig("temp/strategy_distribution_map.png", dpi=300)



if __name__ == "__main__":
    analyze_diversity()
    extract_strategies_to_csv()