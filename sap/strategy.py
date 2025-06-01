import numpy as np
import json
import re


class Strategy:

    UNIT2IDX = {
        "worker": 0,
        "heavy": 1,
        "light": 2,
        "ranged": 3,
        "base": 4,
        "barracks": 5,
    }

    IDX2UNIT = {
        0: "worker",
        1: "heavy",
        2: "light",
        3: "ranged",
        4: "base",
        5: "barracks",
    }
    _map_size: tuple = (8, 8)  # default is  8x8 map

    def __init__(self, strategy: str, description: str=""):
        self.strategy = strategy
        self.description = description
        self.economic = None
        self.barracks = None
        self.military = None
        self.aggression = None
        self.attack = None
        self.defense = None

        self._parse_feats()
    
    @classmethod
    def load_from_raw(cls, raw_response: str) -> "Strategy":  # noqa: F821
        """Load strategy from raw response"""
        strategy = raw_response.split("## Description")[0]
        description = "## Description" + raw_response.split("## Description")[1]
        try:
            return cls(strategy, description)
        except Exception as e:
            print(e)
            return None
    
    def _parse_feats(self):
        self.economic = re.search(r'.*?Economic Feature.*?: (\w+)', self.strategy).group(1)
        self.barracks = re.search(r'.*?Barracks Feature.*?: ([^\(\#]*)', self.strategy).group(1)
        self.military = re.search(r'.*?Military Feature.*?: ([^\(\#]*)', self.strategy).group(1)
        self.aggression = re.search(r'.*?Aggression Feature.*?: (\w+)', self.strategy).group(1)
        self.attack = re.search(r'.*?Attack Feature.*?: (\w+)', self.strategy).group(1)
        self.defense = re.search(r'.*?Defense Feature.*?: (\w+)', self.strategy).group(1)

        self.economic = int(self.economic)
        if "False" in self.barracks:
            self.barracks = False
        else:
            self.barracks = float(re.search(r'resource\s*>=\s*(\d+)', self.barracks).group(1))
        self.aggression = eval(self.aggression)
        if not self.aggression:
            self.attack = None
        self.defense = eval(self.defense)

    
    def encode(self) -> np.ndarray:
        # 2 + 2 + 4 + 1 + 2 + 1 = 12
        # economic
        economy_feat = [1, 0] if self.economic == 1 else [0, 1]
        
        # barracks
        if self.barracks:
            # normalize
            barracks_feat = [1, (min(self.barracks, 10) - 5) / 5]
        else:
            barracks_feat = [0, 0]
        
        # military
        militaries = [self.UNIT2IDX[unit_type.lower()] for unit_type in self.military.split(" and ") if unit_type.lower() in self.UNIT2IDX]
        military_feat = [1 if idx in militaries else 0 for idx in range(4)]
        
        # aggression
        aggression_feat = [1] if self.aggression else [0]
        
        # attack
        if self.attack is not None:
            attack_feat = [1, 0] if self.attack.lower() == "building" else [0, 1]
        else:
            attack_feat = [0] * 2

        # defense
        if self.defense is not None:
            # normalization
            defense_feat = [self.defense / self._map_size[0]]
        else:
            defense_feat = [0]

        return np.array(economy_feat + barracks_feat + military_feat + aggression_feat + attack_feat + defense_feat)
    
    @property
    def feats(self) -> np.ndarray:
        return self.encode()
    
    @classmethod
    def decode(cls, feats: np.ndarray) -> "Strategy":
        """Decode features to strategy"""
        economic = 1 if feats[0] == 1 else 2
        barracks = f"resource >= {int(feats[3] * 5 + 5)}" if feats[2] == 1 else False
        military = map(lambda i: cls.IDX2UNIT[i].capitalize(), np.where(feats[4 : 8] == 1)[0])
        military = " and ".join(military)
        aggression = True if feats[8] == 1 else False
        if aggression:
            attack = "Building" if feats[9] == 1 else "Unit"
            defense = None
        else:
            defense = int(feats[11] * cls._map_size[0])
            attack = None

        strategy = "## Strategy\n"
        strategy += f"Economic Feature: {economic}\n"
        strategy += f"Barracks Feature: {barracks}\n"
        strategy += f"Military Feature: {military}\n"
        strategy += f"Aggression Feature: {aggression}\n"
        strategy += f"Attack Feature: {attack}\n"
        strategy += f"Defense Feature: {defense}\n"

        return cls(strategy, "")

    def __eq__(self, other):
        return np.array_equal(self.feats, other.feats)
    
    def to_json(self, filename, map_name):
        import os
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        self.id = int(filename.split("/")[-1].split(".")[0].split("_")[-1])
        structure = {
            "id": self.id,
            "economic": self.economic,
            "barracks": self.barracks,
            "military": self.military,
            "aggression": self.aggression,
            "attack": self.attack,
            "defense": self.defense,
            "strategy": self.strategy,
            "description": self.description,
            "map": map_name
        }
        with open(filename, "w") as f:
            json.dump(structure, f, indent=4)
    
    @classmethod
    def load_from_json(cls, filename) -> "Strategy":
        """Load a strategy from a JSON file"""
        import re

        with open(filename, "r") as f:
            structure = json.load(f)
        instance = cls(structure["strategy"], structure["description"])
        instance.id = structure["id"]
        instance.economic = structure["economic"]
        instance.barracks = structure["barracks"]
        instance.military = structure["military"]
        instance.aggression = structure["aggression"]
        instance.attack = structure["attack"]
        instance.defense = structure["defense"]
        instance.strategy = structure["strategy"]
        instance.description = structure["description"]
        match = re.search(r"(\d+)x(\d+)", structure["map"])
        if match:
            instance._map_size = (int(match.group(1)), int(match.group(2)))
        return instance
    
    @staticmethod
    def feat_space() -> np.ndarray:
        """Return the feature space of the strategy"""
        import itertools
        
        economic_space = [[1, 0], [0 ,1]]  # 2
        barracks_space = [[1, i / 5] for i in range(6)]  # 6 + 1 ([0, 0])
        military_space = [
            [1 if i in feats else 0 for i in range(4)]
            for i in range(1, 5)
            for feats in itertools.combinations(range(4), i)
        ]  # C(4, 1) + C(4, 2) + C(4, 3) + C(4, 4) = 15
        military_space.remove([1, 0, 0, 0])
        attack_space = [[1, 0], [0, 1]]  # 2
        defense_space = [[i / Strategy._map_size[0]] for i in range(1, 5)]  # 4

        # 2 * 7 * 15 * 2 + 2 * 7 * 15 * 4 = 1,260
        feat_space = list(
            itertools.product(
                economic_space,
                [[0, 0]],
                [[1, 0, 0, 0]],  # worker rush
                [[1]],  # aggressive
                attack_space,
                [[0]],
            )
        ) + list(
            itertools.product(
                economic_space,
                barracks_space,
                military_space,  # advanced military
                [[1]],  # aggressive
                attack_space,
                [[0]]
            )
        ) + list(
            itertools.product(
                economic_space,
                [[0, 0]],
                [[1, 0, 0, 0]],  # worker rush
                [[0]],  # defensive
                [[0, 0]],
                defense_space,
            )
        ) + list(
            itertools.product(
                economic_space,
                barracks_space,
                military_space,  # advanced military
                [[0]],
                [[0, 0]],
                defense_space
            )
        )
        feat_space = [np.hstack(feats) for feats in feat_space]

        return np.vstack(feat_space)
    
    def __str__(self):
        return self.strategy + self.description
    
    def to_string(self):
        return self.strategy + self.description
    
    def __hash__(self):
        return hash(self.strategy)


if __name__ == "__main__":
    feat_space = Strategy.feat_space()
    print(feat_space.shape)
    print(feat_space[-1])
    strategy = Strategy.decode(feat_space[-1])
    print(strategy.strategy)
    strategy = Strategy(strategy.strategy, "")
    print(strategy.feats)
