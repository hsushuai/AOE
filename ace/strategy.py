import numpy as np
import json


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
        self.economic = self.strategy.split("Economic Feature: ")[1].split("\n")[0].strip()
        self.barracks = self.strategy.split("Barracks Feature: ")[1].split("\n")[0].strip()
        self.military = self.strategy.split("Military Feature: ")[1].split("\n")[0].strip()
        self.aggression = self.strategy.split("Aggression Feature: ")[1].split("\n")[0].strip()
        self.attack = self.strategy.split("Attack Feature: ")[1].split("\n")[0].strip()
        self.defense = self.strategy.split("Defense Feature: ")[1].split("\n")[0].strip()

        self.economic = int(self.economic)
        if "resource" in self.barracks:
            self.barracks = int(self.barracks.split(">= ")[1])
        else:
            self.barracks = False
        self.aggression = eval(self.aggression)
        if not self.aggression:
            self.attack = eval(self.attack)
        self.defense = eval(self.defense)

    
    def encode(self) -> np.ndarray:
        feats = []
        
        # economic
        feats.append(self.economic)
        
        # barracks
        if isinstance(self.barracks, str):
            if "resource" in self.barracks:
                feats.append(self.barracks.split(">= ")[1])
            else:
                feats.append(-1)
        elif isinstance(self.barracks, int):
            feats.append(self.barracks)
        else:
            feats.append(-1)        
        
        # military
        military_feat = [self.UNIT2IDX[unit_type.lower()] for unit_type in self.military.split(" and ")]
        military_feat.extend([-1] * (4 - len(military_feat)))
        feats.extend(military_feat)
        
        # aggression
        aggression_feat = 1 if self.aggression else 0
        feats.append(aggression_feat)
        
        # attack
        if isinstance(self.attack, str) and ">" in self.attack:
            attack_feat = [self.UNIT2IDX[unit_type.lower()] for unit_type in self.attack.split(" > ")]
        else:
            attack_feat = []
        attack_feat.extend([-1] * (6 - len(attack_feat)))
        feats.extend(attack_feat)

        # defense
        if self.defense is not None:
            locs = self.defense
            defense_feat = list(map(lambda loc: loc[0] * self._map_size[0] + loc[1], locs))
        else:
            defense_feat = [-1, -1]
        feats.extend(defense_feat)

        return np.array(feats)
    
    @property
    def feats(self) -> np.ndarray:
        return self.encode()
    
    @property
    def one_hot_feats(self) -> np.ndarray:
        feats_size = 1 + 1 + 4 + 1 + 6 * 6 + self._map_size[0] * self._map_size[1]
        one_hot_feats = np.zeros((feats_size), dtype=int)

        # economic
        one_hot_feats[0] = self.feats[0]

        # barracks
        one_hot_feats[1] = self.feats[1]
        
        # military
        military_feat = np.sort(self.feats[2 : 6])
        military_feat = military_feat[np.where(military_feat != -1)]
        one_hot_feats[military_feat + 2] = 1

        # aggression
        one_hot_feats[6] = self.feats[6]

        # attack
        if one_hot_feats[6] == 1:
            attack_feat = self.feats[7 : 13]
            attack_feat = attack_feat[np.where(attack_feat != -1)]
            for i, feat in enumerate(attack_feat):
                one_hot_feats[feat + 7 + i * 6] = 1
        # defense
        else:
            defense_feat = self.feats[13:]
            left_upper, right_lower = tuple(map(lambda x: (x // self._map_size[0], x % self._map_size[0]), defense_feat))
            for x in range(left_upper[0], right_lower[0] + 1):
                for y in range(left_upper[1], right_lower[1] + 1):
                    one_hot_feats[x * self._map_size[0] + y + 43] = 1

        return one_hot_feats
    
    @classmethod
    def decode(cls, feats: np.ndarray, one_hot=True) -> "Strategy":
        """Decode features to strategy"""
        economic = feats[0]
        barracks = f"resource >= {feats[1]}"
        if one_hot:
            military = map(lambda i: cls.IDX2UNIT[i].capitalize(), np.where(feats[2 : 6] == 1)[0])
        else:
            military = [cls.IDX2UNIT[i].capitalize() for i in feats[2 : 6] if i != -1]
        military = " and ".join(military)
        aggression = True if feats[6] == 1 else False
        if aggression:
            if one_hot:
                attack = [np.where(feats[7 + 6 * (i - 1) : 7 + 6 * i] == 1)[0] for i in range(1, 7)]
                attack = [cls.IDX2UNIT[i[0]].capitalize() for i in attack if i.size > 0]
            else:
                attack = [cls.IDX2UNIT[i].capitalize() for i in feats[7 : 13]]
            attack = " > ".join(attack)
            defense = None
        else:
            if one_hot:
                defense = np.where(feats[43:] == 1)[0]
                defense = (defense[0], defense[-1])
            else:
                defense = feats[13:]
            defense = tuple(map(lambda i: (i // cls._map_size[0], i % cls._map_size[0]), defense))
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
        return np.array_equal(self.one_hot_feats, other.one_hot_feats)
    
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
        
        economic_space = [1, 2]  # 2
        barracks_space = [5, 6, 7, 8, 9, 10]  # 6
        military_space = [
            list(feats) + [-1] * (4 - len(feats))
            for i in range(1, 5)
            for feats in itertools.combinations(range(4), i)]  # 15
        aggression_space = [1]  # aggressive only
        attack_space = list(itertools.permutations(range(6), 6))  # P(6, 4) = 360
        defense_space = [[-1, -1]]

        # 2 * 6 * 15 * 1 * 720 * 1 = 129,600
        feat_space = list(itertools.product(
            economic_space,
            barracks_space,
            military_space,
            aggression_space,
            attack_space,
            defense_space
        ))
        feat_space = [
            [economic, barracks] + list(military) + [aggression] + list(attack) + list(defense)
            for economic, barracks, military, aggression, attack, defense in feat_space
        ]
        return np.array(feat_space)
    
    def __str__(self):
        return self.strategy + self.description
    
    def to_string(self):
        return self.strategy + self.description


if __name__ == "__main__":
    feat_space = Strategy.feat_space()
    print(feat_space.shape)
    print(feat_space[0])
    strategy = Strategy.decode(feat_space[0], False)
    print(strategy.strategy)