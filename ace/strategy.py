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
    map_size: tuple = (8, 8)  # default is  8x8 map

    def __init__(self, strategy: str, description: str):
        self.strategy = strategy
        self.description = description
        self.economic = None
        self.military = None
        self.aggression = None
        self.attack = None
        self.defense = None
    
    @classmethod
    def load_from_raw(cls, raw_response: str) -> "Strategy":  # noqa: F821
        """Load strategy from raw response"""
        strategy = raw_response.split("## Description")[0]
        description = "## Description" + raw_response.split("## Description")[1]
        try:
            instance = cls(strategy, description)
            instance._parse_feats()
            return instance
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
            self.barracks = self.barracks.split(">= ")[1]
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
        if "resource" in self.barracks:
            feats.append(int(self.barracks.split(">= ")[1]))
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
        if ">" in self.attack:
            attack_feat = [self.UNIT2IDX[unit_type.lower()] for unit_type in self.attack.split(" > ")]
        else:
            attack_feat = []
        attack_feat.extend([-1] * (6 - len(attack_feat)))
        feats.extend(attack_feat)

        # defense
        if self.defense is not None:
            locs = self.defense
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
    
    def __eq__(self, other):
        is_equal = True
        is_equal &= self.economic == other.economic
        is_equal &= self.barracks == other.barracks
        is_equal &= self.military == other.military
        is_equal &= self.aggression == other.aggression
        is_equal &= self.attack == other.attack
        is_equal &= self.defense == other.defense
        return is_equal
    
    def to_json(self, filename):
        import os
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        structure = {
            "economic": self.economic,
            "barracks": self.barracks,
            "military": self.military,
            "aggression": self.aggression,
            "attack": self.attack,
            "defense": self.defense,
            "strategy": self.strategy,
            "description": self.description,
        }
        with open(filename, "w") as f:
            json.dump(structure, f, indent=4)
    
    @classmethod
    def load_from_json(cls, filename) -> "Strategy":
        """Load a strategy from a JSON file"""
        with open(filename, "r") as f:
            structure = json.load(f)
        instance = cls(structure["strategy"], structure["description"])
        instance.economic = structure["economic"]
        instance.barracks = structure["barracks"]
        instance.military = structure["military"]
        instance.aggression = structure["aggression"]
        instance.attack = structure["attack"]
        instance.defense = structure["defense"]
        return instance
    