import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterator


COST = {"worker": 1, "heavy": 2, "light": 2, "ranged": 2, "base": 10, "barrack": 5}
ATTACK_DAMAGE = {"worker": 1, "heavy": 4, "light": 2, "ranged": 1, "base": 0, "barrack": 0}
ATTACK_RANGE = {"worker": 1, "heavy": 1, "light": 1, "ranged": 3, "base": 0, "barrack": 0}


@dataclass
class UnitStatus:
    owner: int
    id: int
    type: str
    location: tuple
    resource: int
    hp: int
    action: str | None
    task: str | None
    task_params: tuple | None
    cost: int | None
    attack_damage: int | None
    attack_range: int | None


@dataclass
class EnvStatus:
    height: int
    width: int
    resources: Dict[Tuple[int, int], UnitStatus]


@dataclass
class PlayerStatus:
    id: int
    resource: int
    units: dict[tuple[int, int], UnitStatus]

    @property
    def workers(self) -> list[UnitStatus]:
        return [UnitStatus for UnitStatus in self.units.values() if UnitStatus.type == "worker"]

    @property
    def bases(self) -> list[UnitStatus]:
        return [UnitStatus for UnitStatus in self.units.values() if UnitStatus.type == "base"]

    @property
    def lights(self) -> list[UnitStatus]:
        return [UnitStatus for UnitStatus in self.units.values() if UnitStatus.type == "light"]

    @property
    def barracks(self) -> list[UnitStatus]:
        return [UnitStatus for UnitStatus in self.units.values() if UnitStatus.type == "barracks"]

    @property
    def rangeds(self) -> list[UnitStatus]:
        return [UnitStatus for UnitStatus in self.units.values() if UnitStatus.type == "ranged"]

    @property
    def heavys(self) -> list[UnitStatus]:
        return [UnitStatus for UnitStatus in self.units.values() if UnitStatus.type == "heavy"]
    
    def __iter__(self) -> Iterator[UnitStatus]:
        """Iteration over each unit in units."""
        return iter(self.units.values())
    
    def __getitem__(self, location: Tuple[int, int]) -> UnitStatus:
        """Returns the UnitStatus at the specified location."""
        return self.units[location]


class Observation:
    """Observation from the environment."""
    env: EnvStatus
    players: List[PlayerStatus]
    units: Dict[Tuple[int, int], UnitStatus | None]

    def __init__(self, raw_obs: List[np.ndarray]):
        """
        Args:
            raw_obs (List[np.ndarray]): raw observation tensor from the environment
        """
        self.raw_obs = raw_obs
        self.num_players = 2
        self._init_from_raw_obs()
    
    def text(self):
        pass
    
    def _init_from_raw_obs(self):
        obs, players_resource = self.raw_obs[0], self.raw_obs[1]
        obs = np.squeeze(obs)
        players_resource = np.squeeze(players_resource)

        self.env = EnvStatus(height=obs.shape[1], width=obs.shape[2], resources={})
        self.units = {}
        for i in range(self.env.height):
            for j in range(self.env.width):
                self.units[(i, j)] = None

        self.players = []
        for owner in [-1, 1, 2]:
            if owner != -1:
                self.players.append(PlayerStatus(owner - 1, players_resource[owner - 1], {}))
            self._get_owner_units(owner, obs)

    def _get_owner_units(self, owner, obs):
        ID_INDEX = 0
        HP_INDEX = 1
        RESOURCE_INDEX = 2
        OWNER_INDEX = 3
        TYPE_INDEX = 4
        ACTION_INDEX = 5
        TYPE_MAP = {
            1: "resource",
            2: "base",
            3: "barrack",
            4: "worker",
            5: "light",
            6: "heavy",
            7: "ranged",
        }
        ACTION_MAP = {
            0: "noop",
            1: "move",
            2: "harvest",
            3: "return",
            4: "produce",
            5: "attack",
        }
        locs = np.where(obs[OWNER_INDEX] == owner)
        locs = np.array(locs).T
        for loc in locs:
            loc = tuple(loc.tolist())
            unit_type = TYPE_MAP[obs[TYPE_INDEX][loc]]
            unit = UnitStatus(
                owner=owner if owner == -1 else owner - 1,
                id=int(obs[ID_INDEX][loc]),
                type=unit_type,
                location=loc,
                hp=int(obs[HP_INDEX][loc]),
                action=ACTION_MAP[obs[ACTION_INDEX][loc]],
                resource=int(obs[RESOURCE_INDEX][loc]),
                cost=COST.get(unit_type),
                attack_damage=ATTACK_DAMAGE.get(unit_type),
                attack_range=ATTACK_RANGE.get(unit_type),
                task=None,
                task_params=None,
            )
            self.units[loc] = unit
            if owner == -1:
                self.env.resources[loc] = unit
            else:
                self.players[owner - 1].units[loc] = unit
