import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterator


COST = {"worker": 1, "heavy": 2, "light": 2, "ranged": 2, "base": 10, "barracks": 5}
ATTACK_DAMAGE = {"worker": 1, "heavy": 4, "light": 2, "ranged": 1, "base": 0, "barracks": 0}
ATTACK_RANGE = {"worker": 1, "heavy": 1, "light": 1, "ranged": 3, "base": 0, "barracks": 0}
ACTION = {0: "noop", 1: "move", 2: "harvest", 3: "return", 4: "produce", 5: "attack"}
DIRECTION = {0: "north", 1: "east", 3: "south", 4: "west"}


@dataclass
class UnitState:
    owner: int
    id: int
    type: str
    location: tuple
    resource: int
    hp: int
    action: str = None
    action_params: str | tuple = None
    task: str = None
    task_params: tuple = None
    cost: int = None
    attack_damage: int = None
    attack_range: int = None


@dataclass
class EnvState:
    height: int
    width: int
    resources: Dict[Tuple[int, int], UnitState]


@dataclass
class PlayerState:
    id: int
    resource: int
    units: dict[tuple[int, int], UnitState]

    @property
    def workers(self) -> list[UnitState]:
        return [UnitState for UnitState in self.units.values() if UnitState.type == "worker"]

    @property
    def bases(self) -> list[UnitState]:
        return [UnitState for UnitState in self.units.values() if UnitState.type == "base"]

    @property
    def lights(self) -> list[UnitState]:
        return [UnitState for UnitState in self.units.values() if UnitState.type == "light"]

    @property
    def barracks(self) -> list[UnitState]:
        return [UnitState for UnitState in self.units.values() if UnitState.type == "barracks"]

    @property
    def rangeds(self) -> list[UnitState]:
        return [UnitState for UnitState in self.units.values() if UnitState.type == "ranged"]

    @property
    def heavys(self) -> list[UnitState]:
        return [UnitState for UnitState in self.units.values() if UnitState.type == "heavy"]
    
    def __iter__(self) -> Iterator[UnitState]:
        """Iteration over each unit in units."""
        return iter(self.units.values())
    
    def __getitem__(self, location: Tuple[int, int]) -> UnitState:
        """Returns the UnitState at the specified location."""
        return self.units.get(location)


class GameState:
    """GameState from the environment."""
    env: EnvState
    players: List[PlayerState]
    units: Dict[Tuple[int, int], UnitState | None]

    def __init__(self, raw_obs: list[np.ndarray] = None, raw_entry: dict = None):
        """
        Initialize the game state from raw observations or an entry dictionary.
        
        Args:
            raw_obs (List[np.ndarray], optional): Raw observation tensor from the environment.
                If provided, it will be used to initialize the state.
            raw_entry (dict, optional): Raw entry data containing game state information.
                This will be used to initialize the state if raw_obs is not provided.
        
        Raises:
            ValueError: If neither raw_obs nor raw_entry is provided, initialization cannot proceed.
        """
        self.num_players = 2
        if raw_obs is not None:
            self._init_from_raw_obs(raw_obs[0])
        elif raw_entry is not None:
            self._init_from_raw_entry(raw_entry)
        else:
            raise ValueError("Initialization failed: Either 'raw_obs' or 'raw_entry' must be provided.")

    
    def to_string(self):
        # TODO: Implement
        pass

    def get_player_obs(self, player_id: int, partial=False):
        """Get player observation in the game."""
        if not partial:
            return self
        else:
            raise NotImplementedError("Partial observation mode is not implemented.")
    
    def _init_from_raw_obs(self, raw_obs):
        obs, players_resource = raw_obs[0], raw_obs[1]
        obs = np.squeeze(obs)
        players_resource = np.squeeze(players_resource)

        self.env = EnvState(height=obs.shape[1], width=obs.shape[2], resources={})
        self.units = {}
        for i in range(self.env.height):
            for j in range(self.env.width):
                self.units[(i, j)] = None

        self.players = []
        for owner in [-1, 1, 2]:
            if owner != -1:
                self.players.append(PlayerState(owner, players_resource[owner - 1], {}))
            self._get_owner_units(owner, obs)
    
    def _init_from_raw_entry(self, raw_entry: dict):
        self.env = EnvState(height=raw_entry["pgs"]["height"], width=raw_entry["pgs"]["width"], resources={})
        self.units = {}
        for i in range(self.env.height):
            for j in range(self.env.width):
                self.units[(i, j)] = None
        self.players = []
        self.players.append(PlayerState(1, raw_entry["pgs"]["players"][0]["resources"], {}))
        self.players.append(PlayerState(2, raw_entry["pgs"]["players"][0]["resources"], {}))
        actions = raw_entry["actions"]
        actions = {a["unitID"]: a["action"] for a in actions}
        for unit in raw_entry["pgs"]["units"]:
            loc = (unit["x"], unit["y"])
            a = self._get_action(actions.get(unit["ID"]))
            u = UnitState(
                id=unit["ID"],
                owner=unit["player"] + 1,
                type=unit["type"],
                location=loc,
                resource=unit["resources"],
                hp=unit["hitpoints"],
                action=a["type"],
                action_params=a["parameter"],
            )
            self.units[loc] = u
            if unit["player"] == -1:
                self.env.resources[loc] = u
            else:
                self.players[unit["player"]].units[loc] = u
    
    def _get_action(self, action: dict | None):
        if action is None:
            return {"type": None, "parameter": None}
        a_t = ACTION.get(action["type"])
        if a_t == "attack":
            return {"type": a_t, "parameter": (action["x"], action["y"])}
        elif a_t in ["harvest", "move", "return"]:
            return {"type": a_t, "parameter": DIRECTION[action["parameter"]]}
        elif a_t == "produce":
            return {"type": a_t, "parameter": (DIRECTION[action["parameter"]], action["unitType"].lower())}
        else:  # noop
            return {"type": a_t, "parameter": None}

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
            3: "barracks",
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
            unit = UnitState(
                owner=owner,
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
    
    def __iter__(self) -> Iterator[UnitState]:
        """Iteration over each unit in units."""
        return iter(self.units.values())
    
    def __getitem__(self, location: Tuple[int, int]) -> UnitState:
        """Returns the UnitState at the specified location."""
        return self.units.get(location)
