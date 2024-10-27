from skill_rts.game.game_state import GameState, UnitState
import json

ACTION = {0: "noop", 1: "move", 2: "harvest", 3: "return", 4: "produce", 5: "attack"}
DIRECTION = {0: "north", 1: "east", 3: "south", 4: "west"}


class Trajectory:
    def __init__(self, raw_traj):
        self.raw_traj = json.loads(raw_traj)

    def update(self, obs: GameState):
        ...
    
    def get_player_trajectory(self, player_id: int=None):
        p1_traj, p2_traj = {}, {}
        for entry in self.raw_traj["entries"]:
            ...
    
    def get_gametime(self):
        """Get the latest game time."""
        return self.raw_traj["entries"][-1]["time"]
    
    def get_game_state(self, gametime):
        """Get the game state at the specified time."""
        entry = next((e for e in self.raw_traj["entries"] if e["time"] == gametime), None)
        return entry["pgs"] if entry else None

    def get_actions(self, gametime):
        entry = next((e for e in self.raw_traj["entries"] if e["time"] == gametime), None)
        if entry:
            return self._get_actions(entry["actions"], entry["pgs"]["units"])
        return [], []

    def _get_actions(self, raw_actions: list, units: list):
        pa1, pa2 = [], []
        unit_dict = {unit["ID"]: unit for unit in units}
        for action in raw_actions:
            unit_id = action["unitID"]
            a = self._get_action(action)
            unit = unit_dict.get(unit_id)
            if unit:
                u = UnitState(
                    id=unit["ID"],
                    owner=unit["player"],
                    type=unit["type"],
                    location=(unit["x"], unit["y"]),
                    resource=unit["resources"],
                    hp=unit["hitpoints"],
                    action=a["type"],
                    action_params=a["parameter"]
                )
                (pa1 if unit_id == 1 else pa2).append(u)
        return pa1, pa2
    
    def _get_action(self, action: dict):
        a_t = ACTION[action["type"]]
        if a_t == "attack":
            return {"type": a_t, "parameter": (action["x"], action["y"])}
        elif a_t in ["harvest", "move", "return"]:
            return {"type": a_t, "parameter": DIRECTION[action["parameter"]]}
        elif a_t == "produce":
            return {"type": a_t, "parameter": (DIRECTION[action["parameter"]], action["unitType"].lower())}
        else:
            return {"type": a_t, "parameter": None}  # noop case

