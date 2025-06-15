from skill_rts.game.game_state import GameState
from typing import Iterator

ACTION = {0: "noop", 1: "move", 2: "harvest", 3: "return", 4: "produce", 5: "attack"}
DIRECTION = {0: "north", 1: "east", 3: "south", 4: "west"}


class Trajectory:
    def __init__(self, raw_traj):
        self.raw_traj = raw_traj
    
    def get_gametime(self) -> int:
        """Get the latest game time."""
        return self.raw_traj["entries"][-1]["time"]
    
    def get_game_state(self, gametime=None) -> GameState:
        """Get the game state at the specified time."""
        if gametime is None:
            entry = self.raw_traj["entries"][-1]
        else:
            entry = next((e for e in self.raw_traj["entries"] if e["time"] == gametime), None)
        return GameState(entry) if entry else None
    
    def to_string(self, begin=None, end=None) -> str:
        """Convert the trajectory to a string representation, including all game states from begin to end.
        
        Args:
            begin (int, optional): Start index for the entries. Defaults to 0.
            end (int, optional): End index for the entries. Defaults to the length of entries.
        """
        text = ""
        begin = begin if begin is not None else 0
        end = end if end is not None else self.get_gametime()
        end = min(end, self.get_gametime())
        for entry in self.raw_traj["entries"]:
            if entry["time"] < begin or entry["time"] > end:
                continue
            gs = GameState(entry)
            text += gs.to_string() + "\n"
        return text
    
    def to_json(self, filename):
        import json
        
        with open(filename, "w") as f:
            json.dump(self.raw_traj, f, indent=4)
    
    @staticmethod
    def load(filename) -> "Trajectory":  # noqa: F821
        """Load a trajectory from a JSON file."""
        import json
        
        with open(filename, "r") as f:
            return Trajectory(json.load(f))
    
    def __iter__(self) -> Iterator[GameState]:
        for entry in self.raw_traj["entries"]:
            yield GameState(entry)
