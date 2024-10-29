from skill_rts.game.unit import Unit
from skill_rts.game.game_state import PlayerState
from skill_rts.game.utils import PathPlanner
import skill_rts.game.skill as skills
import numpy as np
from skill_rts import logger


class Player(PlayerState):
    """Player class that represents a player in the environment."""
    
    def __init__(self, player_id: int, obs: "GameState"):  # noqa: F821
        """
        Initializes the Player instance with the given player ID and observation.

        Args:
            player_id (int): the ID of the player
            obs (GameState): the current observation from the environment
        """
        player_status = obs.players[player_id]
        super().__init__(**vars(player_status))
        self.obs = obs
        self.path_planner = PathPlanner(obs)
        self.units = {loc: Unit(unit_status) for loc, unit_status in self.units.items()}
        self.auto_attack = True  # auto attack enemy in range

    def step(self, tasks: list) -> np.ndarray:
        """
        Executes a series of tasks for the player and returns the action vectors.

        Args:
            tasks (list): a list of tasks to be executed, each containing a skill name and its parameters

        Returns:
            np.ndarray: an array representing the action vectors of the player's actions
        """
        act_vecs = np.zeros((self.obs.env.height, self.obs.env.width, 7), dtype=int)
        
        # Execute auto-attack first if enabled
        if self.auto_attack:
            for unit in self.units.values():
                auto_attack = skills.AutoAttack(self)
                if auto_attack.assign_to_unit(self.units):
                    act_vec = auto_attack.execute_step()
                    act_vecs[auto_attack.unit.location] = act_vec
        
        # Execute other tasks
        for task in tasks:
            skill_name, skill_params = task
            skill = self._get_skill(skill_name, skill_params)
            act_vec = skill.step()
            if act_vec is not None:
                act_vecs[skill.unit.location] = act_vec
        return act_vecs


    def _get_skill(self, skill_name, skill_params) -> skills.Skill | None:
        """Retrieves the skill class corresponding to the given skill name."""
        for skill_class in vars(skills).values():
            if isinstance(skill_class, type) and issubclass(skill_class, skills.Skill):
                if skill_class.name == skill_name:
                    return skill_class(self, skill_params)
        return None
    
    def update_obs(self, pre_obs, cur_obs):
        self.pre_obs = pre_obs
        self.obs = cur_obs
