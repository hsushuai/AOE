from skill_rts.game.unit import Unit
from skill_rts.game.game_state import PlayerState
from skill_rts.game.utils import PathPlanner
import skill_rts.game.skill as skills
import numpy as np
from skill_rts import logger


class Player(PlayerState):
    """Player class that mapping task plan to action executed in the environment."""
    
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
        self.tasks = None
        self._is_task_updated = False

    def step(self) -> np.ndarray:
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
        for task in self.tasks:
            skill_name, skill_params = task
            skill = self._get_skill(skill_name, skill_params)
            act_vec = skill.step()
            if act_vec is not None:
                act_vecs[skill.unit.location] = act_vec
        return act_vecs
    
    def set_tasks(self, tasks: str):
        self.tasks = self._parse_tasks(tasks)
        self._is_task_updated = True
    
    def _update_tasks(self):
        for 

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

    def _parse_task(self, text: str) -> list:
        import ast
        import re

        task_list = []
        params_list = []
        text = text.split("START OF TASK")[1].split("END OF TASK")[0]
        text_list = text.split("\n")
        for task_with_params in text_list:
            task_beg = task_with_params.find("[")
            task_end = task_with_params.find("]")
            param_beg = task_with_params.find("(")
            param_end = task_with_params.rfind(")")
            if task_beg + 1 and task_end + 1:
                task = task_with_params[task_beg : task_end + 1]
            else:
                task = None
            params = re.sub(
                r"(?<!\')(\b[a-zA-Z_]+\b)(?!\')",
                r"'\1'",
                task_with_params[param_beg : param_end + 1],
            )
            params = re.sub(r"'(\d+)'", r"\1", params)
            if param_beg + 1 and param_end + 1:
                params = ast.literal_eval(params)
                if task is not None:
                    task_list.append(task)
                    params_list.append(params)
        logger.info("Parsed Tasks from LLM's Respond:")
        for task, params in zip(task_list, params_list):
            logger.info(f"{task}{params}")
        return list(zip(task_list, params_list))
