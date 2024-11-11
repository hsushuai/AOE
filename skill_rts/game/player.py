from skill_rts.game.unit import Unit
from skill_rts.game.game_state import PlayerState, GameState
from skill_rts.game.utils import PathPlanner
from skill_rts.game.metric import Metric
import skill_rts.game.skill as skills
import numpy as np
from skill_rts import logger


class Player(PlayerState):
    """Player class that mapping task plan to action executed in the environment."""
    
    def __init__(self, player_id: int, obs: GameState):
        """
        Initializes the Player instance with the given player ID and observation.

        Args:
            player_id (int): the ID of the player
            obs (GameState): the observation of the game state
        """
        self.id = player_id
        self.auto_attack = True  # auto attack enemy in range
        self.tasks = None
        self.obs = None
        self.update_obs(obs)

    def step(self) -> np.ndarray:
        """
        Executes a series of tasks for the player and returns the action vectors.

        Returns:
            np.ndarray: an array representing the action vectors of the player's actions
        """
        self.tasks.update()
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
            skill = self.get_skill(skill_name)(self, skill_params)
            act_vec = skill.step()
            if act_vec is not None:
                act_vecs[skill.unit.location] = act_vec
        return act_vecs
    
    def set_tasks(self, tasks: str):        
        self.tasks = TaskManager(tasks, self)

    @staticmethod
    def get_skill(skill_name) -> skills.Skill | None:
        """Retrieves the skill class corresponding to the given skill name."""
        for skill_class in vars(skills).values():
            if isinstance(skill_class, type) and issubclass(skill_class, skills.Skill):
                if skill_class.name == skill_name:
                    return skill_class
        return None
    
    def update_obs(self, obs: GameState):
        self.obs = obs
        player_state = obs.players[self.id]
        super().__init__(**vars(player_state))
        self.path_planner = PathPlanner(self.obs)
        self.units = {loc: Unit(unit_status) for loc, unit_status in self.units.items()}


class TaskManager:
    def __init__(self, tasks: str, player: Player):
        self.task_list, self.params_list = self.parse_tasks(tasks)
        self.completed_tasks = []
        self.player = player
        self.player_id = player.id
        self.obs = player.obs
        # used for update tasks, a better way is to use action trace which is not implemented yet
        self._metric = Metric(player.obs)
        self.temp_pending = []  # when building a worker harvest, pending remain harvest task
    
    def update(self):
        self._metric.update(self.obs)
        resource = self.player.resource  # noqa: F841
        kills = {unit_type: len(units) for unit_type, units in self._metric.unit_killed[self.player_id].items()}
        prods = {unit_type: len(units) for unit_type, units in self._metric.unit_produced[self.player_id].items()}
        if "[Build Building]" in self.task_list:
            index = self.task_list.index("[Build Building]")
            condition = self.params_list[index][2]
            if eval(condition) and self.task_list.count("[Harvest Mineral]") > 1 and len(self.player.barracks) == 0:
                harvest_idxs = [i for i, task in enumerate(self.task_list) if task == "[Harvest Mineral]"]
                self.temp_pending = [(self.task_list.pop(i), self.params_list.pop(i)) for i in harvest_idxs[1:]]
            elif len(self.temp_pending) > 0:
                for task, params in self.temp_pending:
                    self.task_list.insert(0, task)
                    self.params_list.insert(0, params)
            if len(self.player.barracks) > 0:
                for i, (task, params) in enumerate(self.temp_pending):
                    self.task_list.insert(i, task)
                    self.params_list.insert(i, params)
        for task, params in zip(self.task_list, self.params_list):
            skill = Player.get_skill(task)
            if skill is not None and skill.is_completed(kills=kills, prods=prods, obs=self.obs, params=params):
                self.completed_tasks.append((task, params))
                self.task_list.remove(task)
                self.params_list.remove(params)
    
    @property
    def tasks(self):
        return list(zip(self.task_list, self.params_list))
    
    @staticmethod
    def parse_tasks(text: str) -> list:
        import ast
        import re

        task_list = []
        params_list = []
        
        task_section = text.split("START OF TASK")[1].split("END OF TASK")[0]
        text_lines = task_section.split("\n")
        
        for line in text_lines:
            task_beg = line.find("[")
            task_end = line.find("]")
            param_beg = line.find("(")
            param_end = line.rfind(")")

            task = line[task_beg:task_end + 1] if task_beg != -1 and task_end != -1 else None
            params_str = line[param_beg:param_end + 1] if param_beg != -1 and param_end != -1 else ""

            params_str = re.sub(r"(?<!')(\b[a-zA-Z_]+\b(\s*>=\s*\d+)?)(?<!')", r"'\1'", params_str)
            params_str = re.sub(r"'(\d+)'", r"\1", params_str)

            if params_str:
                try:
                    params = ast.literal_eval(params_str)
                except (ValueError, SyntaxError):
                    continue

                if task and (skill := Player.get_skill(task)) is not None:
                    if skill.params_validate(params):
                        if task == "[Build Building]":
                            task_list.insert(0, task)
                            params_list.insert(0, params)
                        else:
                            task_list.append(task)
                            params_list.append(params)

        logger.info("Parsed Tasks from LLM's Respond:")
        for task, params in zip(task_list, params_list):
            logger.info(f"{task}{params}")
            
        return task_list, params_list
    
    def __iter__(self):
        return iter(zip(self.task_list, self.params_list))