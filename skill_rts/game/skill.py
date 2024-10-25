import numpy as np
from abc import ABC, abstractmethod
from skill_rts.game.utils import PathPlanner, manhattan_distance, get_neighbor, get_direction
from skill_rts.game.unit import Unit
from skill_rts.game.observation import Observation, UnitStatus
from skill_rts import logger


class Skill(ABC):
    """
    Abstract base class for defining skills that units can perform in the game.

    Custom skills can be created by inheriting from this class. To implement a new skill,
    you need to specify the skill name and provide implementations for the following methods:
    
    1. `assign_to_unit`: Assigns the skill to a specific unit based on the provided criteria.
    2.`execute_step`: Defines the actions to be performed when the skill is executed.
    """
    # skill name will be used to instantiate the skill and must be unique
    name: str = ""
    # the unit to execute the skill
    unit: Unit
    # the skill parameters
    params: tuple

    def __init__(self, player: "Player", skill_params: tuple):  # noqa: F821
        """
        Args:
            player (Player): The player that owns the unit executing the skill.
            skill_params (tuple): A tuple containing parameters specific to the skill.
        """
        self.player = player
        self.params = skill_params
        self.is_auto_attack = player.auto_attack
        self.obs: Observation = player.obs
        self.path_planner: PathPlanner = player.path_planner
    
    def step(self) -> np.ndarray:
        """
        Perform a step in the skill's execution.

        Returns:
            np.ndarray: a numpy array representing the action of the step
        """
        assigned = self.assign_to_unit()
        if assigned:
            return self.execute_step()
        logger.info(f"Pending task: {self.name}{self.params}")
        return None  # do nothing

    @abstractmethod
    def execute_step(self) -> np.ndarray:
        """
        Perform a step in the skill's execution.
        
        Returns:
            np.ndarray: a numpy array representing the action of the step
        """
        pass
    
    @abstractmethod
    def assign_to_unit(self) -> bool:
        """
        Assign the skill to a unit in the list of units.
        
        Returns:
            bool: True if the skill is assigned to a unit.
        """
        pass

    def move_to_loc(self, tgt_loc: tuple) -> np.ndarray:
        """
        Move the unit to the specified target location.

        Args:
            tgt_loc (tuple): The target location to move to, specified as (x, y) coordinates.

        Returns:
            np.ndarray: A numpy array representing the action to move the unit.
        
        If the unit is already at the target location, it performs a no-operation.
        Otherwise, it calculates the shortest path to the target location and returns the 
        action to move in the determined direction.
        """
        if self.unit.location == tgt_loc:
            return self.unit.noop()  # stay at target location
        _, direction = self.path_planner.get_shortest_path(self.unit.location, tgt_loc)
        if direction is None:  # no way
            return self.unit.noop()
        return self.unit.move(direction)


class DeployUnit(Skill):
    """
    A skill for deploying a unit to a specified location.

    This class allows the deployment of a specific type of unit based on its 
    proximity to the target location. If the target location is available, 
    the nearest eligible unit of the specified type will be assigned to 
    carry out the deployment task.

    Parameters:
        skill_params (tuple): A tuple containing:
            - unit_type (str): The type of unit to deploy.
            - tgt_loc (tuple): The target location for the deployment, specified as (x, y) coordinates.
    """
    name = "[Deploy Unit]"

    def execute_step(self):
        return self.move_to_loc(self.params)
    
    def assign_to_unit(self):
        unit_type, tgt_loc = self.params
        if self.obs.units.get(tgt_loc) is None:
            candidates = [unit.location for unit in self.player if unit.type == unit_type and unit.task is None]
            if candidates:
                nearest_loc = self.path_planner.get_manhattan_nearest(tgt_loc, candidates)
                self.player[nearest_loc].task = self.name
                self.player[nearest_loc].task_params = self.params
                self.unit = self.player[nearest_loc]
                return True
        return False


class BuildBuilding(Skill):
    """
    A skill for a worker to a build a building on the specified location.

    This class allows a worker unit to construct a specific type of building 
    at a designated location on the map. The nearest worker will be assigned 
    to move to build it if the target location is available for construction.

    Parameters:
        skill_params (tuple): A tuple containing:
            - building_type (str): The type of building to build.
            - tgt_loc (tuple): The target location for the building, specified as (x, y) coordinates.
    """
    name = "[Build Building]"
    
    def execute_step(self):
        building_type, building_loc = self.params
        if manhattan_distance(self.unit.location, building_loc) == 1:
            return self.unit.produce(get_direction(self.unit.location, building_loc), building_type)
        neighbors = self.path_planner.get_neighbors(self.unit.location)
        tgt_locs = [loc for direction, loc in neighbors]
        if len(tgt_locs) > 0:
            tgt_loc = self.path_planner.get_path_nearest(self.unit.location, tgt_locs)
            return self.move_to_loc(tgt_loc)
        return self.unit.noop()
    
    def assign_to_unit(self):
        building_cost = {"base": 10, "barrack": 5}
        building_type, building_loc = self.params
        if self.obs.units.get(building_loc) is None and building_cost[building_type] <= self.player.resource:
            candidates = [unit.location for unit in self.player.workers if unit.task is None]
            if candidates:
                nearest_loc = self.path_planner.get_manhattan_nearest(building_loc, candidates)
                self.player[nearest_loc].task = "[Build Building]"
                self.player[nearest_loc].task_params = self.params
                self.unit = self.player[nearest_loc]
                return True
        return False


class HarvestMineral(Skill):
    """
    A skill for a worker to harvest minerals from a specified location.

    This class allows a worker unit to move to harvest minerals if the location 
    is valid and contains mineral resources. The nearest available worker will 
    be assigned to perform the harvesting task.

    Parameters:
        skill_params (tuple): A tuple containing:
            - mine_loc (tuple): The target location of the mineral, specified as (x, y) coordinates.
    """
    name = "[Harvest Mineral]"

    def execute_step(self):
        mine_loc = self.params
        if self.unit.resource == 0:
            if manhattan_distance(mine_loc, self.unit.location) == 1:
                return self.unit.harvest(get_direction(self.unit.location, mine_loc))
            neighbors = self.path_planner.get_neighbors(mine_loc)
            tgt_locs = [loc for direction, loc in neighbors]
            tgt_loc = self.path_planner.get_path_nearest(self.unit.location, tgt_locs)
            return self.move_to_loc(tgt_loc)
        else:
            bases_locs = [base.location for base in self.player.bases]
            if len(bases_locs) == 0:
                return self.unit.noop()
            base_loc = self.path_planner.get_manhattan_nearest(self.unit.location, bases_locs)
            if manhattan_distance(base_loc, self.unit.location) == 1:
                return self.unit.deliver(get_direction(self.unit.location, base_loc))
            return self.move_to_loc(base_loc)
    
    def assign_to_unit(self):
        mine_loc = self.params
        if len(self.player.bases) == 0:
            return False
        candidates = [unit.location for unit in self.player.workers if unit.resource > 0 and unit.task is None]
        if not candidates and self.obs.env.resources.get(mine_loc) is not None:
            candidates = [unit.location for unit in self.player.workers if unit.task is None]
        
        if candidates:
            nearest_loc = self.path_planner.get_manhattan_nearest(mine_loc, candidates)
            self.player[nearest_loc].task = "[Harvest Mineral]"
            self.player[nearest_loc].task_params = self.params
            self.unit = self.player[nearest_loc]
            return True
        return False


class ProduceUnit(Skill):
    """
    A skill for a base or barracks to produce a specified unit.

    This class enables a base or barracks to initiate the production of a unit. 
    The produced unit will output from specified direction upon completion.

    Parameters:
        skill_params (tuple): A tuple containing:
            - prod_type (str): The type of unit to be produced.
            - direction (str): The output direction from the production facility (e.g., "north", "south").
    """
    name = "[Produce Unit]"

    def execute_step(self):
        prod_type, direction = self.params
        loc = get_neighbor(self.unit.location, direction)
        if self.obs.units[loc] is not None:
            neighbors = self.path_planner.get_neighbors(self.unit.location)
            for _, loc in neighbors:
                if self.obs.units[loc] is not None:
                    break
        return self.unit.produce(get_direction(self.unit.location, loc), prod_type)
    
    def assign_to_unit(self):
        unit_cost = {"worker": 1, "light": 2, "heavy": 2, "ranged": 2}
        prod_type, direction = self.params
        if unit_cost[prod_type] <= self.player.resource:
            units = self.player.bases if prod_type == "worker" else self.player.barracks
            valid_unit = None
            for unit in units:
                neighbors = self.path_planner.get_neighbors(unit.location)
                valid_directions = [d for d, _ in neighbors]
                if unit.task is None and direction in valid_directions:
                    valid_unit = unit
                    break
            if valid_unit:
                valid_unit.task = "[Produce Unit]"
                valid_unit.task_params = self.params
                self.unit = valid_unit
                return True
        return False


class AttackEnemy(Skill):
    """
    A skill for a specified type of unit to attack a designated enemy type.

    The unit of the specified type that is closest to the target enemy will 
    engage in combat with the enemy of the specified type.

    Parameters:
        skill_params (tuple): A tuple containing:
            - unit_type (str): The type of unit to be deployed for the attack.
            - enemy_type (str): The type of enemy to be targeted by the unit.
    """
    name = "[Attack Enemy]"

    def __init__(self, player: "Player", skill_params: tuple):  # noqa: F821
        super().__init__(player, skill_params)
        self.player_id = player.id
        self.enemy_id = 1 - player.id
        self.enemy_loc: tuple = None
        
    def execute_step(self):
        if manhattan_distance(self.unit.location, self.enemy_loc) <= self.unit.attack_range:
            return self.unit.attack(self.enemy_loc)
        
        tgt_locs = self.path_planner.get_range_locs(self.enemy_loc, valid=True, dist=self.unit.attack_range)
        if len(tgt_locs) == 0:  # no location for attack
            return self.unit.noop()
        tgt_loc = self.path_planner.get_manhattan_nearest(self.unit.location, tgt_locs)
        return self.move_to_loc(tgt_loc)
    
    def assign_to_unit(self):
        unit_type, enemy_type = self.params
        unit_locs = [unit.location for unit in self.player if unit.type == unit_type and unit.task is None]
        enemy_locs = [unit.location for unit in self.obs.players[self.enemy_id] if unit.type == enemy_type]

        if unit_locs and enemy_locs:
            nearest_loc, enemy_loc = min(
                ((unit_loc, self.path_planner.get_manhattan_nearest(unit_loc, enemy_locs)) for unit_loc in unit_locs),
                key=lambda loc_pair: manhattan_distance(loc_pair[0], loc_pair[1])
            )
            self.player[nearest_loc].task = "[Attack Enemy]"
            self.player[nearest_loc].task_params = self.params
            self.unit = self.player[nearest_loc]
            self.enemy_loc = enemy_loc
            return True
        return False


class AutoAttack(AttackEnemy):
    name = "[Auto Attack]"

    def __init__(self, player: "Player"):  # noqa: F821
        super().__init__(player, ())

    def assign_to_unit(self, units: dict[tuple, Unit]):
        """
        Auto attack enemy in range.
        
        Note: This skill will automatically assign to a unit if `player.auto_attack` is True. Can not be a task.
        
        The attack priorities are determined by the following unit types (in order):
        "worker", "ranged", "light", "heavy", "barrack", "base".
        Enemies with health less than or equal to the attack damage of the player's unit are prioritized first.

        """
        def choose_target(unit: Unit) -> UnitStatus:
            targets = []
            enemy_id = 1 - self.player.id
            for enemy in self.obs.players[enemy_id]:
                if manhattan_distance(unit.location, enemy.location) <= unit.attack_range:
                    targets.append((enemy, unit.attack_damage >= enemy.hp))

            targets.sort(key=lambda x: (not x[1], target_priority_index(x[0].type)))
            return targets[0][0] if targets else None

        def target_priority_index(unit_type: str) -> int:
            priority = ["worker", "ranged", "light", "heavy", "barrack", "base"]
            return priority.index(unit_type) if unit_type in priority else len(priority)

        for unit in units.values():
            if unit.action == "noop":
                target = choose_target(unit)
                if target:
                    unit.task = self.name
                    unit.task_params = (unit.type, target.type)
                    self.unit = unit
                    self.enemy_loc = target.location
                    return True
        return False