import numpy as np
from typing import Callable, Type, List, Tuple
from skill_rts.game.utils import PathPlanner, manhattan_distance, get_neighbor, get_direction
from skill_rts.game.observation import Observation, UnitStatus
import logging

logger = logging.getLogger()

class Unit(UnitStatus):
    ...

class Skill:
    def task_map(self, task: str) -> Callable[[Type[Unit]], np.ndarray]:
        func_map = {
            "[Deploy Unit]": self.deploy_unit,
            "[Harvest Mineral]": self.harvest_mineral,
            "[Build Building]": self.build_building,
            "[Produce Unit]": self.produce_unit,
            "[Attack Enemy]": self.attack_enemy
        }
        return func_map[task]
    
    def __init__(self, path_planner: PathPlanner, obs: Observation):
        self.path_planner = path_planner
        self.obs = obs

    def deploy_unit(self, unit: Unit) -> np.ndarray:
        return self.move_to_loc(unit, unit.location)
        
    def build_building(self, unit: Unit) -> np.ndarray:
        building_type, building_loc = unit.task_params
        if manhattan_distance(unit.location, building_loc) == 1:
            return unit.produce(get_direction(unit.location, building_loc), building_type)
        tgt_locs = self.get_around_locs(building_loc)
        tgt_locs = [loc for loc in tgt_locs if self.obs.units[loc] is None or unit.location == loc]
        if len(tgt_locs) > 0:
            tgt_loc = self.path_planner.get_path_nearest(unit.location, tgt_locs)
            return self.move_to_loc(unit, tgt_loc)
        return unit.noop()
    
    def harvest_mineral(self, unit: Unit) -> np.ndarray:
        mineral_loc = unit.task_params
        if unit.resource == 0:
            if manhattan_distance(mineral_loc, unit.location) == 1:
                return unit.harvest(get_direction(unit.location, mineral_loc))
            tgt_locs = self.get_around_locs(mineral_loc)
            tgt_locs = [loc for loc in tgt_locs if self.obs.units[loc] is None or unit.location == loc]
            tgt_loc = self.path_planner.get_path_nearest(unit.location, tgt_locs)
            return self.move_to_loc(unit, tgt_loc)
        else:
            bases_locs = [base.location for base in self.obs.players[unit.owner - 1].bases]
            base_loc = self.path_planner.get_manhattan_nearest(unit.location, bases_locs)
            if manhattan_distance(base_loc, unit.location) == 1:
                return unit.deliver(get_direction(unit.location, base_loc))
            tgt_locs = self.get_around_locs(base_loc)
            tgt_locs = [loc for loc in tgt_locs if self.obs.units[loc] is None or unit.location == loc]
            tgt_loc = self.path_planner.get_path_nearest(unit.location, tgt_locs)
            return self.move_to_loc(unit, tgt_loc)

    def produce_unit(self, unit: Unit) -> np.ndarray:
        prod_type, direction = unit.task_params
        loc = get_neighbor(unit.location, direction)
        if self.obs.units[loc] is not None:
            locs = self.get_around_locs(unit.location)
            for loc in locs:
                if self.obs.units[loc] is not None:
                    break
        return unit.produce(get_direction(unit.location, loc), prod_type)

    def attack_enemy(self, unit: Unit) -> np.ndarray:
        enemy_type = unit.task_params[1]
        enemy_locs = [enemy.location for enemy in self.obs.players[unit.owner % 2].units.values() if enemy.type == enemy_type]
        enemy_loc = self.path_planner.get_manhattan_nearest(unit.location, enemy_locs)
        if manhattan_distance(unit.location, enemy_loc) == 1:
            return unit.attack(enemy_loc)
        tgt_locs = self.get_around_locs(enemy_loc)
        tgt_loc = self.path_planner.get_path_nearest(unit.location, tgt_locs)
        return self.move_to_loc(unit, tgt_loc)

    def move_to_loc(self, unit: Unit, tgt_loc: Tuple) -> np.ndarray:
        if unit.location == tgt_loc:
            return unit.noop()
        _, direction = self.path_planner.get_shortest_path(tuple(unit.location), tgt_loc)
        return unit.move(direction)
    
    def get_around_locs(self, loc: Tuple) -> List[Tuple]:
        locs = [
            (loc[0] + 1, loc[1]),
            (loc[0] - 1, loc[1]),
            (loc[0], loc[1] + 1),
            (loc[0], loc[1] - 1),
        ]

        for loc in locs[:]:
            if loc not in self.obs.units:
                locs.remove(loc)
        return locs


ACTION2INDEX = {"noop": 0, "move": 1, "harvest": 2, "return": 3, "produce": 4, "attack": 5}
UNIT_TYPE2INDEX = {"resource": 0, "base": 1, "barrack": 2, "worker": 3, "light": 4, "heavy": 5, "ranged": 6}
DIRECTION2INDEX = {"north": 0, "east": 1, "south": 2, "west": 3}

class Unit(UnitStatus):
    def __init__(self, unit_status: Unit):
        super().__init__(**vars(unit_status))

    def noop(self) -> np.ndarray:
        """
        Atom action: Noop (do nothing)

        Return: action vector, shape of [7]
        """
        
        return np.zero((7), dtype=int)

    def move(self, direction: str) -> np.ndarray:
        """
        Atom action: Move

        Args:
            direction: moving direction

        Return: action vector, shape of [7]
        """
        act_vec = np.zeros((7), dtype=int)
        act_vec[0] = ACTION2INDEX["move"]
        act_vec[1] = DIRECTION2INDEX[direction]
        self.log_action_info("move")
        return act_vec

    def harvest(self, direction: str) -> np.ndarray:
        """
        Atom action: Harvest

        Args:
            direction: harvesting direction

        Return: action vector, shape of [7]
        """
        act_vec = np.zeros((7), dtype=int)
        act_vec[0] = ACTION2INDEX["harvest"]
        act_vec[2] = DIRECTION2INDEX[direction]
        self.log_action_info("harvest")
        return act_vec

    def deliver(self, direction: str) -> np.ndarray:
        """
        Atom action: Return

        Args:
            direction: delivering direction

        Return: action vector, shape of [7]
        """
        act_vec = np.zeros((7), dtype=int)
        act_vec[0] = ACTION2INDEX["return"]
        act_vec[3] = DIRECTION2INDEX[direction]
        self.log_action_info("return")
        return act_vec

    def produce(self, direction: str, prod_type: str) -> np.ndarray:
        """
        Atom action: Produce

        Args:
            direction: production direction
            prod_type: type of unit to produce, 'resource', 'base', 'barrack', 'worker', 'light', 'heavy', 'ranged'

        Return: action vector, shape of [7]
        """
        act_vec = np.zeros((7), dtype=int)
        act_vec[0] = ACTION2INDEX["produce"]
        act_vec[4] = DIRECTION2INDEX[direction]
        act_vec[5] = UNIT_TYPE2INDEX[prod_type]
        self.log_action_info("produce")
        return act_vec

    def attack(self, tgt_loc: tuple) -> np.ndarray:
        """
        Atom action: Attack

        Args:
            tgt_loc: target location for attack

        Return: action vector, shape of [7]
        """
        tgt_relative_loc = (tgt_loc[0] - self.location[0] + 3) * 7 + (tgt_loc[1] - self.location[1] + 3)
        act_vec = np.zeros((7), dtype=int)
        act_vec[0] = ACTION2INDEX["attack"]
        act_vec[6] = tgt_relative_loc
        self.log_action_info("attack")
        return act_vec
    
    def log_action_info(self, action):
        if self.action != "noop":
            action = self.action
        logger.info(f"{self.type}{self.location}: {self.task}{self.task_params}/{action}")