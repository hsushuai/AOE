from skill_rts.game.observation import PlayerStatus, Observation, UnitStatus
from skill_rts.game.utils import PathPlanner, manhattan_distance
from skill_rts.game.skill import Skill, Unit
import logging
import numpy as np

logger = logging.getLogger()


class Player(PlayerStatus):
    def __init__(self, player_id: int, obs: Observation):
        player_status = obs.players[player_id - 1]
        if player_status is not None:
            super().__init__(**vars(player_status))
        else:
            raise ValueError(f"Player with ID {player_id} does not exist in the observation.")

        self.obs = obs

        valid_map = np.ones((obs.env.height, obs.env.width), dtype=int)
        valid_map[obs.raw_obs[0][1] != 0] = 0
        valid_map[obs.raw_obs[0][1] == 0] = 1
        self.path_planner = PathPlanner(valid_map)

    def play(self, tasks: tuple) -> np.ndarray:
        assign_tasks(self, tasks)
        act_tensor = np.zeros((self.obs.env.height, self.obs.env.width, 7), dtype=int)
        skill = Skill(self.path_planner, self.obs)
        for loc, unit in self.units.items():
            if unit.task is not None:
                act_tensor[loc] = skill.task_map(unit.task)(Unit(unit))
        return act_tensor


def assign_tasks(player: Player, tasks: tuple):
    def auto_attack():
        for loc, unit in player.units.items():
            if unit.action == "noop":
                target = choose_target(unit)
                if target:
                    unit.task = "[Attack Enemy]"
                    unit.task_params = (unit.type, target.type)

    def choose_target(unit: UnitStatus) -> UnitStatus:
        """Automatically choose a target to attack."""
        targets = []
        for enemy in player.obs.players[player.id % 2].units.values():
            if manhattan_distance(unit.location, enemy.location) <= unit.attack_range:
                targets.append((enemy, unit.attack_damage >= enemy.hp))

        targets.sort(key=lambda x: (not x[1], target_priority_index(x[0].type)))
        return targets[0][0] if targets else None

    def target_priority_index(unit_type: str) -> int:
        """Get priority index for targeting."""
        priority = ["worker", "light", "ranged", "heavy", "barrack", "base"]
        return priority.index(unit_type) if unit_type in priority else len(priority)

    def deploy_unit(task_params):
        unit_type, tgt_loc = task_params
        if player.obs.units.get(tgt_loc) is None:
            candidate_locs = [loc for loc, unit in player.units.items() if unit.type == unit_type and unit.task is None]
            if candidate_locs:
                nearest_loc = player.path_planner.get_manhattan_nearest(tgt_loc, candidate_locs)
                player.units[nearest_loc].task = "[Deploy Unit]"
                player.units[nearest_loc].task_params = task_params
                return
        logger.info(f"Pending task: [Deploy Unit] {task_params}")

    def harvest_mineral(task_params):
        mineral_loc = task_params
        candidate_locs = [loc for loc, unit in player.units.items() if unit.resource > 0 and unit.task is None]
        if len(candidate_locs) == 0 and player.obs.env.resources.get(mineral_loc) is not None:
            candidate_locs = [unit.location for unit in player.workers if unit.task is None]
        
        if candidate_locs:
            nearest_loc = player.path_planner.get_manhattan_nearest(mineral_loc, candidate_locs)
            player.units[nearest_loc].task = "[Harvest Mineral]"
            player.units[nearest_loc].task_params = task_params
            return
        logger.info(f"Pending task: [Harvest Mineral] {task_params}")

    def build_building(task_params):
        building_cost = {"base": 10, "barrack": 5}
        building_type, building_loc = task_params
        if player.obs.units.get(building_loc) is None and building_cost[building_type] <= player.resource:
            candidate_locs = [unit.location for unit in player.workers if unit.task is None]
            if candidate_locs:
                nearest_loc = player.path_planner.get_manhattan_nearest(building_loc, candidate_locs)
                player.units[nearest_loc].task = "[Build Building]"
                player.units[nearest_loc].task_params = task_params
                return
        logger.info(f"Pending task: [Build Building] {task_params}")

    def produce_unit(task_params):
        unit_cost = {"worker": 1, "light": 2, "heavy": 2, "ranged": 2}
        prod_type, direction = task_params
        if unit_cost[prod_type] <= player.resource:
            units = player.bases if prod_type == "worker" else player.barracks
            valid_units = []
            for unit in units:
                neighbors = player.path_planner.get_neighbors(unit.location)
                valid_directions = [d for d, _ in neighbors]
                if unit.task is None and direction in valid_directions:
                    valid_units.append(unit)
            if valid_units:
                player.units[valid_units[0].location].task = "[Produce Unit]"
                player.units[valid_units[0].location].task_params = task_params
                return
        logger.info(f"Pending task: [Produce Unit] {task_params}")

    def attack_enemy(task_params):
        unit_type, enemy_type = task_params
        unit_locs = [unit.location for unit in player.units.values() if unit.type == unit_type and unit.task is None]
        enemy_locs = [unit.location for unit in player.obs.players[player.id % 2].units.values() if unit.type == enemy_type]
        
        if unit_locs and enemy_locs:
            nearest_loc = min(
                unit_locs,
                key=lambda unit_loc: min(player.path_planner.get_shortest_path(unit_loc, enemy_loc)[0] for enemy_loc in enemy_locs)
            )
            player.units[nearest_loc].task = "[Attack Enemy]"
            player.units[nearest_loc].task_params = task_params
            return
        logger.info(f"Pending task: [Attack Enemy] {task_params}")
    
    task_map = {
        "[Deploy Unit]": deploy_unit,
        "[Harvest Mineral]": harvest_mineral,
        "[Build Building]": build_building,
        "[Produce Unit]": produce_unit,
        "[Attack Enemy]": attack_enemy,
    }
    for task in tasks:
        if task[0] in task_map:
            task_map[task[0]](task[1])

    auto_attack()
