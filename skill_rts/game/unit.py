import numpy as np
from skill_rts.game.observation import UnitStatus
from skill_rts import logger

ACTION2INDEX = {"noop": 0, "move": 1, "harvest": 2, "return": 3, "produce": 4, "attack": 5}
UNIT_TYPE2INDEX = {"resource": 0, "base": 1, "barrack": 2, "worker": 3, "light": 4, "heavy": 5, "ranged": 6}
DIRECTION2INDEX = {"north": 0, "east": 1, "south": 2, "west": 3}


class Unit(UnitStatus):
    def __init__(self, unit_status: UnitStatus):
        super().__init__(**vars(unit_status))

    def noop(self) -> np.ndarray:
        """
        Atom action: Noop (do nothing)

        Return: action vector, shape of [7]
        """
        self.log_action_info("noop", "")
        return np.zeros((7), dtype=int)

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
        self.log_action_info("move", f"({direction})")
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
        self.log_action_info("harvest", f"({direction})")
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
        self.log_action_info("return", f"({direction})")
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
        self.log_action_info("produce", (direction, prod_type))
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
        self.log_action_info("attack", tgt_loc)
        return act_vec
    
    def log_action_info(self, action, action_params):
        logger.info(f"{self.type}{self.location}: {self.task}{self.task_params}/{action}{action_params}")
