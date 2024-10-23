from typing import List, Tuple, Optional
from collections import deque
import numpy as np


class PathPlanner:
    def __init__(self, valid_map: np.ndarray) -> None:
        self.height = valid_map.shape[0]
        self.width = valid_map.shape[1]
        self.valid_map = valid_map
        self.max_dist = 10**10

    def get_neighbors(self, pos: Tuple[int, int], valid=True) -> List[Tuple[str, Tuple[int, int]]]:
        directions = [
            ("north", (-1, 0)),
            ("east", (0, 1)),
            ("south", (1, 0)),
            ("west", (0, -1)),
        ]
        neighbors = []
        for dir_name, (dx, dy) in directions:
            new_x, new_y = pos[0] + dx, pos[1] + dy
            if 0 <= new_x < self.height and 0 <= new_y < self.width:
                if not valid:
                    neighbors.append((dir_name, (new_x, new_y)))
                elif self.valid_map[new_x][new_y] == 1:
                    neighbors.append((dir_name, (new_x, new_y)))
        return neighbors

    def get_shortest_path(self, location: Tuple[int, int], tg_location: Tuple[int, int]) -> Tuple[Optional[int], Optional[int]]:
        """
        Returns:
            int: path length
            str: direction
        """
        if location == tg_location:
            return 0, None

        visited = set()
        queue = deque()
        queue.append((location, 0, None))  # position, path_length, first_direction
        visited.add(location)

        while queue:
            pos, dist, first_dir = queue.popleft()
            for dir_name, neighbor in self.get_neighbors(pos):
                if neighbor == tg_location:
                    if first_dir is None:
                        first_dir = dir_name
                    return dist + 1, first_dir
                if neighbor not in visited:
                    visited.add(neighbor)
                    if first_dir is None:
                        queue.append((neighbor, dist + 1, dir_name))
                    else:
                        queue.append((neighbor, dist + 1, first_dir))

        # Cannot find a path
        return None, None

    def get_path_nearest( self, location: Tuple, targets: List[Tuple]) -> Optional[Tuple]:
        """
        Returns:
            Tuple: the nearest target position
        """
        target_set = set(targets)
        visited = set()
        queue = deque()
        queue.append((location, 0))  # position, path_length
        visited.add(location)

        while queue:
            pos, dist = queue.popleft()
            if pos in target_set:
                return pos
            for _, neighbor in self.get_neighbors(pos):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        # Cannot find any of the targets
        return None

    def get_manhattan_nearest(self, location: tuple, targets: List[tuple]) -> tuple:
        min_i = 0
        min_dist = self.max_dist
        for i in range(len(targets)):
            cur_dist = manhattan_distance(location, targets[i])
            if cur_dist < min_dist:
                min_i = i
                min_dist = cur_dist
        return targets[min_i]


def manhattan_distance(l1, l2) -> int:
    """Get the manhattan distance between two locations."""
    return sum([abs(l1[i] - l2[i]) for i in range(len(l1))])


def get_neighbor(loc, direction):
    """Get a neighbor location based on direction."""
    directions = {
        "north": (loc[0] - 1, loc[1]),
        "south": (loc[0] + 1, loc[1]),
        "west": (loc[0], loc[1] - 1),
        "east": (loc[0], loc[1] + 1),
    }
    return directions.get(direction)


def get_direction(location, tgt_loc) -> str:
    if tgt_loc[0] > location[0]:
        return "south"
    elif tgt_loc[0] < location[0]:
        return "north"
    elif tgt_loc[1] > location[1]:
        return "east"
    elif tgt_loc[1] < location[1]:
        return "west"
    elif tgt_loc == location:
        return "stay"
    else:
        raise ValueError(f"I only live in two dimensions. {location} -> {tgt_loc}")
