from skill_rts.game.trajectory import Trajectory


class TrajectoryFeature:
    def __init__(self, trajectory: Trajectory):
        self.trajectory = trajectory

    def get_features(self, begin=None, end=None):
        begin = begin if begin is not None else 0
        end = end if end is not None else self.trajectory.get_gametime()
        end = min(end, self.trajectory.get_gametime())
        economic_features = self.get_economic_features(begin, end)
        barracks_features = self.get_barracks_features(begin, end)
        military_features = self.get_military_features(begin, end)
        attack_features = self.get_attack_features(begin, end)
        position_features = self.get_position_features(begin, end)
        features = [
            {
                "economic(num of workers to harvest)": economic_features[i],
                "barracks(build_time, player resources, location)": barracks_features[i],
                "military(num of military units)": military_features[i],
                "attack(attack location, type of victim)": attack_features[i],
                "position(location, occurrence count)": position_features[i],
            } for i in range(2)
        ]
        return features

    def get_economic_features(self, begin: int, end: int):
        economic_features = [set() for _ in range(2)]
        for gs in self.trajectory:
            if gs.time < begin or gs.time > end:
                continue
            for unit in gs:
                if unit is not None and unit.action == "harvest":
                    economic_features[unit.owner].add(unit.id)
        return [len(ids) for ids in economic_features]
    
    def get_barracks_features(self, begin: int, end: int):
        barracks_features = [{"time": None, "resources": None, "location": None} for _ in range(2)]
        dir2loc = {
            "north": (0, -1),
            "south": (0, 1),
            "east": (1, 0),
            "west": (-1, 0)
        }
        for gs in self.trajectory:
            if gs.time < begin or gs.time > end:
                continue
            for unit in gs:
                if unit is not None and unit.action == "produce" and unit.action_params[1] == "barracks":
                    barracks_features[unit.owner]["time"] = gs.time
                    barracks_features[unit.owner]["resources"] = gs.players[unit.owner].resource
                    diff = dir2loc[unit.action_params[0]]
                    barracks_features[unit.owner]["location"] = (unit.location[0] + diff[0], unit.location[1] + diff[1])
        return barracks_features
    
    def get_military_features(self, begin: int, end: int):
        military = ["worker", "heavy", "light", "ranged"]
        military_features = [{unit_type: set() for unit_type in military}for _ in range(2)]
        for gs in self.trajectory:
            if gs.time < begin or gs.time > end:
                continue
            for unit in gs:
                if unit is not None and unit.type in military:
                    military_features[unit.owner][unit.type].add(unit.id)
        return [{unit_type: len(ids) for unit_type, ids in unit_type_ids.items()} for unit_type_ids in military_features]

    def get_attack_features(self, begin: int, end: int):
        attack_features = [[], []]
        for gs in self.trajectory:
            if gs.time < begin or gs.time > end:
                continue
            for unit in gs:
                if unit is not None and unit.action == "attack":
                    attack_features[unit.owner].append({unit.action_params: gs[unit.action_params].type})
        return attack_features
    
    def get_position_features(self, begin: int, end: int):
        position_features = [{}, {}]
        military = ["worker", "heavy", "light", "ranged"]
        for gs in self.trajectory:
            if gs.time < begin or gs.time > end:
                continue
            for unit in gs:
                if unit is not None and unit.type in military:
                    if unit.location not in position_features[unit.owner]:
                        position_features[unit.owner][unit.location] = 1
                    else:
                        position_features[unit.owner][unit.location] += 1
        return position_features
    
    def to_string(self, begin=None, end=None):
        features = self.get_features(begin, end)
        text = ""
        for i, feat in enumerate(features):
            text += f"**Player {i}**:\n"
            for k, v in feat.items():
                text += f"{k}: {v}\n"
        return text


if __name__ == "__main__":
    # Example usage
    trajectory = Trajectory.load("runs/eval_ace/coacAI/run_0/traj.json")
    features = TrajectoryFeature(trajectory)
    print(features.to_string())
