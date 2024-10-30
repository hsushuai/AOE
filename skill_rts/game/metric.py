class Metric:

    def __init__(self, game_state):
        self._gs = game_state
        self._unit_types = ["base", "barracks", "worker", "heavy", "light", "ranged"]
        self._init_resources = [player.resource for player in game_state.players]
        self.game_time = game_state.time

        self.unit_produced = [{unit_type: [] for unit_type in self._unit_types} for _ in range(2)]
        self.unit_lost = [{unit_type: [] for unit_type in self._unit_types} for _ in range(2)]
        self.damage_taken = [0, 0]

    def update(self, game_state):
        self._pre_gs = self._gs
        self._gs = game_state
        self.game_time = game_state.time

        self._update_unit_produced()
        self._update_unit_lost()
        self._update_damage_taken()
        

    @property
    def _pre_units(self):
        players = []
        for player in self._pre_gs.players:
            players.append(
                {
                    unit_type: getattr(player, unit_type)
                    for unit_type in self._unit_types
                }
            )
        return players
    
    @property
    def _cur_units(self):
        players = []
        for player in self._gs.players:
            players.append(
                {
                    unit_type: getattr(player, unit_type)
                    for unit_type in self._unit_types
                }
            )
        return players
    
    def _update_unit_produced(self):
        players = []
        for _cur_units, _pre_units in zip(self._cur_units, self._pre_units):
            players.append(
                {
                    unit_type: self._get_diff(_cur_units[unit_type], _pre_units[unit_type])
                    for unit_type in self._unit_types
                }
            )
        
        for i, player in enumerate(players):
            for unit_type in self._unit_types:
                self.unit_produced[i][unit_type].extend(player[unit_type])
    
    @staticmethod
    def _get_diff(units1: list, units2:list):
        """Return list of units that are in units1 but not in units2."""
        return [unit for unit in units1 if unit not in units2]
    
    def _lost_units(self):
        players = []
        for _cur_units, _pre_units in zip(self._cur_units, self._pre_units):
            players.append(
                {
                    unit_type: self._get_diff(_pre_units[unit_type], _cur_units[unit_type])
                    for unit_type in self._unit_types
                }
            )
        return players
    
    def _update_unit_lost(self):
        players = self._lost_units()
        for i, player in enumerate(players):
            for unit_type in self._unit_types:
                self.unit_lost[i][unit_type].extend(player[unit_type])
        return players
    
    @property
    def unit_killed(self):
        players = self.unit_lost[:]
        players[0], players[1] = players[1], players[0]
        return players

    def _update_damage_taken(self):
        players = []
        for cur_units, pre_units in zip(self._cur_units, self._pre_units):
            damage_taken = 0
            for unit_type in self._unit_types:
                for unit in cur_units[unit_type]:
                    pre_unit = None
                    for _pre_unit in pre_units[unit_type]:
                        if _pre_unit.id == unit.id:
                            pre_unit = _pre_unit
                            break
                    if pre_unit is not None:
                        damage_taken += pre_unit.hp - unit.hp
            players.append(damage_taken)
        
        for loss_units in self._lost_units():
            damage_taken = players.pop(0)
            for unit_type in self._unit_types:
                damage_taken += sum(
                    [unit.hp for unit in loss_units[unit_type]]
                )
            players.append(damage_taken)
        for i, player in enumerate(players):
            self.damage_taken[i] += player
    
    @property
    def damage_dealt(self):
        players = self.damage_taken[:]
        players[0], players[1] = players[1], players[0]
        return players
    
    @property
    def resource_spent(self):
        players = []
        for new_units in self.unit_produced:
            players.append(sum([
                unit.cost for unit_type in self._unit_types
                for unit in new_units[unit_type]
            ]))
        return players
    
    @property
    def resource_harvested(self):
        players = []
        for player, player_spent in zip(self._gs.players, self.resource_spent):
            players.append(player.resource - self._init_resources[player.id] + player_spent)
        return players

    def to_json(self, file_stream):
        import json

        metrics = {
            "unit_produced": [],
            "unit_lost": [],
            "unit_killed": [],
            "damage_taken": [],
            "damage_dealt": [],
            "resource_spent": [],
            "resource_harvested": []
        }

        # Units Produced
        for player in self.unit_produced:
            d = {unit_type: len(units) for unit_type, units in player.items()}
            metrics["unit_produced"].append(d)

        # Units Lost
        for player in self.unit_lost:
            d = {unit_type: len(units) for unit_type, units in player.items()}
            metrics["unit_lost"].append(d)

        # Units Killed
        for player in self.unit_killed:
            d = {unit_type: len(units) for unit_type, units in player.items()}
            metrics["unit_killed"].append(d)

        # Damage Taken
        metrics["damage_taken"] = self.damage_taken

        # Damage Dealt
        metrics["damage_dealt"] = self.damage_dealt

        # Resource Spent
        metrics["resource_spent"] = self.resource_spent

        # Resource Harvested
        metrics["resource_harvested"] = self.resource_harvested

        json.dump(metrics, file_stream, indent=4)
