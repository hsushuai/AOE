class Metric:

    def __init__(self, game_state):
        self._gs = game_state
        self._unit_types = ["base", "barracks", "worker", "heavy", "light", "ranged"]
        self._init_resources = [player.resource for player in game_state.players]
        self.game_time = game_state.time

        self.unit_produced = [{unit_type: [] for unit_type in self._unit_types} for _ in range(2)]
        self.unit_lost = [{unit_type: [] for unit_type in self._unit_types} for _ in range(2)]
        self.damage_taken = [0, 0]
        
        self.win_loss = [None, None]
        self.winner = None

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
    
    def set_winner(self, winner):
        self.winner = winner
        if winner == 0:
            self.win_loss = [1, -1]
        elif winner == 1:
            self.win_loss = [-1, 1]
        else:
            self.win_loss = [0, 0]

    def to_json(self, file_path):
        import json

        metrics = {
            "unit_produced": [],
            "unit_lost": [],
            "unit_killed": [],
            "damage_taken": [],
            "damage_dealt": [],
            "resource_spent": [],
            "resource_harvested": [],
            "win_loss": self.win_loss,
            "game_time": self.game_time
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

        with open(file_path, "w") as f:
            json.dump(metrics, f, indent=4)
    
    def to_string(self) -> str:
        texts = ["Player 0: \n", "Player 1: \n"]

        # Units Produced
        for i, player in enumerate(self.unit_produced):
            texts[i] += "  Units Produced: \n    "
            for unit_type, units in player.items():
                texts[i] += f"{unit_type}: {len(units)}; "
            texts[i] += "\n"
        
        # Units Lost
        for i, player in enumerate(self.unit_lost):
            texts[i] += "  Units Lost: \n    "
            for unit_type, units in player.items():
                texts[i] += f"{unit_type}: {len(units)}; "
            texts[i] += "\n"

        # Units Killed
        for i, player in enumerate(self.unit_killed):
            texts[i] += "  Units Killed: \n    "
            for unit_type, units in player.items():
                texts[i] += f"{unit_type}: {len(units)}; "
            texts[i] += "\n"
        
        # Damage Taken
        for i, player in enumerate(self.damage_taken):
            texts[i] += f"  Damage Taken: {self.damage_taken[i]}\n"
        
        # Damage Dealt
        for i, player in enumerate(self.damage_dealt):
            texts[i] += f"  Damage Dealt: {self.damage_dealt[i]}\n"
        
        # Resource Harvested
        for i, player in enumerate(self.resource_harvested):
            texts[i] += f"  Resource Harvested: {self.resource_harvested[i]}\n"
        
        # Resource Spent
        for i, player in enumerate(self.resource_spent):
            texts[i] += f"  Resource Spent: {self.resource_spent[i]}\n"
        
        # winner
        for i, player in enumerate(self.win_loss):
            texts[i] += f"  Winner: player {self.winner}\n"
        
        return "".join(texts)

