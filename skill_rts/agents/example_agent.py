class Player0Agent:
    def __init__(self, **kwargs):
        pass
    
    def step(self, obs: str) -> str:
        tasks = """\
        START OF TASK
        [Harvest Mineral](0, 0)
        [Harvest Mineral](0, 0)
        [Produce Unit](worker, south)
        [Produce Unit](worker, east)
        [Produce Unit](worker, south)
        [Produce Unit](worker, east)
        [Build Building](barracks, (0, 3), resource >= 7)
        [Produce Unit](ranged, east)
        [Produce Unit](ranged, south)
        [Produce Unit](ranged, east)
        [Produce Unit](ranged, south)
        [Attack Enemy](worker, worker)
        [Attack Enemy](worker, worker)
        [Attack Enemy](worker, worker)
        [Attack Enemy](worker, worker)
        [Attack Enemy](worker, barracks)
        [Attack Enemy](worker, base)
        [Attack Enemy](ranged, worker)
        [Attack Enemy](ranged, worker)
        [Attack Enemy](ranged, worker)
        [Attack Enemy](ranged, worker)
        [Attack Enemy](ranged, barracks)
        [Attack Enemy](ranged, base)
        END OF TASK"""
        return tasks

class Player1Agent:
    def __init__(self, **kwargs):
        pass
    
    def step(self, obs: str) -> str:
        tasks = """\
        START OF TASK
        [Harvest Mineral](7, 7)
        [Harvest Mineral](7, 7)
        [Produce Unit](worker, north)
        [Produce Unit](worker, west)
        [Produce Unit](worker, north)
        [Produce Unit](worker, west)
        [Attack Enemy](worker, worker)
        [Attack Enemy](worker, worker)
        [Attack Enemy](worker, worker)
        [Attack Enemy](worker, worker)
        [Attack Enemy](worker, barracks)
        [Attack Enemy](worker, base)
        END OF TASK"""
        return tasks
