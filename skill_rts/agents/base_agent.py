class BlueAgent:
    def __init__(self):
        pass
    
    def step(self):
        tasks = """\
        START OF TASK
        [Harvest Mineral](0, 0)
        [Harvest Mineral](0, 0)
        [Produce Unit](worker, south)
        [Produce Unit](worker, east)
        [Produce Unit](worker, south)
        [Produce Unit](worker, east)
        [Attack Enemy](worker, worker)
        [Attack Enemy](worker, worker)
        [Attack Enemy](worker, worker)
        [Attack Enemy](worker, worker)
        [Attack Enemy](worker, barrack)
        [Attack Enemy](worker, base)
        END OF TASK"""
        return tasks

class RedAgent:
    def __init__(self):
        pass
    
    def step(self):
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
        [Attack Enemy](worker, barrack)
        [Attack Enemy](worker, base)
        END OF TASK"""
        return tasks
