from skill_rts.game.trajectory import Trajectory
from ace.traj_feat import TrajectoryFeature
from ace.strategy import Strategy
from ace.agent import Recognizer
from omegaconf import OmegaConf


class Reviewer:
    """
    Post-match review for the following:
    - Meta strategy to win in the opponent's strategy space
    - Tips for bridge the gap between strategy and plan
    """
    def __init__(self, model, temperature, max_tokens):
        self.recognizer = Recognizer(model, temperature, max_tokens)
        self.client = self.recognizer.client
        self.opponent_strategy_space = set()
        self.meta_strategy = None
        self.planner_tips = ""
    
    def step(self, strategy: str, obs: str, plan: str, traj: Trajectory):
        self.reflect_meta_strategy()
        self.reflect_planner(strategy, obs, plan)
    
    def recognize_strategy(self, traj: Trajectory) -> Strategy:
        traj_feat = TrajectoryFeature(traj)
        return self.recognizer.step(traj_feat.to_string())
    
    def reflect_meta_strategy(self, traj: Trajectory):
        strategy = self.recognize_strategy(traj)
        self.opponent_strategy_space.add(strategy)

        template = OmegaConf.load("ace/templates/post_match.yaml")["REFLECT_META_STRATEGY"]
        opponent_strategy_space = "\n".join([strategy.strategy for strategy in self.opponent_strategy_space])
        prompt = template.format(opponent_strategy_space=opponent_strategy_space)
        self.meta_strategy = Strategy.load_from_raw(self.client(prompt))
    
    def reflect_planner(self, strategy, obs, plan):
        template = OmegaConf.load("ace/templates/post_match.yaml")["REFLECT_PLANNER"]
        prompt = template.format(strategy=strategy, obs=obs, plan=plan)
        self.planner_tips =  self.client(prompt)
