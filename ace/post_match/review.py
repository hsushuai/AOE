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
    def __init__(self, model, temperature, max_token):
        self.recognizer = Recognizer(model, temperature, max_token)
        self.client = self.recognizer.client
        self.opponent_strategy_space = set()
        self.meta_strategy = None
        self.planner_tips = set()
    
    def step(self, strategy, plan, traj: Trajectory):
        traj_feat = TrajectoryFeature(traj)
        traj_feat_str = traj_feat.to_string()
        self.analyze_strategy_space(traj_feat_str)
        self.reflect_meta_strategy()
        self.reflect_planner(strategy, plan, traj)
    
    def analyze_strategy_space(self, traj_feat_str):
        strategy = self.recognizer.step(traj_feat_str)
        self.opponent_strategy_space.add(Strategy(strategy, ""))
    
    def reflect_meta_strategy(self):
        template = OmegaConf.load("ace/post_match/config/template.yaml")["REFLECT_META_STRATEGY"]
        opponent_strategy_space = "\n".join([strategy.strategy] for strategy in self.opponent_strategy_space)
        prompt = template.format(opponent_strategy_space=opponent_strategy_space)
        self.meta_strategy = Strategy.load_from_raw(self.client(prompt))
    
    def reflect_planner(self, strategy, plan, traj_feat_str):
        template = OmegaConf.load("ace/post_match/config/template.yaml")["REFLECT_PLANNER"]
        prompt = template.format(strategy=strategy, plan=plan, trajectory=traj_feat_str)
        self.planner_tips.add(self.client(prompt).split("\n"))


if __name__ == "__main__":
    pass