import yaml
import os
from omegaconf import OmegaConf
from skill_rts.agents import Agent
from skill_rts.game.trajectory import Trajectory
from skill_rts.game.game_state import GameState
from skill_rts import logger
from ace.strategy import Strategy
from ace.traj_feat import TrajectoryFeature
from ace.offline.payoff_net import PayoffNet
import pandas as pd
import json


class Planner(Agent):
    def __init__(
        self, 
        model: str, 
        prompt: str, 
        temperature: float, 
        max_tokens: int, 
        map_name: str,
        player_id: int,
        strategy: str | Strategy = None,
    ):
        """
        Args:
            model (str): foundation large language model name
            prompt (str): prompt type, e.g. "few-shot-w-strategy"
            temperature (float): temperature for sampling
            max_tokens (int): max tokens for generation
            map_name (str): map name
            player_id (int): player id, 0 for blue side, 1 for red side
            strategy (str | Strategy): strategy file path or strategy string or Strategy object
        """
        super().__init__(model, temperature, max_tokens)
        self.prompt = prompt
        self.map_name = map_name
        self.player_id = player_id
        self.template = self._get_template()
        if "few-shot" in self.prompt:
            self._get_examples()
        if "strategy" in self.prompt:
            self._get_strategy(strategy)
    
    def _get_template(self) -> str:
        prompt = {
            "zero-shot": "ZERO_SHOT",
            "few-shot": "FEW_SHOT",
            "zero-shot-w-strategy": "ZERO_SHOT_W_STRATEGY",
            "few-shot-w-strategy": "FEW_SHOT_W_STRATEGY"
        }[self.prompt]
        return OmegaConf.load("ace/templates/planner.yaml")[prompt]
    
    def step(self, obs: GameState, *args, **kwargs) -> str:
        """Make a task plan based on the observation.

        Args:
            obs (GameState): The observation from the environment.
        
        Returns:
            str: The task plan.
        """
        self.obs = obs
        prompt = self._get_prompt()
        response = self.client(prompt)
        logger.debug(f"Prompt: {prompt}")
        logger.debug(f"Response: {response}")
        return response
    
    def _get_prompt(self):
        kwargs = {"observation": self.obs.to_string(), "player_id": self.player_id}
        if self.prompt == "zero-shot":
            return self.template.format(**kwargs)
        elif self.prompt == "few-shot":
            return self.template.format(examples=self.examples, **kwargs)
        elif self.prompt == "zero-shot-w-strategy":
            return self.template.format(strategy=self.strategy, **kwargs)
        elif self.prompt == "few-shot-w-strategy":
            return self.template.format(examples=self.examples, strategy=self.strategy, **kwargs)
    
    def _get_examples(self):
        with open(f"ace/templates/example_{self.map_name}.yaml") as f:
            self.examples = yaml.safe_load(f)["EXAMPLES"][self.player_id]
    
    def _get_strategy(self, strategy):
        if isinstance(strategy, Strategy):
            self.strategy = str(strategy)
        elif isinstance(strategy, str):
            if os.path.isfile(strategy):
                with open(strategy) as f:
                    strategy = json.load(f)
                self.strategy = strategy["strategy"] + strategy["description"]
            else:
                self.strategy = strategy
        else:
            self.strategy = ""


class Recognizer(Agent):
    def __init__(self, model, temperature, max_tokens):
        super().__init__(model, temperature, max_tokens)
        self._get_prompt()
        self.max_retry = 3
    
    def _get_prompt(self):
        self.template = OmegaConf.load("ace/templates/recognizer.yaml")["TEMPLATE"]
    
    def step(self, traj: str, *args, **kwargs) -> Strategy:
        prompt = self.template.format(trajectory=traj)
        for _ in range(self.max_retry):
            try:
                response = self.client(prompt)
                strategy = Strategy(response, "")
                strategy.encode()  # check if the strategy valid
                return strategy
            except Exception as e:
                logger.info(f"Recognizer error {e}, retrying...")
                logger.info(f"Wrong response:\n{response}")
        raise ValueError("Recognizer failed to generate a valid strategy")


class AceAgent(Agent):
    strategy_dir = "ace/data/train"

    def __init__(self, player_id, model, temperature, max_tokens, map_name, strategy_interval=1):
        super().__init__(model, temperature, max_tokens)
        self.player_id = player_id
        self.strategy_interval = strategy_interval
        self.planner = Planner(model, "few-shot-w-strategy", temperature, max_tokens, map_name, player_id)
        self.recognizer = Recognizer(model, temperature, max_tokens)
        self.payoff_matrix = None
        self.payoff_net = None
        # initialized meta strategy is the highest average payoff strategy
        self.meta_strategy = Strategy.load_from_json(f"{self.strategy_dir}/strategy_23.json")
        self.strategy = self.meta_strategy.to_string()
        self.planner.strategy = self.strategy
    
    def step(self, obs: GameState, traj: Trajectory | None):
        if obs.time % self.strategy_interval == 0 and traj is not None:
            self.update_strategy(traj)
        return self.planner.step(obs)
    
    def reset(self):
        self.strategy = self.meta_strategy.to_string()
        self.planner.strategy = self.strategy
    
    def update_strategy(self, traj: Trajectory):
        abs_traj = TrajectoryFeature(traj).to_string()
        opponent = self.recognizer.step(abs_traj)
        idx = self.match_strategy(opponent)
        if idx:  # seen opponent
            logger.debug(f"Matched strategy: {idx}")
            self.strategy = self.response4seen(idx)
        else:  # unseen opponent
            logger.debug(f"Unseen opponent:\n{opponent}")
            self.strategy, win_rate = self.response4unseen(opponent)
        self.planner.strategy = self.strategy
    
    def match_strategy(self, opponent):
        for filename in os.listdir(f"{self.strategy_dir}"):
            if filename.endswith(".json"):
                s = Strategy.load_from_json(f"{self.strategy_dir}/{filename}")
                if s == opponent:
                    return filename.split("_")[1].split(".")[0]
        return None
    
    def response4seen(self, idx) -> str:
        if self.payoff_matrix is None:
            self.payoff_matrix = pd.read_csv("ace/data/payoff/payoff_matrix.csv", index_col=0)
        payoff = self.payoff_matrix[idx]
        resp_idx = payoff.idxmax()
        logger.info(f"Match for seen: strategy_{idx} -> strategy_{resp_idx}")
        with open(f"{self.strategy_dir}/strategy_{resp_idx}.json") as f:
            d = json.load(f)
        return d["strategy"] + d["description"]
    
    def response4unseen(self, opponent: Strategy) -> str:
        if self.payoff_net is None:
            self.payoff_net = PayoffNet.load("ace/data/payoff/payoff_net.pth")
        feat_space = Strategy.feat_space()
        response, win_rate = self.payoff_net.search_best_response(feat_space, opponent.feats)
        if win_rate < 0.5:
            response = self.meta_strategy
        logger.info(f"Search for unseen win rate: {win_rate}")
        logger.info(f"Best strategy: {response.strategy}")
        return response.strategy, win_rate


class NaiveAgent(Agent):
    def __init__(self, player_id, plan=None):
        self.player_id = player_id
        self.strategy = ""
        if plan is not None:
            self.plan = plan
        else:
            self.plan = """START OF TASK
                [Harvest Mineral](0, 0)  # one worker harvests minerals
                [Harvest Mineral](0, 0)  # another worker harvests minerals
                [Produce Unit](worker, east)
                [Produce Unit](worker, south)
                [Produce Unit](worker, east)
                [Produce Unit](worker, south)
                [Build Building](barracks, (0, 3), resource >= 7)
                [Produce Unit](ranged, east)
                [Produce Unit](ranged, south)
                [Produce Unit](ranged, east)
                [Produce Unit](ranged, south)
                [Attack Enemy](worker, base)  # when no barracks use worker to attack
                [Attack Enemy](worker, barracks)
                [Attack Enemy](worker, worker)
                [Attack Enemy](worker, worker)
                [Attack Enemy](worker, barracks)
                [Attack Enemy](worker, base)
                [Attack Enemy](ranged, base)  # when has barracks use ranged to attack
                [Attack Enemy](ranged, barracks)
                [Attack Enemy](ranged, worker)
                [Attack Enemy](ranged, worker)
                [Attack Enemy](ranged, barracks)
                [Attack Enemy](ranged, base)
                END OF TASK"""
    
    def step(self, *args, **kwargs):
        return self.plan


if __name__ == "__main__":
    from skill_rts.envs.wrappers import MicroRTSLLMEnv
    import time

    agent_config = {
        "model": "Qwen2.5-72B-Instruct",
        "temperature": 0,
        "max_tokens": 8192
    }
    agent = AceAgent(0, **agent_config)
    opponent = Planner(
        player_id=1, 
        prompt="few-shot-w-strategy",
        strategy="ace/data/strategies/strategy_1.json",
        **agent_config
    )
    start = time.time()
    env = MicroRTSLLMEnv([agent, opponent], record_video=True)
    logger.set_level(logger.DEBUG)

    payoffs, trajectory = env.run()
    print(f"Payoffs: {payoffs} | Steps: {env.time} | Runtime: {(time.time() - start) / 60:.2f} min")
