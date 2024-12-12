import yaml
import os
from omegaconf import OmegaConf
from ace.configs.templates import zero_shot, few_shot, few_shot_w_strategy
from skill_rts.agents.llm_clients import Qwen, GLM, WebChatGPT
from skill_rts.game.trajectory import Trajectory
from skill_rts import logger
from ace.strategy import Strategy
from ace.traj_feat import TrajectoryFeature
from ace.pre_match.payoff_net import PayoffNet
import pandas as pd
import numpy as np
import json
import torch


class Agent:
    def __init__(self, model: str, temperature: float, max_tokens: int):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = self._get_client()
    
    def _get_client(self):
        if "qwen" in  self.model.lower():
            return Qwen(self.model, self.temperature, self.max_tokens)
        elif "glm" in self.model.lower():
            return GLM(self.model, self.temperature, self.max_tokens)
        elif "chatgpt" in self.model.lower():
            return WebChatGPT(self.model, self.temperature, self.max_tokens)
        else:
            raise ValueError("Model not supported")


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
        self.prompt_template = self._get_prompt_template()
        self.tips = ""
        self._get_strategy(strategy)
    
    def _get_prompt_template(self) -> str:
        return {
            "zero-shot": zero_shot,
            "few-shot": few_shot,
            "few-shot-w-strategy": few_shot_w_strategy
        }[self.prompt]
    
    def step(self, obs: str, *args, **kwargs) -> str:
        """Make a task plan based on the observation.

        Args:
            obs (str): The observation from the environment.
        
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
        if self.prompt == "zero-shot":
            return self.prompt_template.format(observation=self.obs, player_id=self.player_id)
        elif self.prompt == "few-shot":
            return self.prompt_template.format(observation=self.obs, player_id=self.player_id, examples=self._get_shot())
        elif self.prompt == "few-shot-w-strategy":
            return self.prompt_template.format(observation=self.obs, player_id=self.player_id, examples=self._get_shot(), strategy=self.strategy, tips=self.tips)
    
    def _get_shot(self):
        with open(f"ace/configs/templates/example_{self.map_name}.yaml") as f:
            return yaml.safe_load(f)["EXAMPLES"][self.player_id]
    
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
    
    def _get_prompt(self):
        self.prompt_template = OmegaConf.load("ace/in_match/config/template.yaml")["TEMPLATE"]
    
    def step(self, traj: str, *args, **kwargs) -> Strategy:
        prompt = self.prompt_template.format(trajectory=traj)
        while True:
            try:
                response = self.client(prompt)
                strategy = Strategy(response, "")
                strategy.encode()  # check if the strategy is valid
                return strategy
            except Exception as e:
                print(f"Recognizer error {e}, retrying...")
                print(f"Wrong response:\n{response}")
        raise ValueError("Recognizer failed to generate a valid strategy")


class AceAgent(Agent):
    strategy_dir = "ace/data/train"

    def __init__(self, player_id, model, temperature, max_tokens, map_name):
        super().__init__(model, temperature, max_tokens)
        self.player_id = player_id
        self.planner = Planner(model, "few-shot-w-strategy", temperature, max_tokens, map_name, player_id)
        self.recognizer = Recognizer(model, temperature, max_tokens)
        self.payoff_matrix = None
        self.payoff_net = None
        # initialized meta strategy is the highest average payoff strategy
        self.meta_strategy = Strategy.load_from_json(f"{self.strategy_dir}/strategy_38.json")
        self.strategy = self.meta_strategy.to_string()
    
    def step(self, obs: str, traj: Trajectory | None):
        if traj:
            abs_traj = TrajectoryFeature(traj).to_string()
            opponent = self.recognizer.step(abs_traj)
            idx = self.match_strategy(opponent)
            if idx:  # seen opponent
                logger.debug(f"Matched strategy: {idx}")
                self.strategy = self.response4seen(idx)
            else:  # unseen opponent
                logger.debug(f"Unseen opponent:\n{opponent}")
                strategy, win_rate = self.response4unseen(opponent)
                win_rate = 0
                self.strategy = strategy if win_rate >= 0.8 else self.meta_strategy.to_string()
            self.planner.strategy = self.strategy
        else:
            self.planner.strategy = self.meta_strategy
        return self.planner.step(obs)
    
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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.payoff_net is None:
            with open("ace/pre_match/config/payoff.yaml") as f:
                config = yaml.safe_load(f)
            self.payoff_net = PayoffNet(**config["model"])
            self.payoff_net.load_state_dict(torch.load("ace/data/payoff/payoff_net.pth", weights_only=True))
            self.payoff_net.to(device)
        
        feat_space = Strategy.feat_space()
        batch_size = 1024
        best_feats = None
        best_win_rate = -float("inf")
        max_iter = len(feat_space) // batch_size if len(feat_space) % batch_size == 0 else len(feat_space) // batch_size + 1
        for i in range(max_iter):
            end = (i + 1) * batch_size
            end = end if end < len(feat_space) else len(feat_space)
            feats = feat_space[i * batch_size : end]
            # prepare input data
            strategies = [Strategy.decode(feat) for feat in feats]
            X0 = np.vstack([strategy.feats for strategy in strategies])
            X1 = opponent.feats
            X1 = np.tile(X1, (X0.shape[0], 1))
            X = torch.tensor(np.hstack([X0, X1]), dtype=torch.float32, device=device)
            with torch.no_grad():
                win_rate = torch.nn.functional.softmax(self.payoff_net(X), dim=1)
            max_idx = win_rate[:, 0].argmax().item()
            if win_rate[max_idx, 0] > best_win_rate:
                best_win_rate = win_rate[max_idx, 0]
                best_feats = feats[max_idx]
                if best_win_rate >= 0.8:
                    break
        response = Strategy.decode(best_feats).strategy
        logger.info(f"Search for unseen win rate: {best_win_rate}")
        logger.info(f"Best strategy: {response}")
        return response, best_win_rate


if __name__ == "__main__":
    from skill_rts.envs.wrappers import MicroRTSLLMEnv
    import time

    agent_config = {
        "model": "Qwen2.5-72B-Instruct",
        "temperature": 0,
        "max_tokens": 8192,
        "map_name": "basesWorkers8x8"
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


class NaiveAgent(Agent):
    def __init__(self, player_id):
        self.player_id = player_id
        self.strategy = """\
        ## Strategy
        - Economic Feature: 2
        - Barracks Feature: resource >= 7
        - Military Feature: Ranged and Worker
        - Aggression Feature: True
        - Attack Feature: Building
        - Defense Feature: None
        """
    
    def step(self, *args, **kwargs):
        s = """\
        START OF TASK
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

        return s