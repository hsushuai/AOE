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
import random
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
        self.strategy = strategy
    
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
            return self.prompt_template.format(observation=self.obs, player_id=self.player_id, examples=self._get_shot(), strategy=self._get_strategy())
    
    def _get_shot(self):
        with open(f"ace/configs/templates/example_{self.map_name}.yaml") as f:
            return yaml.safe_load(f)["EXAMPLES"][self.player_id]
    
    def _get_strategy(self):
        if isinstance(self.strategy, Strategy):
            return str(self.strategy)
        elif isinstance(self.strategy, str):
            if os.path.isfile(self.strategy):
                with open(self.strategy) as f:
                    strategy = yaml.safe_load(f)
                return strategy["strategy"] + strategy["description"]
            return self.strategy
        return ""


class Recognizer(Agent):
    def __init__(self, model, temperature, max_tokens):
        super().__init__(model, temperature, max_tokens)
        self._get_prompt()
    
    def _get_prompt(self):
        self.prompt_template = OmegaConf.load("ace/in_match/config/template.yaml")["RECOGNIZE_TEMPLATE"]
    
    def step(self, traj: str, *args, **kwargs) -> Strategy:
        prompt = self.prompt_template.format(trajectory=traj)
        response = self.client(prompt)
        return Strategy(response, "")


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
        self.meta_strategy = Strategy.load_from_json(f"{self.strategy_dir}/strategies/strategy_35.json")
    
    def step(self, obs: str, traj: Trajectory | None):
        if traj:
            abs_traj = TrajectoryFeature(traj).to_string()
            opponent = self.recognizer.step(abs_traj)
            idx = self.match_strategy(opponent)
            if idx:
                logger.debug(f"Matched strategy: {idx}")
                strategy = self.response4seen(idx)
            else:
                logger.debug(f"Unseen opponent:\n{opponent}")
                strategy, win_rate = self.response4unseen(opponent)
                strategy = strategy if win_rate >= 0.8 else self.meta_strategy
            self.planner.strategy = strategy
        else:
            self.planner.strategy = self.meta_strategy
        return self.planner.step(obs)
    
    def match_strategy(self, opponent):
        for filename in os.listdir(f"{self.strategy_dir}/opponents"):
            if filename.endswith(".json"):
                s = Strategy.load_from_json(f"{self.strategy_dir}/opponents/{filename}")
                if s == opponent:
                    return filename.split("_")[1].split(".")[0]
        return None
    
    def response4seen(self, idx) -> str:
        if self.payoff_matrix is None:
            self.payoff_matrix = pd.read_excel("ace/data/payoff/payoff_matrix.csv", index_col=0)
        payoff = self.payoff_matrix[idx]
        resp_idx = payoff.idxmax()
        logger.info(f"Match for seen: strategy_{idx} -> strategy_{resp_idx}")
        with open(f"{self.strategy_dir}/strategies/strategy_{resp_idx}.json") as f:
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
            strategies = [Strategy.decode(feat, one_hot=False) for feat in feats]
            X0 = np.vstack([strategy.one_hot_feats for strategy in strategies])
            X1 = opponent.one_hot_feats
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
        response = Strategy.decode(best_feats, one_hot=False).strategy
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
        strategy="ace/data/opponents/strategy_1.json",
        **agent_config
    )
    start = time.time()
    env = MicroRTSLLMEnv([agent, opponent], record_video=True)
    logger.set_level(logger.DEBUG)

    payoffs, trajectory = env.run()
    print(f"Payoffs: {payoffs} | Steps: {env.time} | Runtime: {(time.time() - start) / 60:.2f} min")
