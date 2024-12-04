import yaml
import os
from omegaconf import OmegaConf
from ace.configs.templates import zero_shot, few_shot, few_shot_w_strategy
from skill_rts.agents.llm_clients import Qwen, GLM, WebChatGPT
from skill_rts import logger
from ace.strategy import Strategy
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
        strategy: str = None,
    ):
        """
        Args:
            model (str): foundation large language model name
            prompt (str): prompt type, e.g. "few-shot-w-strategy"
            temperature (float): temperature for sampling
            max_tokens (int): max tokens for generation
            map_name (str): map name
            player_id (int): player id, 0 for blue side, 1 for red side
            strategy (str): strategy file path or strategy string
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
        if os.path.isfile(self.strategy):
            with open(self.strategy) as f:
                strategy = yaml.safe_load(f)
            return strategy["strategy"] + strategy["description"]
        elif self.strategy is not None:
            return self.strategy
        else:
            return ""


class Recognizer(Agent):
    def __init__(self, model, temperature, max_tokens):
        super().__init__(model, temperature, max_tokens)
    
    def _get_prompt(self):
        self.prompt_template = OmegaConf.load("ace/in_match/config/template.yaml")["RECOGNIZE_TEMPLATE"]
    
    def step(self, traj: str, *args, **kwargs):
        prompt = self.prompt_template.format(trajectory=traj)
        return self.client(prompt)


class AceAgent(Agent):
    strategy_dir = "ace/data/train"

    def __init__(self, model, temperature, max_tokens, map_name):
        super().__init__(model, temperature, max_tokens)
        self.planner = Planner(model, "few-shot-w-strategy", temperature, max_tokens, map_name, 0)
        self.recognizer = Recognizer(model, temperature, max_tokens)
        self.payoff_matrix = None
        self.payoff_net = None
    
    def step(self, obs, traj):
        opponent = self.recognizer.step(traj)
        opponent = Strategy(opponent, "")
        idx = self.match_strategy(opponent)
        if idx:
            strategy = self.response4seen(idx)
        else:
            strategy = self.response4unseen(opponent)
        self.planner.strategy = strategy
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
        if self.payoff_net is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.payoff_net = torch.load("ace/data/payoff/payoff_net.pth", device=device)
        
        feat_space = Strategy.feat_space()
        n_samples = 1000
        best_feats = None
        best_win_rate = -float("inf")
        for _ in range(n_samples):
            feats = feat_space[np.random.choice(feat_space.shape[0])]
            strategy = Strategy.decode(feats, one_hot=False)
            x = torch.tensor([strategy.one_hot_feats, opponent.one_hot_feats], dtype=torch.float32, device=device)
            with torch.no_grad():
                win_rate = torch.F.softmax(self.payoff_net(x))
            if win_rate[0] > best_win_rate:
                best_win_rate = win_rate[0]
                best_feats = feats
        logger.info(f"Search for unseen win rate: {best_win_rate}")
        logger.info(f"Best strategy: {best_feats}")
        return Strategy.decode(best_feats, one_hot=False).strategy
