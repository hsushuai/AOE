import yaml
from ace.configs.templates import zero_shot, few_shot, few_shot_w_strategy
from skill_rts.agents.llm_clients import Qwen, GLM, ChatGPT
from skill_rts import logger


class AceAgent:
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
        self.model = model
        self.prompt = prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.map_name = map_name
        self.player_id = player_id
        self.client = self._get_client()
        self.prompt_template = self._get_prompt_template()
        self.strategy = strategy
    
    def _get_client(self):
        if "qwen" in  self.model.lower():
            return Qwen(self.model, self.temperature, self.max_tokens)
        elif "glm" in self.model.lower():
            return GLM(self.model, self.temperature, self.max_tokens)
        elif "chatgpt" in self.model.lower():
            return ChatGPT(self.model, self.temperature, self.max_tokens)
        else:
            raise ValueError("Model not supported")
    
    def _get_prompt_template(self) -> str:
        return {
            "zero-shot": zero_shot,
            "few-shot": few_shot,
            "few-shot-w-strategy": few_shot_w_strategy
        }[self.prompt]
    
    def step(self, obs: str) -> str:
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
        with open(f"ace/configs/templates/planning_{self.map_name}.yaml") as f:
            return yaml.safe_load(f)["EXAMPLES"][self.player_id]
    
    def _get_strategy(self):
        with open(self.strategy) as f:
            return yaml.safe_load(f)["raw_response"]
