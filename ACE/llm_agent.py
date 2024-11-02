from ACE.configs.templates import zero_shot, few_shot
from ACE.clients import Qwen, GLM, ChatGPT
from skill_rts import logger


class LLMAgent:
    def __init__(
        self, 
        model: str, 
        prompt: str, 
        temperature: float, 
        max_tokens: int, 
        map_path: str,
        player_id: int,
    ):
        self.model = model
        self.prompt = prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.map = map_path.split("/")[-1].split(".")[0]
        self.player_id = player_id
        self.client = self._get_client()
        self.prompt_template = self._get_prompt_template()
    
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
            "few-shot": few_shot
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
        if "zero_shot" in self.prompt:
            return self.prompt_template.format(observation=self.obs, player_id=self.player_id)
        else:
            return self.prompt_template.format(observation=self.obs, player_id=self.player_id, examples=self._get_shot())
    
    def _get_shot(self):
        import yaml

        with open(f"ACE/configs/templates/{self.map}.yaml") as f:
            return yaml.safe_load(f)["EXAMPLES"][self.player_id]
