from skill_rts.agents.llm_clients import Qwen, GLM, LLM


class LLMAgent:
    """A LLM agent for Leduc Hold"em. It can be used to play against trained models"""
    def __init__(self, config: dict, num_actions: int):
        """Initialize the LLM agent

        Args:
            config (dict): LLM configuration
            num_actions (int): the size of the output action space
        """
        self.model = config["model"]
        self.temperature = config["temperature"]
        self.max_tokens = config["max_tokens"]
        self.prompt = config["prompt"]
        self.num_actions = num_actions

        self.llm = self._get_llm()
        self.prompt_template = self._get_prompt_template()

        self.use_raw = True   

    def call(self, observation: str, legal_actions: str) -> str:
        import re

        prompt_content = self.prompt_template.format(observation=observation, legal_actions=legal_actions)
        response = self.llm(prompt_content)
        return re.search(r"action:\s*(\w+)", response).group(1)
    
    def step(self, state):
        """LLM agent will display the state and make decisions through interfaces

        Args:
            state (dict): A dictionary that represents the current state

        Returns:
            action (int): The action decided by human
        """
        # _print_state(state["raw_obs"], state["action_record"])
        action = self.call(state["raw_obs"], state["raw_legal_actions"])
        while action not in state["raw_legal_actions"]:
            print("Action illegal...")
            action = input(">> Re-choose action (str): ")
        return action
    
    def eval_step(self, state):
        """Predict the action given the current state for evaluation. The same to step here.

        Args:
            state (numpy.array): an numpy array that represents the current state

        Returns:
            action (int): the action predicted (randomly chosen) by the random agent
        """
        return self.step(state), {}
    
    def _get_llm(self) -> LLM:
        if "qwen" in self.model.lower():
            return Qwen(self.model, self.temperature, self.max_tokens)
        elif "glm" in self.model.lower():
            return GLM(self.model, self.temperature, self.max_tokens)
        else:
            raise ValueError(f"Invalid model name: {self.model}")
    
    def _get_prompt_template(self) -> str:
        try:
            return getattr(prompts, self.prompt)
        except AttributeError as e:
            avalible_prompts = [p for p in dir(prompts) if not p.startswith("__")]
            raise ValueError(f"{e}\nThe available prompts are: {avalible_prompts}")