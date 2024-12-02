from abc import ABC, abstractmethod
from skill_rts import logger

class LLM(ABC):
    """Base class for LLM"""
    def __init__(self, model, temperature, max_tokens):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = None

    def __call__(self, prompt: str) -> str | None:
        if self.is_excessive_token(prompt):
            raise ValueError("The prompt exceeds the maximum input token length limit.")
        try:
            return self.call(prompt)
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")

    def is_excessive_token(self, prompt: str) -> bool:
        pass

    @abstractmethod
    def call(self, prompt: str) -> str:
        raise NotImplementedError


class Qwen(LLM):
    def __init__(self, model, temperature, max_tokens):
        from openai import OpenAI
        import os

        super().__init__(model, temperature, max_tokens)
        if "qwen" in model:  # deploy on localhost
            self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        else:
            self.client = OpenAI(base_url=os.getenv("QWEN_API_BASE"), api_key=os.getenv("QWEN_API_KEY"))
        # openai.Model.list()
    
    def call(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content


class GLM(LLM):
    def __init__(self, model, temperature, max_tokens):
        from zhipuai import ZhipuAI

        super().__init__(model, temperature, max_tokens)
        self.client = ZhipuAI()
    
    def call(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content

class ChatGPT(LLM):
    def __init__(self, *args, **kwargs):
        pass

    def call(self, prompt: str) -> str:
        print(prompt)
        response = []
        while True:
            line = input()
            if line == "":
                break
            response.append(line)
        return "\n".join(response)


if __name__ == "__main__":
    # llm = GLM("glm-4-flash", 0, 1024)
    llm = Qwen("qwen2.5:72b-instruct-q4_K_M", 0, 1024)
    print(llm("who are you"))
