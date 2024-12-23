from abc import ABC
from skill_rts import logger
from openai import OpenAI
from zhipuai import ZhipuAI
import os

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

    def call(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content


class Qwen(LLM):
    def __init__(self, model, temperature, max_tokens):
        super().__init__(model, temperature, max_tokens)
        if "qwen" in model:  # deploy on localhost
            self.client = OpenAI(base_url="http://172.18.36.55:11434/v1", api_key="ollama")
        else:
            self.client = OpenAI(base_url=os.getenv("QWEN_API_BASE"), api_key=os.getenv("QWEN_API_KEY"))


class GLM(LLM):
    def __init__(self, model, temperature, max_tokens):
        super().__init__(model, temperature, max_tokens)
        self.client = ZhipuAI()


class WebChatGPT(LLM):
    """Using the web version of ChatGPT, need MANUALLY copy the output"""
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


class Llama(LLM):
    def __init__(self, model, temperature, max_tokens):
        super().__init__(model, temperature, max_tokens)
        self.client = OpenAI(base_url="http://172.18.36.59:11434/v1", api_key="ollama")


class TaiChu(LLM):
    def __init__(self, model="taichu_70b", temperature=0, max_tokens=8192):
        super().__init__(model, temperature, max_tokens)
        self.client = OpenAI(base_url=os.getenv("TAICHU_API_BASE"), api_key=os.getenv("TAICHU_API_KEY"))


class LLMs(LLM):
    def __init__(self, model, temperature, max_tokens):
        super().__init__(model, temperature, max_tokens)
        if "qwen" in model.lower():
            self.client = Qwen(model, temperature, max_tokens)
        elif "taichu" in model.lower():
            self.client = TaiChu(model, temperature, max_tokens)
        elif "llama" in model.lower():
            self.client = Llama(model, temperature, max_tokens)
        elif "glm" in model.lower():
            self.client = GLM(model, temperature, max_tokens)
        else:
            raise ValueError(f"Model {model} not available.")
    

    def __call__(self, prompt: str) -> str | None:
        return self.client(prompt)


if __name__ == "__main__":
    llm = LLMs("Qwen2.5-72B-Instruct", 0, 8192)
    print(llm("who are you"))
    