from abc import ABC
from email import header
from operator import call
from skill_rts import logger
from openai import OpenAI
from zhipuai import ZhipuAI
import os
import requests


class LLM(ABC):
    """Base class for LLM"""
    def __init__(self, model, temperature, max_tokens):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = None

    def __call__(self, prompt: str=None, messages=None) -> str | None:
        if self.is_excessive_token(prompt):
            raise ValueError("The prompt exceeds the maximum input token length limit.")
        try:
            return self.call(prompt, messages)
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            logger.error(f"Input prompt: {prompt}")

    def is_excessive_token(self, prompt: str) -> bool:
        pass

    def call(self, prompt: str=None, messages: list=None) -> str:
        """Call the LLM with a prompt or messages."""
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content


class SapAgent(LLM):
    def __init__(self, model, temperature, max_tokens):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def call(self, prompt: str=None, messages=None) -> str:
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        response = call_llm(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )
        return response


class Qwen(LLM):
    def __init__(self, model, temperature, max_tokens):
        super().__init__(model, temperature, max_tokens)
        if "qwen" in model:  # deploy on localhost
            self.client = OpenAI(base_url="http://172.18.36.55:11434/v1", api_key="")
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

    def call(self, prompt: str, **kwargs) -> str:
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
    def __init__(self, model, temperature=0, max_tokens=8192):
        super().__init__(model, temperature, max_tokens)
        if "qwen" in model.lower():
            self.client = Qwen(model, temperature, max_tokens)
        elif "taichu" in model.lower():
            self.client = TaiChu(model, temperature, max_tokens)
        elif "llama" in model.lower():
            self.client = Llama(model, temperature, max_tokens)
        elif "glm" in model.lower():
            self.client = GLM(model, temperature, max_tokens)
        elif "sap" in model.lower():
            self.client = SapAgent(model, temperature, max_tokens)
        else:
            raise ValueError(f"Model {model} not available.")
    

    def __call__(self, prompt: str=None, messages: list=None) -> str | None:
        return self.client(prompt, messages)


def call_llm(
        model: str,
        prompt: str=None,
        base_url: str="http://127.0.0.1:20020/v1",
        api_key: str = None,
        messages: list=None,
        **sampling_params
    ) -> str | None:
    if messages is None:
        messages = [{"role": "user", "content": prompt}]
    api_key = "" if api_key is None else api_key
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": model,
        "messages": messages,
        **sampling_params
    }
    response = requests.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json=data
    )
    if response.status_code != 200:
        logger.error(f"Error calling LLM: {response.status_code} {response.text}")
        return None
    response_data = response.json()
    if "choices" not in response_data or len(response_data["choices"]) == 0:
        logger.error("No choices returned from LLM.")
        return None
    return response_data["choices"][0]["message"]["content"]


if __name__ == "__main__":
    llm = LLMs("sap_distill")
    print(llm("who are you"))
