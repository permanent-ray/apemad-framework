# core/llm_wrapper.py
"""
统一的 LLM 调用封装，支持 HuggingFace 本地模型（含 4bit 量化）和 OpenAI/Groq API
支持返回 logits（用于不确定性计算）或仅文本输出
适配 6GB 显存电脑，使用 BitsAndBytes 4bit 量化
"""

import os
import yaml
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# 加载 .env 文件（存放 API Key 或 HF_TOKEN）
load_dotenv()

# 读取 config.yaml（假设在项目根目录/config下）
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

LLM_CONFIG = CONFIG["llm"]


class LLMWrapper:
    """统一的 LLM 调用类"""

    def __init__(self):
        self.provider = LLM_CONFIG["provider"].lower()
        self.model_name = LLM_CONFIG["model_name"]
        self.temperature = LLM_CONFIG.get("temperature", 0.7)
        self.max_tokens = LLM_CONFIG.get("max_tokens", 1024)
        self.top_p = LLM_CONFIG.get("top_p", 0.9)

        if self.provider == "huggingface":
            self._init_hf()
        elif self.provider in ["openai", "groq"]:
            self._init_api()
        else:
            raise ValueError(f"不支持的 LLM provider: {self.provider}")

    def _init_hf(self):
        """初始化 HuggingFace 本地模型，支持在线和本地路径加载"""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        from pathlib import Path

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = self.model_name

        print(f"加载 HuggingFace 模型: {model_path} on {self.device}")

        # 判断是本地路径还是在线 repo
        local_path = Path(model_path)
        if local_path.is_dir():
            print(f"使用本地路径加载: {model_path}")
            load_from = str(local_path)
        else:
            print(f"使用在线 repo 加载: {model_path}")
            load_from = model_path

        self.tokenizer = AutoTokenizer.from_pretrained(
            load_from,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            load_from,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        print("模型加载完成")

    def _init_api(self):
        """初始化 OpenAI 或 Groq API（不支持 logits）"""
        from openai import OpenAI

        api_key_env = "OPENAI_API_KEY" if self.provider == "openai" else "GROQ_API_KEY"
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"请在 .env 中设置 {api_key_env}")

        base_url = "https://api.groq.com/openai/v1" if self.provider == "groq" else None
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        return_logits: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        统一生成接口
        Args:
            prompt: 用户输入 prompt
            system_prompt: 系统提示（可选）
            return_logits: 是否返回 logits（仅 HuggingFace 支持）
        Returns:
            dict: {"text": 生成文本, "logits": logits (可选), "usage": token 消耗}
        """
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        top_p = kwargs.get("top_p", self.top_p)

        if self.provider == "huggingface":
            return self._generate_hf(prompt, system_prompt, return_logits, temperature, max_tokens, top_p)
        else:
            return self._generate_api(prompt, system_prompt, temperature, max_tokens, top_p)

    def _generate_hf(
        self,
        prompt: str,
        system_prompt: Optional[str],
        return_logits: bool,
        temperature: float,
        max_tokens: int,
        top_p: float
    ) -> Dict:
        """HuggingFace 本地生成（支持 logits 输出）"""
        from transformers import GenerationConfig
        import torch

        # 拼接完整 prompt
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)

        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_tokens,
            do_sample=True if temperature > 0 else False,
            pad_token_id=self.tokenizer.eos_token_id,
            output_scores=return_logits,                # 返回 logits
            return_dict_in_generate=True,
        )

        with torch.no_grad():
            outputs = self.model.generate(**inputs, generation_config=generation_config)

        generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        result = {
            "text": text,
            "usage": {
                "prompt_tokens": inputs.input_ids.shape[1],
                "completion_tokens": len(generated_ids)
            }
        }

        if return_logits and hasattr(outputs, "scores"):
            result["logits"] = outputs.scores  # tuple of tensors

        return result

    def _generate_api(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        top_p: float
    ) -> Dict:
        """OpenAI/Groq API 生成（不支持 logits）"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        text = response.choices[0].message.content.strip()
        usage = response.usage.dict() if hasattr(response, "usage") else {}

        return {"text": text, "usage": usage}

    def get_logits(self, prompt: str) -> Optional[list]:
        """便捷方法：只返回 logits（用于不确定性计算）"""
        result = self.generate(prompt, return_logits=True)
        return result.get("logits", None)


# 测试入口（运行本文件时可直接测试）
if __name__ == "__main__":
    llm = LLMWrapper()
    response = llm.generate(
        prompt="你好，请用一句话介绍 Qwen2.5 模型。",
        system_prompt="你是一个友好的AI助手。"
    )
    print("生成文本：", response["text"])
    print("Token 使用：", response["usage"])