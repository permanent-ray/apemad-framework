# test_llm_wrapper.py
from core.llm_wrapper import LLMWrapper

llm = LLMWrapper()

# 测试文本生成
response = llm.generate(
    prompt="你好，请用一句话介绍自己。",
    system_prompt="你是一个友好的AI助手。"
)

print("生成文本：", response["text"])
print("Token 使用：", response["usage"])

# 如果是 HuggingFace 模型，测试 logits
if llm.provider == "huggingface":
    logits = llm.get_logits("你好")
    print("Logits 类型：", type(logits) if logits else "None")