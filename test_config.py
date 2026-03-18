# test_config.py
import yaml

try:
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print("config.yaml 加载成功！")
    print("LLM 模型:", config["llm"]["model_name"])
    print("复杂度权重 α:", config["complexity"]["weights"]["length"])
    print("高复杂度代理上限:", config["complexity"]["max_agents_high"])
except Exception as e:
    print("加载失败:", e)