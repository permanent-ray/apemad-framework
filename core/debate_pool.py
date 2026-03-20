# core/debate_pool.py
"""
多智能体辩论池模块 - 最终修复版
完全绕过 AutoGen 的 llm_config 和 OpenAI 校验
使用自定义 reply_func + 手动循环实现辩论
支持动态代理创建、角色分配、辩论运行
"""

import yaml
import os
from typing import List, Dict, Optional
from autogen.agentchat import AssistantAgent, UserProxyAgent
from autogen import GroupChat
from core.llm_wrapper import LLMWrapper

# 读取 config.yaml
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

COMPLEXITY_CONFIG = CONFIG["complexity"]
MAX_AGENTS = {
    "low": COMPLEXITY_CONFIG["max_agents_low"],
    "medium": COMPLEXITY_CONFIG["max_agents_medium"],
    "high": COMPLEXITY_CONFIG["max_agents_high"]
}

# 角色模板
ROLE_TEMPLATES = {
    "supporter": "你是支持者（Supporter），用逻辑和证据支持主要观点，语气积极、建设性。只回答当前任务，不要跑题或扩展无关内容。",
    "critic": "你是批评者（Critic），从相反角度质疑观点，指出逻辑漏洞和风险，语气理性但尖锐。只回答当前任务，不要跑题或扩展无关内容。",
    "neutral_analyzer": "你是中立分析者（Neutral Analyzer），客观总结双方观点，提供平衡分析。只回答当前任务，不要跑题或扩展无关内容。",
    "summarizer": "你是总结者（Summarizer），在每轮结束时提炼共识和分歧，语言简洁。只回答当前任务，不要跑题或扩展无关内容。",
    "verifier": "你是验证者（Verifier），检查事实准确性、证据可靠性，指出潜在幻觉。只回答当前任务，不要跑题或扩展无关内容。",
    "domain_expert": "你是领域专家（Domain Expert），提供专业知识和背景，针对任务领域。只回答当前任务，不要跑题或扩展无关内容。"
}


class DebatePool:
    """多智能体辩论池管理器"""

    def __init__(self, complexity_level: str = "medium", custom_num_agents: Optional[int] = None):
        self.complexity_level = complexity_level
        self.num_agents = custom_num_agents or MAX_AGENTS.get(complexity_level, 7)
        self.agents = []
        self.llm = LLMWrapper()  # 统一本地 LLM
        self.conversation_history = []  # 共享历史
        self.current_speaker_index = 0  # 当前发言代理索引

        self._create_agents()

    def _create_agents(self):
        """创建代理 - 不使用 AutoGen llm_config"""
        roles = ["supporter", "critic", "neutral_analyzer", "summarizer"]
        if self.complexity_level == "high":
            roles += ["verifier", "domain_expert"] * (self.num_agents // 6 + 1)
        roles = roles[:self.num_agents]

        for i, role in enumerate(roles):
            name = f"{role.capitalize()}Agent_{i+1}"
            system_prompt = ROLE_TEMPLATES.get(role, ROLE_TEMPLATES["neutral_analyzer"])
            system_prompt += f"\n你是 {name}，请基于共享历史和任务进行辩论。"

            # 使用 AssistantAgent，但不传 llm_config（避免校验）
            agent = AssistantAgent(
                name=name,
                system_message=system_prompt,
                llm_config=None,  # 关键：不传 config
                human_input_mode="NEVER",
            )
            self.agents.append(agent)

        print(f"创建 {len(self.agents)} 个代理，复杂度等级: {self.complexity_level}")

    def _generate_reply(self, agent, messages):
        """代理生成回复 - 直接调用本地 LLMWrapper"""
        system_prompt = agent.system_message  # 原始角色提示
        # 每次都强化角色，避免模型忘记
        reinforced_system = f"{system_prompt}\n记住你的角色：你是 {agent.name}，请严格按照你的角色风格回复，不要模仿其他代理。"

        history_str = "\n".join([f"{msg['name']}: {msg['content']}" for msg in messages[-5:]])  # 取最近5轮历史
        full_prompt = f"{reinforced_system}\n严格只回答当前任务，不要跑题。\n\n共享历史:\n{history_str}\n\n你的回复："

        response = self.llm.generate(
            prompt=full_prompt,
            system_prompt=reinforced_system,
            return_logits=False
        )
        return response["text"].strip()

    def run_debate(self, task_description: str, max_rounds: Optional[int] = None) -> List[Dict]:
        """
        手动运行辩论循环（绕过 AutoGen 内置调用）
        """
        max_rounds = max_rounds or (5 if self.complexity_level == "low" else 8 if self.complexity_level == "medium" else 12)

        # 初始消息
        self.conversation_history = [{"name": "User", "content": task_description}]

        print("辩论开始...")
        print(f"任务: {task_description}")

        for round_num in range(max_rounds):
            speaker = self.agents[self.current_speaker_index]
            reply = self._generate_reply(speaker, self.conversation_history)

            self.conversation_history.append({"name": speaker.name, "content": reply})
            print(f"{speaker.name}: {reply}")

            self.current_speaker_index = (self.current_speaker_index + 1) % len(self.agents)

            # 简单终止条件：如果最后3轮回复相似度高，可早停（可选扩展）
            if round_num >= 2:
                last3 = self.conversation_history[-3:]
                if len(set([msg["content"][:50] for msg in last3])) <= 1:
                    print("共识达成，早停")
                    break

        print("辩论结束")
        return self.conversation_history

    def get_last_response(self) -> str:
        if self.conversation_history:
            return self.conversation_history[-1]["content"]
        return "暂无响应"


# 测试入口
if __name__ == "__main__":
    # 测试低复杂度
    pool_low = DebatePool(complexity_level="low")
    history_low = pool_low.run_debate("计算 2 + 2 = ?")
    print("\n低复杂度最后一轮响应：", pool_low.get_last_response())

    # 测试高复杂度
    pool_high = DebatePool(complexity_level="high")
    history_high = pool_high.run_debate(
        "患者35岁男性，症状：高热、干咳、淋巴结肿大、疲劳。近期接触野生动物。诊断可能疾病，并评估风险。"
    )
    print("\n高复杂度最后一轮响应：", pool_high.get_last_response())