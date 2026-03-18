# core/complexity_assessor.py
"""
任务复杂度评估模块 - 优化版
输入：任务描述文本 + 可选多模态文件
输出：复杂度分数 C (0-1) + 等级 + 推荐代理数量
"""

import yaml
import os
import spacy
from typing import Optional, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np

# 读取 config.yaml
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

COMPLEXITY_CONFIG = CONFIG["complexity"]
WEIGHTS = COMPLEXITY_CONFIG["weights"]
THRESHOLDS = COMPLEXITY_CONFIG["thresholds"]
MAX_AGENTS = {
    "low": COMPLEXITY_CONFIG["max_agents_low"],
    "medium": COMPLEXITY_CONFIG["max_agents_medium"],
    "high": COMPLEXITY_CONFIG["max_agents_high"]
}

# 全局加载 spaCy
NLP = spacy.load("en_core_web_sm")

# 全局加载 sentence-transformers（用于 U 计算）
SEMANTIC_MODEL = SentenceTransformer('all-MiniLM-L6-v2')


class TaskComplexityAssessor:
    """任务复杂度评估器 - 优化版"""

    def __init__(self):
        self.max_length = 4096  # 最大 token 长度参考

    def compute_complexity(self, task_text: str, images: Optional[list] = None, tables: Optional[list] = None) -> Tuple[
        float, str, int]:
        """
        计算任务复杂度分数 C - 优化版
        """
        # 1. 输入长度归一化 (L) - 更敏感（短文本也容易复杂）
        tokens_approx = len(task_text.split()) + len(task_text) / 5  # 结合字符数
        L = min(tokens_approx / 300, 1.0)  # 分母调低，300 tokens 就接近 1

        # 2. 实体关系依赖深度 (D) - 优化版
        doc = NLP(task_text)
        entities = len(doc.ents)
        if len(doc) > 0:
            depths = []
            for token in doc:
                depth = 0
                current = token
                visited = set()
                while current.head != current and current.i not in visited:
                    depth += 1
                    visited.add(current.i)
                    current = current.head
                depths.append(depth)
            avg_depth = np.mean(depths) if depths else 0.0
            entity_density = entities / (len(doc) + 1)
            D = min(avg_depth / 5 + entity_density * 3, 1.0)  # 除数调低，实体密度 *3
        else:
            D = 0.0

        # 3. 初步不确定性 (U) - 语义多样性优化
        U = 0.0
        if len(task_text) > 20:
            try:
                sentences = [s.strip() for s in task_text.split('.') if s.strip()]
                if len(sentences) >= 2:
                    embeddings = SEMANTIC_MODEL.encode(sentences)
                    sims = []
                    for i in range(len(embeddings)):
                        for j in range(i + 1, len(embeddings)):
                            sim = np.dot(embeddings[i], embeddings[j]) / (
                                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8)
                            sims.append(sim)
                    avg_sim = np.mean(sims) if sims else 1.0
                    U = 1.0 - avg_sim  # 相似度越低，不确定性越高
                    U = U ** 1.5  # 指数放大，让差异更明显
                else:
                    U = 0.7  # 句子少，默认较高不确定性
            except Exception as e:
                print(f"语义不确定性计算失败: {e}, 默认 U=0.7")
                U = 0.7
        else:
            U = 0.0

        # 4. 多模态复杂度 (M)
        M = 0.0
        if images:
            M += len(images) * 0.6
        if tables:
            M += len(tables) * 0.6
        M = min(M, 1.0)

        # 加权计算 C
        C = (
                WEIGHTS["length"] * L +
                WEIGHTS["depth"] * D +
                WEIGHTS["uncertainty"] * U +
                WEIGHTS["modality"] * M
        )
        C = min(max(C, 0.0), 1.0)

        # 分类等级
        if C < THRESHOLDS["low"]:
            level = "low"
        elif C < THRESHOLDS["medium"]:
            level = "medium"
        else:
            level = "high"

        num_agents = MAX_AGENTS[level]

        print(f"复杂度评估结果 - C: {C:.2f}, 等级: {level}, 推荐代理数: {num_agents}")
        return C, level, num_agents


# 测试入口
if __name__ == "__main__":
    assessor = TaskComplexityAssessor()

    # 测试 1: 简单任务
    simple_task = "计算 2 + 2 = ?"
    c1, level1, agents1 = assessor.compute_complexity(simple_task)
    print(f"简单任务: C={c1:.2f}, {level1}, {agents1} 代理\n")

    # 测试 2: 中等复杂度任务
    medium_task = "写一篇 300 字的自我介绍，包括兴趣爱好和职业规划。"
    c2, level2, agents2 = assessor.compute_complexity(medium_task)
    print(f"中等任务: C={c2:.2f}, {level2}, {agents2} 代理\n")

    # 测试 3: 高复杂度任务（医疗诊断示例）
    complex_task = """
    患者35岁男性，症状：持续高热（39°C）、干咳、淋巴结肿大、疲劳、近期接触野生动物。
    化验单图像显示：IgM阳性、白细胞升高。诊断可能疾病，并评估概率和风险。
    """
    c3, level3, agents3 = assessor.compute_complexity(complex_task, images=["lab.jpg", "chart.png"], tables=["data.csv"])
    print(f"高复杂度任务: C={c3:.2f}, {level3}, {agents3} 代理")