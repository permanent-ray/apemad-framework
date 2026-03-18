# test_imports.py
import sys

print("Python 版本:", sys.version)

try:
    import pyautogen
    print("pyautogen 导入成功！")
    # 安全检查版本（新版没有 __version__，我们跳过或用其他方式）
    # 如果你非要版本，可以用 pip show（但这里先注释掉）
    # import subprocess
    # version = subprocess.check_output(["pip", "show", "pyautogen"]).decode().split("Version: ")[1].split("\n")[0]
    # print("pyautogen 版本:", version.strip())
except ImportError as e:
    print("pyautogen 导入失败:", e)

try:
    import autogen
    print("旧版 autogen 导入成功（如果存在）")
except ImportError:
    print("旧版 autogen 未找到（正常，新版已改为 pyautogen）")

try:
    import transformers
    print("Transformers 导入成功，版本:", transformers.__version__)
except ImportError as e:
    print("Transformers 导入失败:", e)

try:
    import torch
    print("Torch 导入成功，版本:", torch.__version__)
    print("CUDA 是否可用:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("当前 GPU:", torch.cuda.get_device_name(0))
except ImportError as e:
    print("Torch 导入失败:", e)

try:
    import sentence_transformers
    print("Sentence-Transformers 导入成功")
except ImportError as e:
    print("Sentence-Transformers 导入失败:", e)

try:
    import faiss
    print("FAISS 导入成功")
except ImportError as e:
    print("FAISS 导入失败:", e)

try:
    import wandb
    print("WandB 导入成功")
except ImportError as e:
    print("WandB 导入失败:", e)

try:
    import stable_baselines3
    print("Stable-Baselines3 导入成功")
except ImportError as e:
    print("Stable-Baselines3 导入失败:", e)

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    print("spaCy + en_core_web_sm 加载成功！")
except Exception as e:
    print("spaCy 加载失败:", e)

print("\n如果上面大部分成功（尤其是 pyautogen），则环境配置完成！")