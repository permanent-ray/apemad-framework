# test_spacy.py
import spacy

try:
    nlp = spacy.load("en_core_web_sm")
    print("spaCy 模型加载成功！")
    print("版本：", spacy.__version__)
    
    # 测试一句简单句子
    doc = nlp("This is a test sentence for entity recognition.")
    print("实体：", [(ent.text, ent.label_) for ent in doc.ents])
except Exception as e:
    print("加载失败！错误信息：", e)