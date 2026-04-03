# day05_fewshot/task2_classifier.py
from utils import chat
import json

def classify_review(review: str) -> dict:
    prompt = f"""
对用户评论进行多维度分类。

示例1：
评论："界面很好看，操作也流畅，就是电池不太耐用"
分类：{{"sentiment": "mixed", "aspects": ["ui_positive", "performance_positive", "battery_negative"], "recommend": true}}

示例2：
评论："完全无法使用，闪退严重，客服也不理人，差评！"
分类：{{"sentiment": "negative", "aspects": ["stability_negative", "service_negative"], "recommend": false}}

示例3：
评论："性价比超高，功能够用，物流也快"
分类：{{"sentiment": "positive", "aspects": ["price_positive", "feature_positive", "delivery_positive"], "recommend": true}}

现在请分类：
评论："{review}"
分类："""

    result = chat(prompt, temperature=0.0)

    try:
        return json.loads(result)
    except:
        return {"raw": result}


# 测试
reviews = [
    "音质很好，但是连接蓝牙总是断开，希望能修复",
    "买了两年了还在用，质量杠杠的",
    "包装破损，产品也有划痕，退货流程还很麻烦",
]

for review in reviews:
    print(f"评论：{review}")
    print(f"分类：{classify_review(review)}")
    print()