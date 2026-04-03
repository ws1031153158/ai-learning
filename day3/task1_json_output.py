# day03_output/task1_json_output.py
import sys
sys.path.append('.')
import json
from utils import chat

# 场景：从用户的自然语言中提取结构化信息
def extract_task_info(user_input: str) -> dict:
    prompt = f"""
从以下用户输入中提取任务信息，以JSON格式输出。

用户输入：{user_input}

输出格式（严格按照此JSON格式，不要输出任何其他内容）：
{{
    "title": "任务标题",
    "deadline": "截止日期（如没有则为null）",
    "priority": "优先级（high/medium/low）",
    "tags": ["标签1", "标签2"]
}}

只输出JSON，不要有任何解释或markdown代码块。
"""

    result = chat(prompt, temperature=0.0)

    try:
        return json.loads(result)
    except json.JSONDecodeError:
        print(f"解析失败，原始输出：{result}")
        return None


# 测试
test_inputs = [
    "明天下午3点之前要提交季度报告，很紧急",
    "有空的时候整理一下桌面文件",
    "下周五前完成用户调研问卷，需要发给产品和运营团队",
]

for text in test_inputs:
    print(f"输入：{text}")
    result = extract_task_info(text)
    print(f"输出：{json.dumps(result, ensure_ascii=False, indent=2)}")
    print()