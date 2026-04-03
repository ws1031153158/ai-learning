# day04_cot/task3_decision_assistant.py
import sys
sys.path.append('.')
from utils import chat

def analyze_decision(scenario: str) -> str:
    """
    用CoT帮助用户做复杂决策
    """
    system = """
你是一个理性的决策分析助手。
对于每个决策问题，你会：
1. 先识别核心问题
2. 列出所有相关因素
3. 分析每个选项的利弊
4. 给出明确的建议和理由
保持客观，不带感情色彩。
"""

    prompt = f"""
请帮我分析以下决策场景：

{scenario}

请按照以下结构分析：

## 核心问题
（用一句话定义真正需要决策的是什么）

## 关键因素
（列出影响决策的3-5个最重要因素）

## 选项分析
（对每个选项分析优势和劣势）

## 我的建议
（给出明确建议，并说明最重要的理由）
"""

    return chat(prompt, system_message=system, temperature=0.3)


# 测试
scenario = """
我是一名有3年经验的Android开发，现在面临两个选择：
A：留在现在的公司，薪资从20k涨到23k，工作稳定但技术栈老旧
B：跳槽到一家AI创业公司，薪资25k，需要转型做AI应用开发，有期权但有风险

我已婚，有房贷，孩子1岁。
"""

print(analyze_decision(scenario))