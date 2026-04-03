# day03_output/task2_structured_report.py
from utils import chat

def analyze_feedback(feedback: str) -> str:
    prompt = f"""
分析以下用户反馈，按照指定格式输出分析报告。

用户反馈：
{feedback}

请严格按照以下格式输出，每个标签单独一行：

[情感倾向] 正面/负面/中性
[情感强度] 强烈/一般/轻微
[问题类型] 产品功能/用户体验/客服服务/价格/物流/其他
[核心诉求] 用一句话描述用户最想要的是什么
[建议优先级] P0紧急/P1高/P2中/P3低
[处理建议] 具体的处理建议（50字以内）
"""

    return chat(prompt, temperature=0.0)


# 测试
feedbacks = [
    "这个App太难用了！每次打开都要加载很久，而且经常闪退，我已经卸载了",
    "总体还不错，就是价格稍微贵了点，希望能出个学生优惠",
    "客服小姐姐态度超好，帮我解决了问题，给五星好评！",
]

for fb in feedbacks:
    print(f"原始反馈：{fb}")
    print(analyze_feedback(fb))
    print("=" * 50)