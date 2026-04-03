# day02_roles/task2_customer_service.py
import sys
sys.path.append(".")
from utils import chat_with_history

# 这是你第一个"真实场景"练习
# 为一家虚构的智能手表品牌写System Prompt

CUSTOMER_SERVICE_SYSTEM = """
你是"TechWatch"智能手表品牌的客服助手小T。

【品牌信息】
- 品牌：TechWatch
- 主打产品：TW-Pro（售价1299元）、TW-Lite（售价699元）
- 核心卖点：续航14天、心率血氧监测、IP68防水

【你的职责】
- 解答产品咨询
- 处理售后问题
- 引导用户购买（但不要过度推销）

【回答规则】
1. 称呼用户为"您"
2. 回答简洁，不超过100字
3. 不确定的问题说"我帮您转接人工客服确认"
4. 不涉及竞品比较
5. 结尾可以询问是否还有其他问题

【你不能做的事】
- 承诺文档中没有的功能
- 透露内部信息
- 处理退款（引导转人工）
"""

# 模拟多轮对话
def simulate_customer_service():
    history = []
    print("=== TechWatch 客服系统 ===")
    print("输入 'quit' 退出\n")

    while True:
        user_input = input("用户：")
        if user_input.lower() == 'quit':
            break

        history.append({"role": "user", "content": user_input})

        response = chat_with_history(
            history,
            system_message=CUSTOMER_SERVICE_SYSTEM,
            temperature=0.3  # 客服场景用低temperature，保持稳定
        )

        history.append({"role": "assistant", "content": response})
        print(f"小T：{response}\n")

simulate_customer_service()