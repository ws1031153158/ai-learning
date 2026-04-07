# day04_cot/task1_cot_basic.py
from utils import chat

# 一个需要推理的问题
question = """
一家公司有员工120人。
其中40%是技术部门，技术部门中有25%是女性。
非技术部门中有60%是女性。
请问公司女性员工总共有多少人？
"""

# 不用CoT
print("=== 直接回答 ===")
print(chat(f"{question}\n直接给出数字答案。", temperature=0.0))

# 使用CoT
print("\n=== 使用思维链 ===")
print(chat(f"""
{question}

请一步一步地思考，把每个计算步骤都写出来，最后给出答案。
""", temperature=0.0))