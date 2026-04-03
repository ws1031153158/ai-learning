# day02_roles/task1_system_prompt.py
from utils import chat

user_question = "我最近压力很大，睡不着觉，怎么办？"

# 没有System Prompt
print("=== 无角色设定 ===")
print(chat(user_question))

# 心理咨询师角色
print("\n=== 心理咨询师 ===")
print(chat(
    user_question,
    system_message="""
你是一名专业的心理咨询师，拥有10年从业经验。
你的沟通风格：温暖、耐心、不评判。
你的回答方式：先共情，再提供实用建议。
每次回答控制在150字以内。
"""
))

# 健身教练角色
print("\n=== 健身教练 ===")
print(chat(
    user_question,
    system_message="""
你是一名专业健身教练，相信运动是解决一切问题的良药。
你的风格：积极、充满能量、略带幽默。
无论什么问题，你都会从运动和身体健康的角度给出建议。
"""
))