# utils.py - 整个学习过程都会用到
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv(override=True)

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)

def chat(
    user_message: str,
    system_message: str = None,
    temperature: float = 0.7,
    model: str = "deepseek-chat"
) -> str:
    """最基础的对话函数，后续所有练习都基于此"""
    messages = []

    if system_message:
        messages.append({"role": "system", "content": system_message})

    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )

    return response.choices[0].message.content


def chat_with_history(
    messages: list,
    system_message: str = None,
    temperature: float = 0.7
) -> str:
    """支持多轮对话的函数"""
    full_messages = []

    if system_message:
        full_messages.append({"role": "system", "content": system_message})

    full_messages.extend(messages)

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=full_messages,
        temperature=temperature
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    # 测试连通性
    result = chat("你好，用一句话介绍你自己")
    print(result)
    print("✅ API连接成功！")