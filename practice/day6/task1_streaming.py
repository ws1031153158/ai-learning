# day06_engineering/task1_streaming.py
# 流式输出 = 像ChatGPT那样一个字一个字地出现
import sys
sys.path.append('.')
from openai import OpenAI
from dotenv import load_dotenv
import os
import time

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)

def stream_chat(user_message: str, system_message: str = None):
    """流式输出版本的chat函数"""
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})

    stream = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=True  # 关键参数
    )

    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)  # 实时打印
            full_response += content

    print()  # 换行
    return full_response


# 测试流式输出
print("=== 流式输出效果 ===")
stream_chat("用300字介绍一下RAG技术的原理和应用场景")