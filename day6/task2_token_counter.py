# day06_engineering/task2_token_counter.py
# 实际开发中必须监控Token用量和费用
import sys
sys.path.append('.')
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)

def chat_with_stats(user_message: str, system_message: str = None) -> dict:
    """返回回答内容和Token统计"""
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.7
    )

    # DeepSeek 计费参考（实时价格以官网为准）
    INPUT_PRICE_PER_1M = 0.27   # 元/百万token
    OUTPUT_PRICE_PER_1M = 1.10  # 元/百万token

    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    total_tokens = response.usage.total_tokens

    cost = (input_tokens * INPUT_PRICE_PER_1M +
            output_tokens * OUTPUT_PRICE_PER_1M) / 1_000_000

    return {
        "content": response.choices[0].message.content,
        "stats": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_cny": f"¥{cost:.6f}"
        }
    }


# 测试
result = chat_with_stats(
    "用200字解释什么是向量数据库",
    system_message="你是一个技术讲师，用通俗易懂的语言解释技术概念"
)

print("回答：")
print(result["content"])
print("\nToken统计：")
for k, v in result["stats"].items():
    print(f"  {k}: {v}")