# day06_engineering/task3_error_handling.py
# 生产级代码必须有完善的错误处理
import time
import sys
sys.path.append('.')
from openai import OpenAI, RateLimitError, APITimeoutError, APIError
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)

def robust_chat(
    user_message: str,
    system_message: str = None,
    max_retries: int = 3,
    timeout: int = 30
) -> str:
    """带错误处理和重试的生产级chat函数"""

    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.7,
                timeout=timeout
            )
            return response.choices[0].message.content

        except RateLimitError:
            # 触发限流，等待后重试
            wait_time = 2 ** attempt  # 指数退避：1s, 2s, 4s
            print(f"触发限流，{wait_time}秒后重试（第{attempt+1}次）...")
            time.sleep(wait_time)

        except APITimeoutError:
            print(f"请求超时（第{attempt+1}次），正在重试...")
            if attempt == max_retries - 1:
                return "请求超时，请稍后再试"

        except APIError as e:
            print(f"API错误：{e}")
            if attempt == max_retries - 1:
                return f"服务暂时不可用，请稍后再试"

        except Exception as e:
            print(f"未知错误：{e}")
            return "发生未知错误，请联系管理员"

    return "多次重试后仍然失败，请稍后再试"


# 测试
result = robust_chat("你好，请用一句话介绍自己")
print(result)