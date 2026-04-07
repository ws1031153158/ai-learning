# day03_output/task3_json_mode.py
# DeepSeek 支持 response_format 参数，强制输出JSON
import json
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

def extract_with_json_mode(text: str) -> dict:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": "你是一个信息提取助手，总是以JSON格式输出结果"
            },
            {
                "role": "user",
                "content": f"""
从以下文本中提取所有人名和对应的职位信息：

{text}

输出格式：
{{
    "people": [
        {{"name": "姓名", "title": "职位"}}
    ]
}}
"""
            }
        ],
        response_format={"type": "json_object"},  # 强制JSON输出
        temperature=0.0
    )

    return json.loads(response.choices[0].message.content)


text = "会议由技术总监张伟主持，产品经理李娜介绍了新功能，运营负责人王强提出了推广方案。"
result = extract_with_json_mode(text)
print(json.dumps(result, ensure_ascii=False, indent=2))