# agent/day1_function_calling.py
import sys
sys.path.append(".")

import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from practice.agent.tools.stock_tools import (
    get_stock_price,
    get_stock_news,
    get_fund_flow,
    get_financial_indicator,
    calculate_position
)

load_dotenv(override=True)

# ============================================================
# 初始化 DeepSeek 客户端
# ============================================================

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# ============================================================
# 定义工具列表
# 告诉 AI 有哪些工具可以用，每个工具的参数是什么
# ============================================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "获取股票最新价格和涨跌情况，当用户询问股票价格、涨跌时调用",
            "parameters": {
                "type": "object",
                "properties": {
                    "stock_code": {
                        "type": "string",
                        "description": "股票代码，如茅台是600519，比亚迪是002594"
                    }
                },
                "required": ["stock_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_news",
            "description": "获取股票最新新闻和公告，当用户询问股票动态、新闻时调用",
            "parameters": {
                "type": "object",
                "properties": {
                    "stock_code": {
                        "type": "string",
                        "description": "股票代码"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "返回新闻数量，默认5条",
                        "default": 5
                    }
                },
                "required": ["stock_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_fund_flow",
            "description": "获取股票主力资金流向，当用户询问主力动向、资金流入流出时调用",
            "parameters": {
                "type": "object",
                "properties": {
                    "stock_code": {
                        "type": "string",
                        "description": "股票代码"
                    }
                },
                "required": ["stock_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_financial_indicator",
            "description": "获取股票财务指标，当用户询问财务状况、盈利能力、ROE等时调用",
            "parameters": {
                "type": "object",
                "properties": {
                    "stock_code": {
                        "type": "string",
                        "description": "股票代码"
                    }
                },
                "required": ["stock_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_position",
            "description": "根据风险等级计算建议仓位，当用户询问买多少、仓位建议时调用",
            "parameters": {
                "type": "object",
                "properties": {
                    "stock_code": {
                        "type": "string",
                        "description": "股票代码"
                    },
                    "total_assets": {
                        "type": "number",
                        "description": "用户总资产，单位元"
                    },
                    "risk_level": {
                        "type": "string",
                        "description": "风险等级：low低风险/medium中等风险/high高风险",
                        "enum": ["low", "medium", "high"]
                    }
                },
                "required": ["stock_code", "total_assets"]
            }
        }
    }
]

# 工具函数映射表
tool_map = {
    "get_stock_price": get_stock_price,
    "get_stock_news": get_stock_news,
    "get_fund_flow": get_fund_flow,
    "get_financial_indicator": get_financial_indicator,
    "calculate_position": calculate_position
}

# ============================================================
# 核心：执行 Function Calling 循环
# ============================================================

def run_agent(user_question: str) -> str:
    """
    执行 Function Calling 循环：
    1. 发送问题给 AI
    2. AI 决定调用哪个工具
    3. 执行工具，把结果返回给 AI
    4. AI 生成最终回答
    5. 如果 AI 还需要调用工具，重复 2-4
    """
    print(f"\n{'='*50}")
    print(f"用户问题：{user_question}")
    print(f"{'='*50}")

    messages = [
        {
            "role": "system",
            "content": """你是专业的投资分析助手。
你有以下工具可以使用：
- 获取股票实时价格
- 获取股票最新新闻
- 获取主力资金流向
- 获取财务指标
- 计算建议仓位

请根据用户问题主动调用合适的工具获取数据，
然后给出专业、客观的分析。
适当提示投资风险。"""
        },
        {
            "role": "user",
            "content": user_question
        }
    ]

    # 最多循环5次，防止死循环
    for i in range(5):
        print(f"\n[第{i+1}轮对话]")

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools,
            tool_choice="auto"  # 让AI自己决定要不要调用工具
        )

        message = response.choices[0].message

        # 情况1：AI 不需要调用工具，直接给出回答
        if not message.tool_calls:
            print(f"\n✅ AI最终回答：")
            print(message.content)
            return message.content

        # 情况2：AI 需要调用工具
        print(f"🔧 AI决定调用 {len(message.tool_calls)} 个工具：")

        # 把 AI 的决策加入消息历史
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in message.tool_calls
            ]
        })

        # 执行每个工具调用
        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            print(f"  → 调用：{func_name}({func_args})")

            # 执行工具函数
            if func_name in tool_map:
                result = tool_map[func_name](**func_args)
            else:
                result = f"未知工具：{func_name}"

            print(f"  ← 结果：{result[:100]}...")

            # 把工具结果加入消息历史
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

    return "达到最大循环次数，请重试"


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":

    # 测试1：单工具调用
    run_agent("茅台今天股价多少？")

    # 测试2：多工具调用
    run_agent("帮我分析一下茅台，包括最新价格和近期新闻")

    # 测试3：复杂分析
    run_agent("我有50万，想买茅台，主力资金怎么样，建议买多少？")

    # 测试4：AI自主决策
    run_agent("比亚迪值得投资吗？")