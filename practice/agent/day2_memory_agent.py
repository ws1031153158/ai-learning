# agent/day2_memory_agent.py
import sys
sys.path.append(".")

import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
from practice.agent.tools.stock_tools import (
    get_stock_price,
    get_stock_news,
    get_fund_flow,
    get_financial_indicator,
    calculate_position
)

load_dotenv(override=True)

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# ============================================================
# 工具定义（复用Day1的）
# ============================================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "获取股票最新价格和涨跌情况",
            "parameters": {
                "type": "object",
                "properties": {
                    "stock_code": {
                        "type": "string",
                        "description": "股票代码，如茅台600519，比亚迪002594，宁德时代300750"
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
            "description": "获取股票最新新闻和公告",
            "parameters": {
                "type": "object",
                "properties": {
                    "stock_code": {
                        "type": "string",
                        "description": "股票代码"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "返回新闻数量，默认5",
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
            "description": "获取股票主力资金流向",
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
            "description": "获取股票财务指标，包括ROE、净利润增长率等",
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
            "description": "根据风险等级计算建议仓位",
            "parameters": {
                "type": "object",
                "properties": {
                    "stock_code": {
                        "type": "string",
                        "description": "股票代码"
                    },
                    "total_assets": {
                        "type": "number",
                        "description": "总资产，单位元"
                    },
                    "risk_level": {
                        "type": "string",
                        "description": "风险等级 low/medium/high",
                        "enum": ["low", "medium", "high"]
                    }
                },
                "required": ["stock_code", "total_assets"]
            }
        }
    }
]

tool_map = {
    "get_stock_price": get_stock_price,
    "get_stock_news": get_stock_news,
    "get_fund_flow": get_fund_flow,
    "get_financial_indicator": get_financial_indicator,
    "calculate_position": calculate_position
}


# ============================================================
# 记忆管理器
# ============================================================

class MemoryManager:
    """
    管理对话历史和用户画像
    
    两种记忆：
    1. 对话历史：完整的消息记录
    2. 用户画像：提取的关键信息（持仓、风险偏好等）
    """

    def __init__(self, max_history: int = 20):
        self.max_history = max_history

        # 系统提示词（固定不变）
        self.system_prompt = f"""你是专业的投资分析助手，今天是{datetime.now().strftime("%Y年%m月%d日")}。

你的能力：
- 获取股票实时价格和涨跌情况
- 获取最新财经新闻
- 分析主力资金流向
- 查看核心财务指标
- 计算建议仓位

工作原则：
1. 主动调用工具获取实时数据，不要凭记忆回答价格类问题
2. 记住用户提到的持仓、风险偏好等个人信息
3. 回答专业客观，适当提示风险
4. 不构成投资建议"""

        # 对话历史（不含system）
        self.history = []

        # 用户画像（从对话中提取）
        self.user_profile = {
            "mentioned_stocks": [],   # 提到过的股票
            "total_assets": None,     # 总资产
            "risk_level": None,       # 风险偏好
            "holdings": []            # 持仓
        }

    def add_user_message(self, content: str):
        """添加用户消息"""
        self.history.append({
            "role": "user",
            "content": content
        })
        self._update_profile(content)

    def add_assistant_message(self, content: str, tool_calls=None):
        """添加AI消息"""
        msg = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.history.append(msg)

    def add_tool_result(self, tool_call_id: str, content: str):
        """添加工具结果"""
        self.history.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content
        })

    def get_messages(self) -> list:
        """
        获取完整消息列表
        超出长度时保留system + 最近N条
        """
        messages = [{"role": "system", "content": self.system_prompt}]

        # 如果历史太长，只保留最近的
        if len(self.history) > self.max_history:
            # 保留最近的消息
            recent = self.history[-self.max_history:]
            # 加一条摘要说明
            messages.append({
                "role": "system",
                "content": f"[对话摘要] 用户画像：{json.dumps(self.user_profile, ensure_ascii=False)}"
            })
            messages.extend(recent)
        else:
            messages.extend(self.history)

        return messages

    def _update_profile(self, user_message: str):
        """从用户消息中提取关键信息"""
        # 提取资产信息
        import re
        asset_match = re.search(r'(\d+)万', user_message)
        if asset_match:
            self.user_profile["total_assets"] = (
                float(asset_match.group(1)) * 10000
            )

        # 提取风险偏好
        if any(w in user_message for w in ["保守", "稳健", "低风险"]):
            self.user_profile["risk_level"] = "low"
        elif any(w in user_message for w in ["激进", "高风险", "冒险"]):
            self.user_profile["risk_level"] = "high"
        elif any(w in user_message for w in ["中等", "适中"]):
            self.user_profile["risk_level"] = "medium"

    def get_profile_summary(self) -> str:
        """获取用户画像摘要"""
        profile = self.user_profile
        parts = []
        if profile["total_assets"]:
            parts.append(f"总资产{profile['total_assets']/10000:.0f}万")
        if profile["risk_level"]:
            parts.append(f"风险偏好{profile['risk_level']}")
        if profile["mentioned_stocks"]:
            parts.append(f"关注股票{profile['mentioned_stocks']}")
        return "、".join(parts) if parts else "暂无"

    def clear(self):
        """清空对话历史"""
        self.history = []
        self.user_profile = {
            "mentioned_stocks": [],
            "total_assets": None,
            "risk_level": None,
            "holdings": []
        }


# ============================================================
# 带记忆的 Agent
# ============================================================

class FinanceAgent:
    """
    带记忆的理财分析 Agent
    """

    def __init__(self):
        self.memory = MemoryManager(max_history=20)
        print("✅ 理财分析助手已启动")
        print("   输入 'quit' 退出")
        print("   输入 'clear' 清空对话")
        print("   输入 'profile' 查看用户画像")
        print("-" * 50)

    def _execute_tools(self, tool_calls) -> list:
        """执行工具调用，返回结果列表"""
        results = []
        for tool_call in tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            print(f"  🔧 调用工具：{func_name}{func_args}")

            if func_name in tool_map:
                result = tool_map[func_name](**func_args)
            else:
                result = f"未知工具：{func_name}"

            print(f"  ✅ 工具返回：{result[:80]}...")
            results.append((tool_call.id, result))

        return results

    def chat(self, user_input: str) -> str:
        """
        处理用户输入，返回AI回复
        """
        # 添加用户消息到记忆
        self.memory.add_user_message(user_input)

        # 最多循环5次
        for i in range(5):
            messages = self.memory.get_messages()

            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )

            message = response.choices[0].message

            # AI 不需要调用工具，直接回答
            if not message.tool_calls:
                self.memory.add_assistant_message(message.content)
                return message.content

            # AI 需要调用工具
            print(f"\n[AI决定调用 {len(message.tool_calls)} 个工具]")

            # 保存AI的工具调用决策
            tool_calls_data = [
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
            self.memory.add_assistant_message(
                message.content or "",
                tool_calls=tool_calls_data
            )

            # 执行工具
            tool_results = self._execute_tools(message.tool_calls)

            # 把工具结果加入记忆
            for tool_call_id, result in tool_results:
                self.memory.add_tool_result(tool_call_id, result)

        return "处理超时，请重试"

    def run(self):
        """
        启动交互式对话循环
        """
        while True:
            try:
                user_input = input("\n你：").strip()

                if not user_input:
                    continue

                if user_input.lower() == 'quit':
                    print("再见！")
                    break

                if user_input.lower() == 'clear':
                    self.memory.clear()
                    print("✅ 对话已清空")
                    continue

                if user_input.lower() == 'profile':
                    print(f"用户画像：{self.memory.get_profile_summary()}")
                    continue

                print("\nAI分析中...")
                response = self.chat(user_input)
                print(f"\nAI：{response}")

            except KeyboardInterrupt:
                print("\n再见！")
                break
            except Exception as e:
                print(f"错误：{e}")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    agent = FinanceAgent()
    agent.run()