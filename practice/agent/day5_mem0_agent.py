# agent/day5_mem0_agent.py
import sys
sys.path.append(".")

import os
import json
from datetime import datetime
from openai import OpenAI
from mem0 import Memory
from dotenv import load_dotenv

load_dotenv(override=True)

# ============================================================
# 初始化
# ============================================================

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# Mem0 配置
# 使用本地存储，不需要额外服务
config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "deepseek-chat",
            "api_key": os.getenv("DEEPSEEK_API_KEY"),
            "openai_base_url": "https://api.deepseek.com"
        }
    },
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "BAAI/bge-small-zh-v1.5"
        }
    },
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "finance_agent_memory",
            "path": "./agent/memory_store"
        }
    }
}

memory = Memory.from_config(config)

# 工具定义
from practice.agent.tools.stock_tools import (
    get_stock_price,
    get_stock_news,
    get_fund_flow,
    get_financial_indicator,
    calculate_position
)

TOOLS = [
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
                        "description": "股票代码，如茅台600519"
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
            "description": "获取股票最新新闻",
            "parameters": {
                "type": "object",
                "properties": {
                    "stock_code": {"type": "string"},
                    "limit": {"type": "integer", "default": 5}
                },
                "required": ["stock_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_fund_flow",
            "description": "获取主力资金流向",
            "parameters": {
                "type": "object",
                "properties": {
                    "stock_code": {"type": "string"}
                },
                "required": ["stock_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_financial_indicator",
            "description": "获取财务指标",
            "parameters": {
                "type": "object",
                "properties": {
                    "stock_code": {"type": "string"}
                },
                "required": ["stock_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_position",
            "description": "计算建议仓位",
            "parameters": {
                "type": "object",
                "properties": {
                    "stock_code": {"type": "string"},
                    "total_assets": {"type": "number"},
                    "risk_level": {
                        "type": "string",
                        "enum": ["low", "medium", "high"]
                    }
                },
                "required": ["stock_code", "total_assets"]
            }
        }
    }
]

TOOL_MAP = {
    "get_stock_price": get_stock_price,
    "get_stock_news": get_stock_news,
    "get_fund_flow": get_fund_flow,
    "get_financial_indicator": get_financial_indicator,
    "calculate_position": calculate_position
}


# ============================================================
# 带长期记忆的 Agent
# ============================================================

class Mem0FinanceAgent:
    """
    带长期记忆的理财分析 Agent

    记忆分两层：
    1. 短期：当前对话的消息历史
    2. 长期：Mem0 自动提取并持久化的用户信息
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.current_messages = []  # 当前会话消息
        print(f"✅ Agent 初始化完成，用户：{user_id}")

        # 加载已有记忆
        existing = self._get_memories()
        if existing:
            print(f"📚 找到 {len(existing)} 条历史记忆：")
            for m in existing[:3]:  # 只显示前3条
                print(f"   - {m['memory']}")
        else:
            print("📭 暂无历史记忆")

    def _get_memories(self) -> list:
        """获取该用户的所有长期记忆"""
        try:
            results = memory.get_all(user_id=self.user_id)
            return results.get("results", [])
        except Exception:
            return []

    def _search_memories(self, query: str) -> str:
        """搜索相关记忆"""
        try:
            results = memory.search(
                query=query,
                user_id=self.user_id,
                limit=5
            )
            memories = results.get("results", [])
            if not memories:
                return ""
            return "\n".join([
                f"- {m['memory']}"
                for m in memories
            ])
        except Exception:
            return ""

    def _save_memories(self, messages: list):
        """
        把对话内容交给 Mem0 自动提取并保存记忆
        Mem0 会自动识别：
        - 用户资产信息
        - 风险偏好
        - 关注的股票
        - 历史分析结论
        """
        try:
            memory.add(
                messages=messages,
                user_id=self.user_id
            )
        except Exception as e:
            print(f"记忆保存失败：{e}")

    def chat(self, user_input: str) -> str:
        """处理用户输入"""

        # 1. 搜索相关历史记忆
        relevant_memories = self._search_memories(user_input)

        # 2. 构建系统提示（包含历史记忆）
        system_content = f"""你是专业投资分析助手。
今天是{datetime.now().strftime("%Y年%m月%d日")}。

主动调用工具获取实时数据，不要用旧数据回答价格类问题。
回答专业客观，适当提示投资风险。"""

        if relevant_memories:
            system_content += f"""

关于该用户你记得以下信息：
{relevant_memories}

请在回答时参考这些信息，提供个性化建议。"""

        # 3. 构建消息列表
        messages = [
            {"role": "system", "content": system_content},
            *self.current_messages,
            {"role": "user", "content": user_input}
        ]

        # 4. Function Calling 循环
        final_response = ""
        tool_call_messages = []

        for _ in range(5):
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto"
            )
            message = response.choices[0].message

            if not message.tool_calls:
                final_response = message.content
                break

            # 处理工具调用
            print(f"\n[调用 {len(message.tool_calls)} 个工具]")

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

            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": tool_calls_data
            })

            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                print(f"  → {func_name}({func_args})")

                result = (
                    TOOL_MAP[func_name](**func_args)
                    if func_name in TOOL_MAP
                    else f"未知工具：{func_name}"
                )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

        # 5. 更新当前会话历史
        self.current_messages.append({
            "role": "user",
            "content": user_input
        })
        self.current_messages.append({
            "role": "assistant",
            "content": final_response
        })

        # 6. 保存记忆（异步提取关键信息）
        self._save_memories([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": final_response}
        ])

        return final_response

    def show_memories(self):
        """显示所有记忆"""
        memories = self._get_memories()
        if not memories:
            print("暂无记忆")
            return
        print(f"\n📚 {self.user_id} 的所有记忆（{len(memories)}条）：")
        for i, m in enumerate(memories, 1):
            print(f"  {i}. {m['memory']}")

    def clear_memories(self):
        """清空所有记忆"""
        try:
            memory.delete_all(user_id=self.user_id)
            self.current_messages = []
            print("✅ 记忆已清空")
        except Exception as e:
            print(f"清空失败：{e}")

    def run(self):
        """启动交互式对话"""
        print("\n" + "=" * 50)
        print("理财AI助手（长期记忆版）")
        print("命令：quit退出 | memory查看记忆 | clear清空记忆")
        print("=" * 50)

        while True:
            try:
                user_input = input("\n你：").strip()
                if not user_input:
                    continue
                if user_input == "quit":
                    print("再见！记忆已保存，下次继续。")
                    break
                if user_input == "memory":
                    self.show_memories()
                    continue
                if user_input == "clear":
                    self.clear_memories()
                    continue

                print("AI思考中...")
                response = self.chat(user_input)
                print(f"\nAI：{response}")

            except KeyboardInterrupt:
                print("\n再见！")
                break


# ============================================================
# 主程序：测试长期记忆
# ============================================================

if __name__ == "__main__":

    # ── 第一轮对话：告诉AI你的信息 ──────────────────────
    print("=" * 50)
    print("第一轮对话：建立用户画像")
    print("=" * 50)

    agent = Mem0FinanceAgent(user_id="test_user_001")

    # 模拟用户告知个人信息
    test_conversations = [
        "我有50万资产，风险偏好中等，主要关注白酒和新能源板块",
        "帮我分析一下茅台，我最近在考虑买入",
        "我目前持有比亚迪，成本价是80元",
    ]

    for msg in test_conversations:
        print(f"\n你：{msg}")
        response = agent.chat(msg)
        print(f"AI：{response[:200]}...")

    print("\n\n查看提取的记忆：")
    agent.show_memories()

    # ── 第二轮对话：模拟重新打开App ─────────────────────
    print("\n\n" + "=" * 50)
    print("第二轮对话：模拟重新打开App（新会话）")
    print("=" * 50)

    # 创建新的 Agent 实例（模拟重启）
    agent2 = Mem0FinanceAgent(user_id="test_user_001")

    # 测试AI是否还记得
    test_questions = [
        "根据我的情况，现在适合加仓茅台吗？",
        "我持有的比亚迪现在盈亏情况怎么样？",
    ]

    for q in test_questions:
        print(f"\n你：{q}")
        response = agent2.chat(q)
        print(f"AI：{response[:300]}...")

    # ── 交互模式 ─────────────────────────────────────────
    print("\n\n进入交互模式，输入quit退出")
    agent2.run()