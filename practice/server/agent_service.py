# server/agent_service.py
import sys
sys.path.append(".")

import json
import asyncio
import os
from datetime import datetime
from openai import OpenAI
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

from practice.agent.tools.stock_tools import (
    get_stock_price,
    get_stock_news,
    get_fund_flow,
    get_financial_indicator,
    calculate_position
)
from practice.server.config import *


# ============================================================
# CrewAI 工具定义（英文名）
# ============================================================

@tool("get_price_tool")
def tool_get_price(stock_code: str) -> str:
    """获取股票最新价格和涨跌幅，输入股票代码如600519"""
    return get_stock_price(stock_code)

@tool("get_news_tool")
def tool_get_news(stock_code: str) -> str:
    """获取股票最新新闻和公告，输入股票代码如600519"""
    return get_stock_news(stock_code, limit=5)

@tool("get_fund_flow_tool")
def tool_get_fund_flow(stock_code: str) -> str:
    """获取股票主力资金流向，输入股票代码如600519"""
    return get_fund_flow(stock_code)

@tool("get_financial_tool")
def tool_get_financial(stock_code: str) -> str:
    """获取股票核心财务指标，输入股票代码如600519"""
    return get_financial_indicator(stock_code)

@tool("calculate_position_tool")
def tool_calculate_position(
    stock_code: str,
    total_assets: float,
    risk_level: str
) -> str:
    """计算建议仓位，需要股票代码、总资产（元）、风险等级low/medium/high"""
    return calculate_position(stock_code, total_assets, risk_level)


# ============================================================
# OpenAI 工具定义（用于对话Agent）
# ============================================================

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
# 会话管理（多用户对话隔离）
# ============================================================

class SessionManager:
    """
    管理多个用户的对话历史
    每个 session_id 对应一个独立的对话
    """

    def __init__(self):
        # {session_id: [messages]}
        self._sessions: dict[str, list] = {}
        # {session_id: user_profile}
        self._profiles: dict[str, dict] = {}

    def get_or_create(self, session_id: str) -> list:
        if session_id not in self._sessions:
            self._sessions[session_id] = []
            self._profiles[session_id] = {
                "total_assets": None,
                "risk_level": None,
                "mentioned_stocks": []
            }
        return self._sessions[session_id]

    def add_message(self, session_id: str, message: dict):
        history = self.get_or_create(session_id)
        history.append(message)
        # 最多保留30条
        if len(history) > 30:
            self._sessions[session_id] = history[-30:]

    def get_messages(self, session_id: str) -> list:
        return self.get_or_create(session_id)

    def clear(self, session_id: str):
        self._sessions[session_id] = []
        self._profiles[session_id] = {
            "total_assets": None,
            "risk_level": None,
            "mentioned_stocks": []
        }

    def get_profile(self, session_id: str) -> dict:
        self.get_or_create(session_id)
        return self._profiles[session_id]

    def update_profile(self, session_id: str, message: str):
        import re
        profile = self.get_profile(session_id)
        asset_match = re.search(r'(\d+)万', message)
        if asset_match:
            profile["total_assets"] = float(
                asset_match.group(1)
            ) * 10000
        if any(w in message for w in ["保守", "稳健", "低风险"]):
            profile["risk_level"] = "low"
        elif any(w in message for w in ["激进", "高风险"]):
            profile["risk_level"] = "high"
        elif any(w in message for w in ["中等", "适中"]):
            profile["risk_level"] = "medium"


# 全局会话管理器
session_manager = SessionManager()

# OpenAI 客户端
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)


# ============================================================
# 对话 Agent（带记忆）
# ============================================================

async def chat_with_agent(
    session_id: str,
    user_message: str
) -> str:
    """
    带记忆的对话Agent
    在线程池中执行，不阻塞事件循环
    """

    def _sync_chat():
        # 更新用户画像
        session_manager.update_profile(session_id, user_message)

        # 添加用户消息
        session_manager.add_message(session_id, {
            "role": "user",
            "content": user_message
        })

        profile = session_manager.get_profile(session_id)
        system_prompt = f"""你是专业投资分析助手，今天是{datetime.now().strftime("%Y年%m月%d日")}。
主动调用工具获取实时数据，不要用训练数据里的旧价格回答。
回答专业客观，适当提示风险。
用户画像：{json.dumps(profile, ensure_ascii=False)}"""

        messages = [
            {"role": "system", "content": system_prompt}
        ] + session_manager.get_messages(session_id)

        for _ in range(5):
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto"
            )
            message = response.choices[0].message

            if not message.tool_calls:
                session_manager.add_message(session_id, {
                    "role": "assistant",
                    "content": message.content
                })
                return message.content

            # 处理工具调用
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

            session_manager.add_message(session_id, {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": tool_calls_data
            })

            messages = [
                {"role": "system", "content": system_prompt}
            ] + session_manager.get_messages(session_id)

            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                if func_name in TOOL_MAP:
                    result = TOOL_MAP[func_name](**func_args)
                else:
                    result = f"未知工具：{func_name}"

                session_manager.add_message(session_id, {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

                messages = [
                    {"role": "system", "content": system_prompt}
                ] + session_manager.get_messages(session_id)

        return "处理超时，请重试"

    # 在线程池中执行同步代码，不阻塞FastAPI
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _sync_chat)


# ============================================================
# 多Agent综合分析（CrewAI）
# ============================================================

async def run_crew_analysis(
    stock_code: str,
    total_assets: float = None
) -> str:
    """
    多Agent综合分析
    在线程池中执行，不阻塞事件循环
    """

    def _sync_crew():
        data_collector = Agent(
            role="股票数据收集师",
            goal=f"收集 {stock_code} 的完整数据",
            backstory="专业金融数据分析师，负责收集整理股票数据。",
            tools=[
                tool_get_price,
                tool_get_news,
                tool_get_fund_flow
            ],
            verbose=False,
            allow_delegation=False,
            max_iter=3
        )

        technical_analyst = Agent(
            role="技术分析师",
            goal=f"分析 {stock_code} 技术面走势",
            backstory="资深技术分析师，擅长K线和量价分析。",
            tools=[tool_get_price, tool_get_fund_flow],
            verbose=False,
            allow_delegation=False,
            max_iter=3
        )

        fundamental_analyst = Agent(
            role="基本面分析师",
            goal=f"分析 {stock_code} 财务状况",
            backstory="专业基本面分析师，擅长财务报表分析。",
            tools=[tool_get_financial, tool_get_news],
            verbose=False,
            allow_delegation=False,
            max_iter=3
        )

        risk_assessor = Agent(
            role="风险评估师",
            goal=f"评估投资 {stock_code} 的风险",
            backstory="风险管理专家，保护投资者利益。",
            tools=[tool_get_news, tool_get_financial],
            verbose=False,
            allow_delegation=False,
            max_iter=3
        )

        chief_analyst = Agent(
            role="首席投资分析师",
            goal=f"整合所有分析，输出 {stock_code} 投资报告",
            backstory="20年经验的首席分析师，输出专业投资报告。",
            tools=[tool_calculate_position] if total_assets else [],
            verbose=False,
            allow_delegation=False,
            max_iter=3
        )

        task_collect = Task(
            description=f"""收集 {stock_code} 的数据：
1. 最新价格和涨跌幅
2. 最新5条新闻
3. 近3日资金流向
今天：{datetime.now().strftime("%Y-%m-%d")}""",
            expected_output="结构化数据报告",
            agent=data_collector
        )

        task_technical = Task(
            description=f"对 {stock_code} 进行技术面分析，给出评级（强势/中性/弱势）",
            expected_output="技术面分析报告",
            agent=technical_analyst,
            context=[task_collect]
        )

        task_fundamental = Task(
            description=f"对 {stock_code} 进行基本面分析，给出评级（优质/良好/一般/较差）",
            expected_output="基本面分析报告",
            agent=fundamental_analyst,
            context=[task_collect]
        )

        task_risk = Task(
            description=f"评估投资 {stock_code} 的风险，给出综合风险等级（高/中/低）",
            expected_output="风险评估报告",
            agent=risk_assessor,
            context=[task_collect, task_fundamental]
        )

        assets_info = (
            f"用户总资产：{total_assets}元"
            if total_assets
            else "用户未提供资产信息"
        )

        task_report = Task(
            description=f"""整合所有分析，输出 {stock_code} 完整报告。
{assets_info}

格式：
## 📊 基本信息
## 📈 技术面分析
## 💼 基本面分析
## ⚠️ 风险评估
## 💡 综合建议
{"## 💰 仓位建议" if total_assets else ""}
## 📝 免责声明""",
            expected_output="完整投资分析报告",
            agent=chief_analyst,
            context=[
                task_collect,
                task_technical,
                task_fundamental,
                task_risk
            ]
        )

        crew = Crew(
            agents=[
                data_collector,
                technical_analyst,
                fundamental_analyst,
                risk_assessor,
                chief_analyst
            ],
            tasks=[
                task_collect,
                task_technical,
                task_fundamental,
                task_risk,
                task_report
            ],
            process=Process.sequential,
            verbose=False
        )

        result = crew.kickoff()
        return str(result)

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _sync_crew)