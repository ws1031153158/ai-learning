# app/services/agent_service.py
import sys
sys.path.append(".")

import json
import asyncio
import time
from datetime import datetime
from openai import OpenAI
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

from practice.agent.tools.stock_tools import (
    get_stock_price,
    get_stock_news,
    get_fund_flow,
    get_financial_indicator,
    calculate_position,
    get_fund_info,
    get_fund_performance,
    get_fund_manager,
    get_bond_info,
    get_bond_detail
)
from app.config import *
from app.services.memory_service import MemoryService


# ── CrewAI 工具 ───────────────────────────────────────

@tool("get_price_tool")
def tool_get_price(stock_code: str) -> str:
    """获取股票最新价格和涨跌幅，输入股票代码如600519"""
    return get_stock_price(stock_code)

@tool("get_news_tool")
def tool_get_news(stock_code: str) -> str:
    """获取股票最新新闻，输入股票代码如600519"""
    return get_stock_news(stock_code, limit=5)

@tool("get_fund_flow_tool")
def tool_get_fund_flow(stock_code: str) -> str:
    """获取主力资金流向，输入股票代码如600519"""
    return get_fund_flow(stock_code)

@tool("get_financial_tool")
def tool_get_financial(stock_code: str) -> str:
    """获取核心财务指标，输入股票代码如600519"""
    return get_financial_indicator(stock_code)

@tool("calculate_position_tool")
def tool_calculate_position(
    stock_code: str,
    total_assets: float,
    risk_level: str
) -> str:
    """计算建议仓位，需要股票代码、总资产（元）、风险等级"""
    return calculate_position(stock_code, total_assets, risk_level)


# ── Function Calling 工具 ─────────────────────────────

FC_TOOLS = [
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


# ── 会话管理 ──────────────────────────────────────────

class SessionManager:

    def __init__(self):
        self.sessions: dict = {}

    def get_or_create(self, session_id: str) -> list:
        now = time.time()
        expired = [
            sid for sid, data in self.sessions.items()
            if now - data["last_active"] > SESSION_TIMEOUT
        ]
        for sid in expired:
            del self.sessions[sid]

        if (session_id not in self.sessions
                and len(self.sessions) >= MAX_SESSIONS):
            oldest = min(
                self.sessions.items(),
                key=lambda x: x[1]["last_active"]
            )
            del self.sessions[oldest[0]]

        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "history": [],
                "last_active": now
            }

        self.sessions[session_id]["last_active"] = now
        return self.sessions[session_id]["history"]

    def add_message(self, session_id: str, message: dict):
        history = self.get_or_create(session_id)
        history.append(message)
        if len(history) > MAX_HISTORY:
            history.pop(0)

    def clear(self, session_id: str):
        if session_id in self.sessions:
            self.sessions[session_id]["history"] = []

    def get_info(self, session_id: str) -> dict:
        if session_id not in self.sessions:
            return {"exists": False, "message_count": 0}
        data = self.sessions[session_id]
        return {
            "exists": True,
            "message_count": len(data["history"]),
            "last_active": datetime.fromtimestamp(
                data["last_active"]
            ).strftime("%Y-%m-%d %H:%M:%S")
        }


# ── Agent 服务 ────────────────────────────────────────

class AgentService:

    def __init__(self, memory_service: MemoryService):
        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL
        )
        self.session_manager = SessionManager()
        self.memory_service = memory_service
        print("✅ Agent服务初始化完成")

    # ── Function Calling 循环 ─────────────────────────

    def _run_fc_loop(
        self,
        messages: list,
        session_id: str,
        user_input: str
    ) -> str:
        for _ in range(5):
            response = self.client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=messages,
                tools=FC_TOOLS,
                tool_choice="auto"
            )
            message = response.choices[0].message

            if not message.tool_calls:
                self.session_manager.add_message(
                    session_id,
                    {"role": "user", "content": user_input}
                )
                self.session_manager.add_message(
                    session_id,
                    {
                        "role": "assistant",
                        "content": message.content
                    }
                )
                self.memory_service.save(
                    messages=[
                        {"role": "user", "content": user_input},
                        {
                            "role": "assistant",
                            "content": message.content
                        }
                    ],
                    user_id=session_id
                )
                return message.content

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
                func_args = json.loads(
                    tool_call.function.arguments
                )
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

        return "处理超时，请重试"

    # ── 对话 ──────────────────────────────────────────

    def chat(self, session_id: str, user_input: str) -> str:
        memories = self.memory_service.search(
            user_input, session_id
        )

        system_content = f"""你是专业投资分析助手。
今天是{datetime.now().strftime("%Y年%m月%d日")}。
主动调用工具获取实时数据，回答专业客观，适当提示风险。"""

        if memories:
            system_content += f"\n\n关于该用户你记得：\n{memories}"

        messages = [
            {"role": "system", "content": system_content},
            *self.session_manager.get_or_create(session_id),
            {"role": "user", "content": user_input}
        ]

        return self._run_fc_loop(messages, session_id, user_input)

    # ── 个股分析（CrewAI） ────────────────────────────

    def _run_crew(
        self,
        stock_code: str,
        total_assets: float = None
    ) -> str:
        import os
        from crewai import LLM
        os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
        os.environ["OTEL_SDK_DISABLED"] = "true"

        sllm = LLM(
            model="openai/deepseek-chat",
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL
        )

        data_collector = Agent(
            role="股票数据收集师",
            goal=f"收集 {stock_code} 的完整数据",
            backstory="专业金融数据分析师。",
            tools=[tool_get_price, tool_get_news, tool_get_fund_flow],
            verbose=False,
            allow_delegation=False,
            max_iter=3,
            llm=sllm
        )
        technical_analyst = Agent(
            role="技术分析师",
            goal=f"分析 {stock_code} 技术面",
            backstory="资深技术分析师。",
            tools=[tool_get_price, tool_get_fund_flow],
            verbose=False,
            allow_delegation=False,
            max_iter=3,
            llm=sllm
        )
        fundamental_analyst = Agent(
            role="基本面分析师",
            goal=f"分析 {stock_code} 基本面",
            backstory="专业基本面分析师。",
            tools=[tool_get_financial, tool_get_news],
            verbose=False,
            allow_delegation=False,
            max_iter=3,
            llm=sllm
        )
        risk_assessor = Agent(
            role="风险评估师",
            goal=f"评估 {stock_code} 投资风险",
            backstory="风险管理专家。",
            tools=[tool_get_news, tool_get_financial],
            verbose=False,
            allow_delegation=False,
            max_iter=3,
            llm=sllm
        )
        chief_analyst = Agent(
            role="首席投资分析师",
            goal=f"输出 {stock_code} 完整报告",
            backstory="20年经验首席分析师。",
            tools=[tool_calculate_position] if total_assets else [],
            verbose=False,
            allow_delegation=False,
            max_iter=3,
            llm=sllm
        )

        t_collect = Task(
            description=f"收集{stock_code}价格、新闻、资金流向数据。今天：{datetime.now().strftime('%Y-%m-%d')}",
            expected_output="结构化数据报告",
            agent=data_collector
        )
        t_tech = Task(
            description=f"分析{stock_code}技术面，给出评级",
            expected_output="技术面报告",
            agent=technical_analyst,
            context=[t_collect]
        )
        t_fund = Task(
            description=f"分析{stock_code}基本面，给出评级",
            expected_output="基本面报告",
            agent=fundamental_analyst,
            context=[t_collect]
        )
        t_risk = Task(
            description=f"评估{stock_code}投资风险",
            expected_output="风险报告",
            agent=risk_assessor,
            context=[t_collect, t_fund]
        )
        assets_info = (
            f"用户总资产：{total_assets}元"
            if total_assets else "未提供资产信息"
        )
        t_report = Task(
            description=f"""整合所有分析输出完整报告。{assets_info}
格式：
## 📊 基本信息
## 📈 技术面
## 💼 基本面
## ⚠️ 风险
## 💡 综合建议
{"## 💰 仓位建议" if total_assets else ""}
## 📝 免责声明""",
            expected_output="完整投资分析报告",
            agent=chief_analyst,
            context=[t_collect, t_tech, t_fund, t_risk]
        )

        crew = Crew(
            agents=[
                data_collector, technical_analyst,
                fundamental_analyst, risk_assessor, chief_analyst
            ],
            tasks=[t_collect, t_tech, t_fund, t_risk, t_report],
            process=Process.sequential,
            verbose=False,
            output_log_file=False
        )
        try:
            return str(crew.kickoff())
        except Exception as e:
            import traceback
            print(f"❌ CrewAI 分析失败：{e}")
            print(traceback.format_exc())
            raise

    # ── 基金分析 ──────────────────────────────────────

    def _run_fund_analysis(
        self,
        fund_code: str,
        total_assets: float = None
    ) -> str:
        info = get_fund_info(fund_code)
        performance = get_fund_performance(fund_code)
        manager = get_fund_manager(fund_code)

        assets_info = (
            f"用户总资产：{total_assets}元"
            if total_assets else "未提供资产信息"
        )

        prompt = f"""你是专业基金分析师，请对以下基金进行全面分析。

基金代码：{fund_code}
{assets_info}

基金基本信息：
{info}

历史净值（近10日）：
{performance}

基金经理：
{manager}

请输出完整分析报告，格式：
## 📊 基金基本信息
## 📈 业绩表现分析
## 👤 基金经理评价
## ⚠️ 风险提示
## 💡 投资建议
{"## 💰 仓位建议" if total_assets else ""}
## 📝 免责声明"""

        response = self.client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=3000
        )
        return response.choices[0].message.content

    # ── 债券分析 ──────────────────────────────────────

    def _run_bond_analysis(
        self,
        bond_code: str,
        total_assets: float = None
    ) -> str:
        info = get_bond_info(bond_code)
        detail = get_bond_detail(bond_code)
        news = get_stock_news(bond_code, limit=3)

        assets_info = (
            f"用户总资产：{total_assets}元"
            if total_assets else "未提供资产信息"
        )

        prompt = f"""你是专业债券分析师，请对以下债券进行全面分析。

债券代码：{bond_code}
{assets_info}

债券行情：
{info}

债券详情（转股价、溢价率等）：
{detail}

相关新闻：
{news}

请输出完整分析报告，格式：
## 📊 债券基本信息
## 💹 价格与溢价分析
## 🔄 转股价值分析（如为可转债）
## ⚠️ 风险提示
## 💡 投资建议
{"## 💰 仓位建议" if total_assets else ""}
## 📝 免责声明"""

        response = self.client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=3000
        )
        return response.choices[0].message.content

    # ── 持仓分析 ──────────────────────────────────────

    def _run_position_analysis(
        self,
        positions_text: str
    ) -> str:
        prompt = f"""你是专业投资组合分析师，请对以下持仓进行全面分析。

持仓信息：
{positions_text}

今天是{datetime.now().strftime("%Y年%m月%d日")}。

请输出完整持仓分析报告，格式：
## 📊 持仓概览
## 🏗️ 持仓结构分析（行业/板块分布）
## ⚖️ 集中度风险评估
## 📈 各持仓简要点评
## 🔄 调仓建议
## ⚠️ 整体风险提示
## 📝 免责声明"""

        response = self.client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=4000
        )
        return response.choices[0].message.content

    # ── Async 封装 ────────────────────────────────────

    async def chat_async(
        self,
        session_id: str,
        user_input: str
    ) -> str:
        loop = asyncio.get_event_loop()
        import functools
        return await loop.run_in_executor(
            None,
            functools.partial(self.chat, session_id, user_input)
        )

    async def analyze_async(
        self,
        stock_code: str,
        total_assets: float = None
    ) -> str:
        loop = asyncio.get_event_loop()
        import functools
        return await loop.run_in_executor(
            None,
            functools.partial(self._run_crew, stock_code, total_assets)
        )

    async def analyze_fund_async(
        self,
        fund_code: str,
        total_assets: float = None
    ) -> str:
        loop = asyncio.get_event_loop()
        import functools
        return await loop.run_in_executor(
            None,
            functools.partial(
                self._run_fund_analysis, fund_code, total_assets
            )
        )

    async def analyze_bond_async(
        self,
        bond_code: str,
        total_assets: float = None
    ) -> str:
        loop = asyncio.get_event_loop()
        import functools
        return await loop.run_in_executor(
            None,
            functools.partial(
                self._run_bond_analysis, bond_code, total_assets
            )
        )

    async def analyze_position_async(
        self,
        positions_text: str
    ) -> str:
        loop = asyncio.get_event_loop()
        import functools
        return await loop.run_in_executor(
            None,
            functools.partial(
                self._run_position_analysis, positions_text
            )
        )