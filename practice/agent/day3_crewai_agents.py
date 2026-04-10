# agent/day3_crewai_agents.py
import sys
sys.path.append(".")

import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

import asyncio
import akshare as ak

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

from practice.agent.tools.stock_tools import (
    get_stock_price,
    get_stock_news,
    get_fund_flow,
    get_financial_indicator,
    calculate_position
)

load_dotenv(override=True)

# CrewAI 需要设置环境变量
os.environ["OPENAI_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com"
os.environ["OPENAI_MODEL_NAME"] = "deepseek-chat"

# ============================================================
# 把工具函数包装成 CrewAI 工具
# ============================================================

# ✅ 改成英文
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
    """获取股票主力资金流向数据，输入股票代码如600519"""
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
# 并行执行
# ============================================================

async def fetch_data_parallel(stock_code: str) -> dict:
    """
    并行获取所有数据，速度比串行快4倍
    """
    loop = asyncio.get_event_loop()

    # 用 run_in_executor 把同步函数变成异步
    # 真正并行执行
    tasks = await asyncio.gather(
        loop.run_in_executor(None, get_stock_price, stock_code),
        loop.run_in_executor(None, get_stock_news, stock_code, 5),
        loop.run_in_executor(None, get_fund_flow, stock_code),
        loop.run_in_executor(None, get_financial_indicator, stock_code),
        return_exceptions=True  # 某个失败不影响其他
    )

    price, news, fund, financial = tasks

    return {
        "price": price if not isinstance(price, Exception) else "获取失败",
        "news": news if not isinstance(news, Exception) else "获取失败",
        "fund": fund if not isinstance(fund, Exception) else "获取失败",
        "financial": financial if not isinstance(financial, Exception) else "获取失败"
    }

# 测试并行 vs 串行速度对比
async def speed_test(stock_code: str = "600519"):
    import time

    print("测试串行速度...")
    start = time.time()
    get_stock_price(stock_code)
    get_stock_news(stock_code)
    get_fund_flow(stock_code)
    get_financial_indicator(stock_code)
    serial_time = time.time() - start
    print(f"串行耗时：{serial_time:.2f}秒")

    print("\n测试并行速度...")
    start = time.time()
    await fetch_data_parallel(stock_code)
    parallel_time = time.time() - start
    print(f"并行耗时：{parallel_time:.2f}秒")

    print(f"\n速度提升：{serial_time/parallel_time:.1f}倍")

# ============================================================
# 定义 Agent 团队
# ============================================================

def create_analysis_crew(stock_code: str, total_assets: float = None):
    """
    创建股票分析团队
    """

    # ── Agent 1：数据收集师 ──────────────────────────────
    data_collector = Agent(
        role="股票数据收集师",
        goal=f"收集 {stock_code} 的完整数据，包括价格、新闻、资金流向",
        backstory="""你是专业的金融数据分析师，擅长收集和整理股票数据。
你的职责是获取最新的市场数据，为后续分析提供准确的数据基础。
你只负责收集数据，不做主观判断。""",
        tools=[tool_get_price, tool_get_news, tool_get_fund_flow],
        verbose=True,
        allow_delegation=False,
        max_iter=3
    )

    # ── Agent 2：技术分析师 ──────────────────────────────
    technical_analyst = Agent(
        role="技术分析师",
        goal=f"基于 {stock_code} 的价格和资金数据，分析技术面走势",
        backstory="""你是资深技术分析师，擅长K线分析、均线系统、量价关系。
你会根据价格走势和资金流向判断短期趋势。
你的分析客观严谨，不受情绪影响。""",
        tools=[tool_get_price, tool_get_fund_flow],
        verbose=True,
        allow_delegation=False,
        max_iter=3
    )

    # ── Agent 3：基本面分析师 ────────────────────────────
    fundamental_analyst = Agent(
        role="基本面分析师",
        goal=f"分析 {stock_code} 的财务状况和基本面质量",
        backstory="""你是专业的基本面分析师，擅长财务报表分析、估值模型。
你会从ROE、净利润增长率、资产负债率等维度评估公司质量。
你注重长期价值，不被短期波动影响。""",
        tools=[tool_get_financial, tool_get_news],
        verbose=True,
        allow_delegation=False,
        max_iter=3
    )

    # ── Agent 4：风险评估师 ──────────────────────────────
    risk_assessor = Agent(
        role="风险评估师",
        goal=f"评估投资 {stock_code} 的主要风险点",
        backstory="""你是专业的风险管理专家，擅长识别投资风险。
你会从市场风险、基本面风险、政策风险等维度评估。
你的职责是保护投资者，永远把风险提示放在第一位。""",
        tools=[tool_get_news, tool_get_financial],
        verbose=True,
        allow_delegation=False,
        max_iter=3
    )

    # ── Agent 5：首席分析师 ──────────────────────────────
    chief_analyst = Agent(
        role="首席投资分析师",
        goal=f"整合所有分析，输出 {stock_code} 的专业投资分析报告",
        backstory="""你是拥有20年经验的首席投资分析师。
你会综合技术面、基本面、风险评估，给出客观的投资建议。
你的报告专业、清晰、有据可查。
你始终提醒：投资有风险，以上仅供参考，不构成投资建议。""",
        tools=[tool_calculate_position] if total_assets else [],
        verbose=True,
        allow_delegation=False,
        max_iter=3
    )

    # ============================================================
    # 定义任务
    # ============================================================

    task_collect = Task(
        description=f"""收集股票 {stock_code} 的以下数据：
1. 最新价格、涨跌幅、成交量
2. 最新5条新闻和公告
3. 近3日主力资金流向

请将所有数据整理成结构化的报告，供后续分析使用。
今天日期：{datetime.now().strftime("%Y-%m-%d")}""",
        expected_output="包含价格、新闻、资金流向的完整数据报告",
        agent=data_collector
    )

    task_technical = Task(
        description=f"""基于已收集的数据，对 {stock_code} 进行技术面分析：
1. 分析最新价格趋势（上涨/下跌/震荡）
2. 分析成交量变化（放量/缩量）
3. 分析主力资金动向（流入/流出）
4. 给出短期技术面评级（强势/中性/弱势）

要求：有数据支撑，不要主观臆断。""",
        expected_output="技术面分析报告，包含趋势判断和评级",
        agent=technical_analyst,
        context=[task_collect]
    )

    task_fundamental = Task(
        description=f"""对 {stock_code} 进行基本面分析：
1. 分析核心财务指标（ROE、净利润增长率、资产负债率）
2. 评估公司盈利质量
3. 结合新闻判断基本面是否有重大变化
4. 给出基本面评级（优质/良好/一般/较差）

要求：数据说话，客观评估。""",
        expected_output="基本面分析报告，包含财务评估和评级",
        agent=fundamental_analyst,
        context=[task_collect]
    )

    task_risk = Task(
        description=f"""评估投资 {stock_code} 的主要风险：
1. 市场风险（当前市场环境）
2. 基本面风险（财务或业务风险）
3. 政策风险（行业政策变化）
4. 舆情风险（负面新闻影响）
5. 综合风险等级（高/中/低）

要求：风险提示要具体，不要泛泛而谈。""",
        expected_output="风险评估报告，包含具体风险点和综合等级",
        agent=risk_assessor,
        context=[task_collect, task_fundamental]
    )

    assets_info = f"用户总资产：{total_assets}元" if total_assets else "用户未提供资产信息"

    task_report = Task(
        description=f"""整合所有分析，为 {stock_code} 输出完整投资分析报告。

{assets_info}

报告格式：
## 📊 股票基本信息
## 📈 技术面分析
## 💼 基本面分析
## ⚠️ 风险评估
## 💡 综合建议
{"## 💰 仓位建议（基于用户资产）" if total_assets else ""}
## 📝 免责声明

要求：
- 综合技术面、基本面、风险三个维度
- 给出明确的综合评级
- 仓位建议要结合风险等级
- 最后必须加免责声明""",
        expected_output="完整的专业投资分析报告",
        agent=chief_analyst,
        context=[
            task_collect,
            task_technical,
            task_fundamental,
            task_risk
        ]
    )

    # ============================================================
    # 组建团队
    # ============================================================

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
        process=Process.sequential,  # 顺序执行
        verbose=True
    )

    return crew


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("理财AI分析师 - 多Agent协作分析系统")
    print(f"当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # 测试1：基础分析（不带资产信息）
    print("\n【测试1】分析茅台（不带资产信息）")
    crew = create_analysis_crew("600519")
    result = crew.kickoff()
    print("\n" + "=" * 60)
    print("最终报告：")
    print(result)

    # 测试2：带资产信息的完整分析
    print("\n【测试2】分析比亚迪（带资产信息）")
    crew2 = create_analysis_crew("002594", total_assets=500000)
    result2 = crew2.kickoff()
    print("\n" + "=" * 60)
    print("最终报告：")
    print(result2)

    # 测试3：速度对比
    asyncio.run(speed_test("600519"))