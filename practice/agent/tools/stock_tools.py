# agent/tools/stock_tools.py
import sys
sys.path.append(".")

import akshare as ak
from datetime import datetime, timedelta
import json


def get_stock_price(stock_code: str) -> str:
    """
    获取股票最新价格和涨跌情况
    
    Args:
        stock_code: 股票代码，如 600519
    Returns:
        包含价格信息的字符串
    """
    try:
        today = datetime.now().strftime("%Y%m%d")
        yesterday = (
            datetime.now() - timedelta(days=5)
        ).strftime("%Y%m%d")

        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date=yesterday,
            end_date=today,
            adjust="qfq"
        )

        if df.empty:
            return f"未找到股票 {stock_code} 的数据"

        latest = df.iloc[-1]
        return json.dumps({
            "stock_code": stock_code,
            "price": latest['收盘'],
            "change_pct": latest['涨跌幅'],
            "change_amount": latest['涨跌额'],
            "volume": latest['成交量'],
            "date": str(latest['日期'])
        }, ensure_ascii=False)

    except Exception as e:
        return f"获取价格失败：{e}"


def get_stock_news(stock_code: str, limit: int = 5) -> str:
    """
    获取股票最新新闻

    Args:
        stock_code: 股票代码，如 600519
        limit: 返回新闻数量
    Returns:
        新闻列表字符串
    """
    try:
        df = ak.stock_news_em(symbol=stock_code)
        df = df.head(limit)

        news_list = []
        for _, row in df.iterrows():
            news_list.append({
                "title": row.get('新闻标题', ''),
                "time": str(row.get('发布时间', '')),
                "summary": row.get('新闻内容', '')[:200]
            })

        return json.dumps(news_list, ensure_ascii=False)

    except Exception as e:
        return f"获取新闻失败：{e}"


def get_fund_flow(stock_code: str) -> str:
    """
    获取股票主力资金流向

    Args:
        stock_code: 股票代码，如 600519
    Returns:
        资金流向数据字符串
    """
    try:
        market = "sh" if stock_code.startswith("6") else "sz"
        df = ak.stock_individual_fund_flow(
            stock=stock_code,
            market=market
        )
        df = df.tail(3)

        flow_list = []
        for _, row in df.iterrows():
            main_flow = row.get('主力净流入-净额', 0)
            flow_list.append({
                "date": str(row.get('日期', '')),
                "close": row.get('收盘价', 0),
                "change_pct": row.get('涨跌幅', 0),
                "main_flow": round(main_flow / 1e8, 2),
                "main_flow_ratio": row.get('主力净流入-净占比', 0),
                "status": "流入" if main_flow > 0 else "流出"
            })

        return json.dumps(flow_list, ensure_ascii=False)

    except Exception as e:
        return f"获取资金流向失败：{e}"


def get_financial_indicator(stock_code: str) -> str:
    """
    获取股票核心财务指标

    Args:
        stock_code: 股票代码，如 600519
    Returns:
        财务指标字符串
    """
    try:
        df = ak.stock_financial_analysis_indicator(
            symbol=stock_code,
            start_year="2024"
        )

        if df.empty:
            return f"未找到 {stock_code} 的财务数据"

        latest = df.iloc[0]
        result = {
            "stock_code": stock_code,
            "report_date": str(latest.get('日期', 'N/A')),
            "eps": latest.get('摊薄每股收益(元)', 'N/A'),
            "roe": latest.get('净资产收益率(%)', 'N/A'),
            "roa": latest.get('总资产净利润率(%)', 'N/A'),
            "profit_margin": latest.get('销售净利率(%)', 'N/A'),
            "revenue_growth": latest.get('主营业务收入增长率(%)', 'N/A'),
            "net_profit_growth": latest.get('净利润增长率(%)', 'N/A'),
            "debt_ratio": latest.get('资产负债率(%)', 'N/A'),
        }

        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        return f"获取财务数据失败：{e}"


def calculate_position(
    stock_code: str,
    total_assets: float,
    risk_level: str = "medium"
) -> str:
    """
    根据风险等级计算建议仓位

    Args:
        stock_code: 股票代码
        total_assets: 总资产（元）
        risk_level: 风险等级 low/medium/high
    Returns:
        仓位建议字符串
    """
    risk_map = {
        "low": 0.1,
        "medium": 0.2,
        "high": 0.3
    }

    ratio = risk_map.get(risk_level, 0.2)
    suggested_amount = total_assets * ratio

    result = {
        "stock_code": stock_code,
        "total_assets": total_assets,
        "risk_level": risk_level,
        "suggested_ratio": f"{ratio*100}%",
        "suggested_amount": suggested_amount,
        "note": "仅供参考，不构成投资建议"
    }

    return json.dumps(result, ensure_ascii=False)