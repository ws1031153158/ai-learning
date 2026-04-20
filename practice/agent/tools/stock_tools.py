# practice/agent/tools/stock_tools.py
import sys
sys.path.append(".")

import requests
import json
from datetime import datetime, timedelta


# ── 新浪财经接口 ──────────────────────────────────────

def _get_sina_realtime(stock_code: str) -> dict:
    """
    新浪财经实时行情
    服务器可以正常访问
    """
    # 判断市场前缀
    if stock_code.startswith("6"):
        symbol = f"sh{stock_code}"
    elif stock_code.startswith("0") or stock_code.startswith("3"):
        symbol = f"sz{stock_code}"
    else:
        symbol = f"sh{stock_code}"

    url = f"http://hq.sinajs.cn/list={symbol}"
    headers = {
        "Referer": "https://finance.sina.com.cn",
        "User-Agent": "Mozilla/5.0"
    }

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.encoding = "gbk"
        content = resp.text

        # 解析新浪返回格式
        # var hq_str_sh600519="贵州茅台,1750.00,..."
        data_str = content.split('"')[1]
        if not data_str:
            return {}

        fields = data_str.split(",")
        if len(fields) < 32:
            return {}

        return {
            "name": fields[0],
            "open": float(fields[1]),
            "prev_close": float(fields[2]),
            "price": float(fields[3]),
            "high": float(fields[4]),
            "low": float(fields[5]),
            "volume": float(fields[8]),
            "amount": float(fields[9]),
            "date": fields[30],
            "time": fields[31]
        }
    except Exception as e:
        return {"error": str(e)}


def get_stock_price(stock_code: str) -> str:
    """
    获取股票最新价格和涨跌情况
    """
    try:
        data = _get_sina_realtime(stock_code)

        if "error" in data or not data:
            return f"获取价格失败：{data.get('error', '数据为空')}"

        price = data["price"]
        prev_close = data["prev_close"]
        change_amount = round(price - prev_close, 2)
        change_pct = round(
            (price - prev_close) / prev_close * 100, 2
        ) if prev_close > 0 else 0

        result = {
            "stock_code": stock_code,
            "name": data["name"],
            "price": price,
            "change_pct": change_pct,
            "change_amount": change_amount,
            "high": data["high"],
            "low": data["low"],
            "volume": data["volume"],
            "date": data["date"]
        }
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        return f"获取价格失败：{e}"


def get_stock_news(stock_code: str, limit: int = 5) -> str:
    """
    获取股票最新新闻
    东方财富新闻接口服务器可以正常访问
    """
    try:
        url = "https://np-listapi.eastmoney.com/comm/wap/getListInfo"
        params = {
            "cb": "callback",
            "client": "wap",
            "type": 1,
            "mTypeAndCode": f"0,{stock_code}",
            "pageSize": limit,
            "pageIndex": 1,
            "callback": "cb"
        }
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(
            url, params=params,
            headers=headers, timeout=10
        )

        # 解析 jsonp
        text = resp.text
        start = text.index("(") + 1
        end = text.rindex(")")
        data = json.loads(text[start:end])

        news_list = []
        items = (
            data.get("data", {})
            .get("list", [])
        )
        for item in items[:limit]:
            news_list.append({
                "title": item.get("title", ""),
                "time": item.get("datetime", ""),
                "summary": item.get("digest", "")[:200]
            })

        if not news_list:
            return f"暂无 {stock_code} 相关新闻"

        return json.dumps(news_list, ensure_ascii=False)

    except Exception as e:
        return f"获取新闻失败：{e}"


def get_fund_flow(stock_code: str) -> str:
    """
    获取主力资金流向
    使用东方财富资金流向接口
    """
    try:
        if stock_code.startswith("6"):
            market = 1  # 沪市
        else:
            market = 0  # 深市

        url = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
        params = {
            "lmt": 5,
            "klt": 101,
            "secid": f"{market}.{stock_code}",
            "fields1": "f1,f2,f3,f7",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63",
            "ut": "b2884a393a59ad64002292a3e90d46a5"
        }
        headers = {
            "Referer": "https://data.eastmoney.com",
            "User-Agent": "Mozilla/5.0"
        }

        resp = requests.get(
            url, params=params,
            headers=headers, timeout=10
        )
        data = resp.json()

        klines = (
            data.get("data", {})
            .get("klines", [])
        )
        if not klines:
            return f"暂无 {stock_code} 资金流向数据"

        flow_list = []
        for kline in klines[-5:]:
            fields = kline.split(",")
            if len(fields) < 7:
                continue
            main_flow = float(fields[1]) if fields[1] != "-" else 0
            flow_list.append({
                "date": fields[0],
                "main_flow": round(main_flow / 1e8, 2),
                "status": "流入" if main_flow > 0 else "流出",
                "main_flow_ratio": fields[6]
            })

        return json.dumps(flow_list, ensure_ascii=False)

    except Exception as e:
        return f"获取资金流向失败：{e}"


def get_financial_indicator(stock_code: str) -> str:
    """
    获取核心财务指标
    使用东方财富财务接口
    """
    try:
        if stock_code.startswith("6"):
            market = "SH"
        else:
            market = "SZ"

        url = "https://emweb.securities.eastmoney.com/PC_HSF10/NewFinanceAnalysis/ZYZBAjaxNew"
        params = {
            "type": 1,
            "code": f"{market}{stock_code}"
        }
        headers = {
            "Referer": "https://emweb.securities.eastmoney.com",
            "User-Agent": "Mozilla/5.0"
        }

        resp = requests.get(
            url, params=params,
            headers=headers, timeout=10
        )
        data = resp.json()

        items = data.get("data", [])
        if not items:
            return f"暂无 {stock_code} 财务数据"

        latest = items[0]
        result = {
            "stock_code": stock_code,
            "report_date": latest.get("REPORTDATE", "N/A"),
            "eps": latest.get("EPSBASIC", "N/A"),
            "roe": latest.get("ROEJQ", "N/A"),
            "roa": latest.get("ZZCJLL", "N/A"),
            "profit_margin": latest.get("XSMLL", "N/A"),
            "revenue_growth": latest.get("YYZSRGRATE", "N/A"),
            "net_profit_growth": latest.get("GSJLRGRATE", "N/A"),
            "debt_ratio": latest.get("ZCFZL", "N/A"),
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
    纯计算，不需要网络请求
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
        "suggested_ratio": f"{ratio * 100}%",
        "suggested_amount": suggested_amount,
        "note": "仅供参考，不构成投资建议"
    }

    return json.dumps(result, ensure_ascii=False)

# ── 基金数据 ──────────────────────────────────────────

def get_fund_info(fund_code: str) -> str:
    """获取基金基本信息和净值"""
    try:
        url = "http://fundgz.1234567.com.cn/js/{}.js".format(fund_code)
        headers = {
            "Referer": "http://fund.eastmoney.com",
            "User-Agent": "Mozilla/5.0"
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.encoding = "utf-8"
        text = resp.text

        # 解析 jsonp
        start = text.index("(") + 1
        end = text.rindex(")")
        data = json.loads(text[start:end])

        result = {
            "fund_code": fund_code,
            "name": data.get("name", ""),
            "nav": data.get("dwjz", ""),           # 单位净值
            "nav_date": data.get("jzrq", ""),       # 净值日期
            "estimated_nav": data.get("gsz", ""),   # 估算净值
            "estimated_change": data.get("gszzl", ""),  # 估算涨跌幅
            "fund_type": data.get("fundtype", "")
        }
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return f"获取基金信息失败：{e}"


def get_fund_performance(fund_code: str) -> str:
    """获取基金历史业绩"""
    try:
        url = "https://api.fund.eastmoney.com/f10/lsjz"
        params = {
            "fundCode": fund_code,
            "pageIndex": 1,
            "pageSize": 10,
            "startDate": "",
            "endDate": ""
        }
        headers = {
            "Referer": "http://fund.eastmoney.com",
            "User-Agent": "Mozilla/5.0"
        }
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        data = resp.json()

        items = data.get("Data", {}).get("LSJZList", [])
        if not items:
            return f"暂无 {fund_code} 历史净值数据"

        result = []
        for item in items[:10]:
            result.append({
                "date": item.get("FSRQ", ""),
                "nav": item.get("DWJZ", ""),
                "acc_nav": item.get("LJJZ", ""),
                "change_pct": item.get("JZZZL", "")
            })
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return f"获取基金业绩失败：{e}"


def get_fund_manager(fund_code: str) -> str:
    """获取基金经理信息"""
    try:
        url = f"https://api.fund.eastmoney.com/f10/jjjl/{fund_code}"
        params = {"fundcode": fund_code}
        headers = {
            "Referer": "http://fund.eastmoney.com",
            "User-Agent": "Mozilla/5.0"
        }
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        data = resp.json()

        managers = data.get("Data", {}).get("currentManager", [])
        if not managers:
            return f"暂无 {fund_code} 基金经理数据"

        result = []
        for m in managers:
            result.append({
                "name": m.get("name", ""),
                "start_date": m.get("startDate", ""),
                "manage_days": m.get("days", ""),
                "total_return": m.get("profit", "")
            })
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return f"获取基金经理失败：{e}"


# ── 债券数据 ──────────────────────────────────────────

def get_bond_info(bond_code: str) -> str:
    """获取债券基本信息和价格"""
    try:
        # 债券代码前缀判断
        if bond_code.startswith("1"):
            symbol = f"sh{bond_code}"
        else:
            symbol = f"sz{bond_code}"

        url = f"http://hq.sinajs.cn/list={symbol}"
        headers = {
            "Referer": "https://finance.sina.com.cn",
            "User-Agent": "Mozilla/5.0"
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.encoding = "gbk"
        text = resp.text

        data_str = text.split('"')[1]
        fields = data_str.split(",")
        if len(fields) < 10:
            return f"暂无 {bond_code} 债券数据"

        price = float(fields[3]) if fields[3] else 0
        prev_close = float(fields[2]) if fields[2] else 0
        change_pct = round(
            (price - prev_close) / prev_close * 100, 2
        ) if prev_close > 0 else 0

        result = {
            "bond_code": bond_code,
            "name": fields[0],
            "price": price,
            "prev_close": prev_close,
            "change_pct": change_pct,
            "high": float(fields[4]) if fields[4] else 0,
            "low": float(fields[5]) if fields[5] else 0,
            "volume": float(fields[8]) if fields[8] else 0,
        }
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return f"获取债券信息失败：{e}"


def get_bond_detail(bond_code: str) -> str:
    """获取债券详细信息（票面利率、到期日等）"""
    try:
        if bond_code.startswith("1"):
            market = "SH"
        else:
            market = "SZ"

        url = "https://push2.eastmoney.com/api/qt/stock/get"
        params = {
            "secid": f"{'1' if market == 'SH' else '0'}.{bond_code}",
            "fields": "f57,f58,f84,f85,f107,f116,f117,f162,f163,f164",
            "ut": "b2884a393a59ad64002292a3e90d46a5"
        }
        headers = {
            "Referer": "https://data.eastmoney.com",
            "User-Agent": "Mozilla/5.0"
        }
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        data = resp.json()

        d = data.get("data", {})
        result = {
            "bond_code": bond_code,
            "name": d.get("f58", ""),
            "pe": d.get("f162", ""),       # 溢价率
            "conv_price": d.get("f163", ""),  # 转股价
            "conv_value": d.get("f164", ""),  # 转股价值
        }
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return f"获取债券详情失败：{e}"