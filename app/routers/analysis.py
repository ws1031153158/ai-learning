# app/routers/analysis.py
from fastapi import APIRouter, Depends, Header
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
import requests
import json
from datetime import datetime

from app.models.database import get_db, WatchlistItem, Position, UserPreference
from app.services.auth_service import AuthService
from app.services.agent_service import AgentService
import app.state as state

router = APIRouter(prefix="/analysis", tags=["analysis"])


# ── 鉴权依赖 ──────────────────────────────────────────

def get_current_user_id(
    authorization: str = Header(None),
    db: Session = Depends(get_db)
) -> int:
    if not authorization or not authorization.startswith("Bearer "):
        raise ValueError("未登录")
    token = authorization.replace("Bearer ", "")
    user = AuthService.get_current_user(db, token)
    if not user:
        raise ValueError("Token无效或已过期")
    return user.id


# ── 请求模型 ──────────────────────────────────────────

class StockAnalysisRequest(BaseModel):
    stock_code: str
    total_assets: Optional[float] = None


class FundAnalysisRequest(BaseModel):
    fund_code: str
    total_assets: Optional[float] = None


class BondAnalysisRequest(BaseModel):
    bond_code: str
    total_assets: Optional[float] = None


class PositionAnalysisRequest(BaseModel):
    positions_text: str  # 用户输入的持仓文本


# ── 大盘行情 ──────────────────────────────────────────

def fetch_market_overview() -> dict:
    """获取大盘行情"""
    try:
        symbols = "sh000001,sz399001,sz399006"
        url = f"http://hq.sinajs.cn/list={symbols}"
        headers = {
            "Referer": "https://finance.sina.com.cn",
            "User-Agent": "Mozilla/5.0"
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.encoding = "gbk"

        result = {}
        lines = resp.text.strip().split("\n")
        names = ["上证指数", "深证成指", "创业板指"]
        codes = ["sh000001", "sz399001", "sz399006"]

        for i, line in enumerate(lines):
            if '"' not in line:
                continue
            data_str = line.split('"')[1]
            fields = data_str.split(",")
            if len(fields) < 10:
                continue
            price = float(fields[3]) if fields[3] else 0
            prev_close = float(fields[2]) if fields[2] else 0
            change_pct = round(
                (price - prev_close) / prev_close * 100, 2
            ) if prev_close > 0 else 0

            result[codes[i]] = {
                "name": names[i],
                "price": price,
                "change_pct": change_pct,
                "change_amount": round(price - prev_close, 2)
            }
        return result
    except Exception as e:
        print(f"大盘行情获取失败：{e}")
        return {}


# ── 接口 ──────────────────────────────────────────────

@router.get("/daily")
async def daily_report(
    authorization: str = Header(None),
    db: Session = Depends(get_db)
):
    """日报：大盘行情 + 自选股走势 + 市场情绪 + 新闻要点"""
    try:
        user_id = get_current_user_id(authorization, db)
    except ValueError as e:
        return JSONResponse(
            status_code=401,
            content={"success": False, "message": str(e)}
        )

    # 大盘行情
    market = fetch_market_overview()

    # 自选股走势
    watchlist = db.query(WatchlistItem).filter(
        WatchlistItem.user_id == user_id
    ).all()

    watchlist_data = []
    for item in watchlist:
        try:
            if item.type == "stock":
                symbol = (
                    f"sh{item.code}"
                    if item.code.startswith("6")
                    else f"sz{item.code}"
                )
                url = f"http://hq.sinajs.cn/list={symbol}"
                headers = {
                    "Referer": "https://finance.sina.com.cn",
                    "User-Agent": "Mozilla/5.0"
                }
                resp = requests.get(
                    url, headers=headers, timeout=5
                )
                resp.encoding = "gbk"
                data_str = resp.text.split('"')[1]
                fields = data_str.split(",")
                if len(fields) >= 10:
                    price = float(fields[3])
                    prev_close = float(fields[2])
                    change_pct = round(
                        (price - prev_close) / prev_close * 100, 2
                    ) if prev_close > 0 else 0

                    # 计算持仓成本
                    positions = db.query(Position).filter(
                        Position.watchlist_id == item.id
                    ).all()
                    total_shares = 0.0
                    total_cost = 0.0
                    for p in positions:
                        if p.action == "buy":
                            total_shares += p.shares
                            total_cost += p.price * p.shares
                        elif p.action == "sell":
                            total_shares -= p.shares
                    avg_cost = (
                        round(total_cost / total_shares, 2)
                        if total_shares > 0 else None
                    )
                    profit_pct = (
                        round((price - avg_cost) / avg_cost * 100, 2)
                        if avg_cost else None
                    )

                    watchlist_data.append({
                        "id": item.id,
                        "code": item.code,
                        "name": item.name,
                        "type": item.type,
                        "price": price,
                        "change_pct": change_pct,
                        "avg_cost": avg_cost,
                        "profit_pct": profit_pct
                    })
        except Exception as e:
            print(f"获取{item.code}行情失败：{e}")

    # 市场情绪（简单判断）
    if market:
        sh = market.get("sh000001", {})
        change = sh.get("change_pct", 0)
        if change > 1:
            sentiment = "乐观"
            sentiment_desc = "大盘上涨，市场情绪积极"
        elif change < -1:
            sentiment = "悲观"
            sentiment_desc = "大盘下跌，市场情绪谨慎"
        else:
            sentiment = "中性"
            sentiment_desc = "大盘震荡，市场情绪平稳"
    else:
        sentiment = "未知"
        sentiment_desc = "暂无数据"

    # ← 新增：财经新闻
    news_raw = fetch_news_raw(50)
    news = await filter_news_by_value(news_raw)

    return {
        "success": True,
        "data": {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "market": market,
            "watchlist": watchlist_data,
            "sentiment": {
                "level": sentiment,
                "desc": sentiment_desc
            },
            "news": news
        }
    }


@router.post("/stock")
async def analyze_stock(
    req: StockAnalysisRequest,
    authorization: str = Header(None),
    db: Session = Depends(get_db)
):
    """个股分析"""
    try:
        user_id = get_current_user_id(authorization, db)
    except ValueError as e:
        return JSONResponse(
            status_code=401,
            content={"success": False, "message": str(e)}
        )

    # 读取用户偏好
    pref = db.query(UserPreference).filter(
        UserPreference.user_id == user_id
    ).first()
    total_assets = req.total_assets or (
        pref.total_assets if pref else None
    )

    try:
        report = await state.agent_service.analyze_async(
            stock_code=req.stock_code,
            total_assets=total_assets
        )
        return {"success": True, "report": report}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e)}
        )


@router.post("/fund")
async def analyze_fund(
    req: FundAnalysisRequest,
    authorization: str = Header(None),
    db: Session = Depends(get_db)
):
    """基金分析（框架）"""
    try:
        user_id = get_current_user_id(authorization, db)
    except ValueError as e:
        return JSONResponse(
            status_code=401,
            content={"success": False, "message": str(e)}
        )

    return {
        "success": True,
        "report": f"## 基金 {req.fund_code} 分析\n\n功能开发中，敬请期待。"
    }


@router.post("/bond")
async def analyze_bond(
    req: BondAnalysisRequest,
    authorization: str = Header(None),
    db: Session = Depends(get_db)
):
    """债券分析（框架）"""
    try:
        user_id = get_current_user_id(authorization, db)
    except ValueError as e:
        return JSONResponse(
            status_code=401,
            content={"success": False, "message": str(e)}
        )

    return {
        "success": True,
        "report": f"## 债券 {req.bond_code} 分析\n\n功能开发中，敬请期待。"
    }

def fetch_news_raw(count: int = 50) -> list:
    """获取原始新闻"""
    try:
        url = "http://newsapi.eastmoney.com/kuaixun/v1/getlist_102_ajaxResult_50_1_.html"
        headers = {
            "Referer": "http://www.eastmoney.com",
            "User-Agent": "Mozilla/5.0"
        }
        resp = requests.get(url, headers=headers, timeout=10)
        text = resp.text.replace("var ajaxResult=", "")
        data = json.loads(text)

        news_list = []
        items = data.get("LivesList", [])[:count]
        for item in items:
            news_list.append({
                "title": item.get("title", ""),
                "time": item.get("showtime", ""),
                "url": item.get("url_w", "")
            })
        return news_list
    except Exception as e:
        print(f"新闻获取失败：{e}")
        return []


async def filter_news_by_value(news_list: list) -> list:
    """用 DeepSeek 筛选投资价值最高的新闻"""
    if not news_list:
        return []

    try:
        # 构建标题列表
        titles_text = "\n".join([
            f"{i+1}. [{item['time']}] {item['title']}"
            for i, item in enumerate(news_list)
        ])

        prompt = f"""你是一位专业的价值投资分析师。
以下是今日财经新闻列表，请从投资价值角度筛选出最值得关注的10条新闻。

评判标准：
1. 对A股市场有重大影响（政策、宏观经济、行业变化）
2. 涉及上市公司重大事件（业绩、并购、重组）
3. 对价值投资者有参考意义
4. 排除无实质内容的公告和无关新闻

新闻列表：
{titles_text}

请返回JSON格式，只返回JSON不要其他内容：
{{
  "selected": [
    {{
      "index": 原序号(1开始),
      "reason": "一句话说明投资价值"
    }}
  ]
}}"""

        from openai import AsyncOpenAI
        from app.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

        client = AsyncOpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL
        )

        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000
        )

        result_text = response.choices[0].message.content.strip()

        # 清理 markdown 代码块
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()

        result = json.loads(result_text)
        selected = result.get("selected", [])

        # 组合结果
        filtered = []
        for item in selected[:10]:
            idx = item.get("index", 0) - 1
            if 0 <= idx < len(news_list):
                news = news_list[idx].copy()
                news["reason"] = item.get("reason", "")
                filtered.append(news)

        return filtered

    except Exception as e:
        print(f"AI筛选新闻失败：{e}")
        # 降级：直接返回前10条
        return news_list[:10]

@router.post("/position")
async def analyze_position(
    req: PositionAnalysisRequest,
    authorization: str = Header(None),
    db: Session = Depends(get_db)
):
    """持仓分析（框架）"""
    try:
        user_id = get_current_user_id(authorization, db)
    except ValueError as e:
        return JSONResponse(
            status_code=401,
            content={"success": False, "message": str(e)}
        )

    return {
        "success": True,
        "report": "## 持仓分析\n\n功能开发中，敬请期待。"
    }