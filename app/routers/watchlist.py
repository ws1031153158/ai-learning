# app/routers/watchlist.py
from fastapi import APIRouter, Depends, Header
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta
import requests
import json

from app.models.database import get_db, WatchlistItem, Position
from app.services.auth_service import AuthService

router = APIRouter(prefix="/watchlist", tags=["watchlist"])


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

class AddWatchlistRequest(BaseModel):
    code: str
    name: str
    type: str           # stock/fund/bond
    data_range: int = 7  # 3/7/14/30


class AddPositionRequest(BaseModel):
    watchlist_id: int
    action: str         # buy/sell
    price: float
    shares: float
    action_date: str    # "2024-01-01"
    note: Optional[str] = None


# ── 获取K线数据 ───────────────────────────────────────

def fetch_kline(
    code: str,
    item_type: str,
    data_range: int
) -> list:
    """获取K线数据"""
    try:
        if item_type == "stock":
            # 判断市场
            if code.startswith("6"):
                symbol = f"sh{code}"
            else:
                symbol = f"sz{code}"

            end = datetime.now().strftime("%Y%m%d")
            start = (
                datetime.now() - timedelta(days=data_range + 10)
            ).strftime("%Y%m%d")

            url = f"https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20_{symbol}_daily/CN_MarketDataService.getKLineData"
            params = {
                "symbol": symbol,
                "scale": 240,
                "ma": "no",
                "datalen": data_range + 10
            }
            headers = {
                "Referer": "https://finance.sina.com.cn",
                "User-Agent": "Mozilla/5.0"
            }
            resp = requests.get(
                url, params=params,
                headers=headers, timeout=10
            )
            resp.encoding = "gbk"
            text = resp.text

            # 跳过前面的注释，取最后一行
            lines = text.strip().split("\n")
            jsonp_line = lines[-1]

            # 解析 jsonp
            start_idx = jsonp_line.index("(") + 1
            end_idx = jsonp_line.rindex(")")
            data = json.loads(jsonp_line[start_idx:end_idx])

            klines = []
            for item in data[-data_range:]:
                klines.append({
                    "date": item.get("day", ""),
                    "open": float(item.get("open", 0)),
                    "high": float(item.get("high", 0)),
                    "low": float(item.get("low", 0)),
                    "close": float(item.get("close", 0)),
                    "volume": float(item.get("volume", 0))
                })
            return klines

    except Exception as e:
        print(f"K线获取失败：{e}")
    return []


# ── 接口 ──────────────────────────────────────────────

@router.get("")
def get_watchlist(
    authorization: str = Header(None),
    db: Session = Depends(get_db)
):
    try:
        user_id = get_current_user_id(authorization, db)
    except ValueError as e:
        return JSONResponse(
            status_code=401,
            content={"success": False, "message": str(e)}
        )

    items = db.query(WatchlistItem).filter(
        WatchlistItem.user_id == user_id
    ).all()

    result = []
    for item in items:
        # 获取持仓记录
        positions = db.query(Position).filter(
            Position.watchlist_id == item.id
        ).order_by(Position.action_date).all()

        position_list = [
            {
                "id": p.id,
                "action": p.action,
                "price": p.price,
                "shares": p.shares,
                "action_date": p.action_date.strftime("%Y-%m-%d"),
                "note": p.note
            }
            for p in positions
        ]

        # 计算持仓成本
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

        result.append({
            "id": item.id,
            "code": item.code,
            "name": item.name,
            "type": item.type,
            "data_range": item.data_range,
            "positions": position_list,
            "avg_cost": avg_cost,
            "total_shares": round(total_shares, 2)
        })

    return {"success": True, "data": result}


@router.post("")
def add_watchlist(
    req: AddWatchlistRequest,
    authorization: str = Header(None),
    db: Session = Depends(get_db)
):
    try:
        user_id = get_current_user_id(authorization, db)
    except ValueError as e:
        return JSONResponse(
            status_code=401,
            content={"success": False, "message": str(e)}
        )

    # 检查是否已添加
    existing = db.query(WatchlistItem).filter(
        WatchlistItem.user_id == user_id,
        WatchlistItem.code == req.code
    ).first()
    if existing:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "已在自选列表中"}
        )

    item = WatchlistItem(
        user_id=user_id,
        code=req.code,
        name=req.name,
        type=req.type,
        data_range=req.data_range
    )
    db.add(item)
    db.commit()
    db.refresh(item)

    return {
        "success": True,
        "data": {
            "id": item.id,
            "code": item.code,
            "name": item.name,
            "type": item.type,
            "data_range": item.data_range
        }
    }


@router.delete("/{item_id}")
def delete_watchlist(
    item_id: int,
    authorization: str = Header(None),
    db: Session = Depends(get_db)
):
    try:
        user_id = get_current_user_id(authorization, db)
    except ValueError as e:
        return JSONResponse(
            status_code=401,
            content={"success": False, "message": str(e)}
        )

    item = db.query(WatchlistItem).filter(
        WatchlistItem.id == item_id,
        WatchlistItem.user_id == user_id
    ).first()

    if not item:
        return JSONResponse(
            status_code=404,
            content={"success": False, "message": "不存在"}
        )

    # 同时删除持仓记录
    db.query(Position).filter(
        Position.watchlist_id == item_id
    ).delete()
    db.delete(item)
    db.commit()

    return {"success": True}


@router.get("/{item_id}/kline")
def get_kline(
    item_id: int,
    authorization: str = Header(None),
    db: Session = Depends(get_db)
):
    try:
        user_id = get_current_user_id(authorization, db)
    except ValueError as e:
        return JSONResponse(
            status_code=401,
            content={"success": False, "message": str(e)}
        )

    item = db.query(WatchlistItem).filter(
        WatchlistItem.id == item_id,
        WatchlistItem.user_id == user_id
    ).first()

    if not item:
        return JSONResponse(
            status_code=404,
            content={"success": False, "message": "不存在"}
        )

    klines = fetch_kline(item.code, item.type, item.data_range)

    # 获取买卖点
    positions = db.query(Position).filter(
        Position.watchlist_id == item_id
    ).order_by(Position.action_date).all()

    markers = [
        {
            "date": p.action_date.strftime("%Y-%m-%d"),
            "action": p.action,
            "price": p.price,
            "shares": p.shares,
            "note": p.note
        }
        for p in positions
    ]

    return {
        "success": True,
        "data": {
            "code": item.code,
            "name": item.name,
            "klines": klines,
            "markers": markers
        }
    }


@router.post("/position")
def add_position(
    req: AddPositionRequest,
    authorization: str = Header(None),
    db: Session = Depends(get_db)
):
    try:
        user_id = get_current_user_id(authorization, db)
    except ValueError as e:
        return JSONResponse(
            status_code=401,
            content={"success": False, "message": str(e)}
        )

    # 验证 watchlist 归属
    item = db.query(WatchlistItem).filter(
        WatchlistItem.id == req.watchlist_id,
        WatchlistItem.user_id == user_id
    ).first()
    if not item:
        return JSONResponse(
            status_code=404,
            content={"success": False, "message": "自选项不存在"}
        )

    position = Position(
        watchlist_id=req.watchlist_id,
        action=req.action,
        price=req.price,
        shares=req.shares,
        action_date=datetime.strptime(req.action_date, "%Y-%m-%d"),
        note=req.note
    )
    db.add(position)
    db.commit()
    db.refresh(position)

    return {
        "success": True,
        "data": {
            "id": position.id,
            "action": position.action,
            "price": position.price,
            "shares": position.shares,
            "action_date": position.action_date.strftime("%Y-%m-%d")
        }
    }


@router.delete("/position/{position_id}")
def delete_position(
    position_id: int,
    authorization: str = Header(None),
    db: Session = Depends(get_db)
):
    try:
        user_id = get_current_user_id(authorization, db)
    except ValueError as e:
        return JSONResponse(
            status_code=401,
            content={"success": False, "message": str(e)}
        )

    position = db.query(Position).filter(
        Position.id == position_id
    ).first()

    if not position:
        return JSONResponse(
            status_code=404,
            content={"success": False, "message": "不存在"}
        )

    # 验证归属
    item = db.query(WatchlistItem).filter(
        WatchlistItem.id == position.watchlist_id,
        WatchlistItem.user_id == user_id
    ).first()
    if not item:
        return JSONResponse(
            status_code=403,
            content={"success": False, "message": "无权限"}
        )

    db.delete(position)
    db.commit()

    return {"success": True}