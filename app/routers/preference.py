# app/routers/preference.py
from fastapi import APIRouter, Depends, Header
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional

from app.models.database import get_db, UserPreference
from app.services.auth_service import AuthService

router = APIRouter(prefix="/user", tags=["user"])


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

class PreferenceRequest(BaseModel):
    risk_level: Optional[str] = None       # low/medium/high
    investment_style: Optional[str] = None  # value/growth/trend
    holding_period: Optional[str] = None    # short/medium/long
    total_assets: Optional[float] = None


# ── 接口 ──────────────────────────────────────────────

@router.get("/preference")
def get_preference(
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

    pref = db.query(UserPreference).filter(
        UserPreference.user_id == user_id
    ).first()

    if not pref:
        return {
            "success": True,
            "data": {
                "risk_level": "medium",
                "investment_style": "value",
                "holding_period": "medium",
                "total_assets": None
            }
        }

    return {
        "success": True,
        "data": {
            "risk_level": pref.risk_level,
            "investment_style": pref.investment_style,
            "holding_period": pref.holding_period,
            "total_assets": pref.total_assets
        }
    }


@router.post("/preference")
def save_preference(
    req: PreferenceRequest,
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

    pref = db.query(UserPreference).filter(
        UserPreference.user_id == user_id
    ).first()

    if not pref:
        pref = UserPreference(user_id=user_id)
        db.add(pref)

    if req.risk_level is not None:
        pref.risk_level = req.risk_level
    if req.investment_style is not None:
        pref.investment_style = req.investment_style
    if req.holding_period is not None:
        pref.holding_period = req.holding_period
    if req.total_assets is not None:
        pref.total_assets = req.total_assets

    db.commit()
    db.refresh(pref)

    return {
        "success": True,
        "data": {
            "risk_level": pref.risk_level,
            "investment_style": pref.investment_style,
            "holding_period": pref.holding_period,
            "total_assets": pref.total_assets
        }
    }