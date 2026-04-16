# app/routers/auth.py
from fastapi import APIRouter, Depends, Header
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from app.models.database import get_db
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])


# ── 请求模型 ──────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: Optional[str] = None
    phone: Optional[str] = None
    password: str
    nickname: Optional[str] = None


class LoginRequest(BaseModel):
    email: Optional[str] = None
    phone: Optional[str] = None
    password: str
    keep_login: bool = False


# ── 接口 ──────────────────────────────────────────────

@router.post("/register")
def register(
    req: RegisterRequest,
    db: Session = Depends(get_db)
):
    try:
        user = AuthService.register(
            db=db,
            email=req.email,
            phone=req.phone,
            password=req.password,
            nickname=req.nickname
        )
        token = AuthService.create_token(user.id)
        return {
            "success": True,
            "token": token,
            "user_id": user.id,
            "nickname": user.nickname
        }
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": str(e)}
        )


@router.post("/login")
def login(
    req: LoginRequest,
    db: Session = Depends(get_db)
):
    try:
        result = AuthService.login(
            db=db,
            email=req.email,
            phone=req.phone,
            password=req.password,
            keep_login=req.keep_login
        )
        return {"success": True, **result}
    except ValueError as e:
        return JSONResponse(
            status_code=401,
            content={"success": False, "message": str(e)}
        )


@router.post("/logout")
def logout():
    # JWT 无状态，客户端删除 token 即可
    return {"success": True, "message": "已退出登录"}


@router.get("/me")
def get_me(
    authorization: str = Header(None),
    db: Session = Depends(get_db)
):
    if not authorization or not authorization.startswith("Bearer "):
        return JSONResponse(
            status_code=401,
            content={"success": False, "message": "未登录"}
        )
    token = authorization.replace("Bearer ", "")
    user = AuthService.get_current_user(db, token)
    if not user:
        return JSONResponse(
            status_code=401,
            content={"success": False, "message": "Token无效或已过期"}
        )
    return {
        "success": True,
        "user_id": user.id,
        "nickname": user.nickname,
        "email": user.email,
        "phone": user.phone,
        "created_at": user.created_at.strftime("%Y-%m-%d")
    }