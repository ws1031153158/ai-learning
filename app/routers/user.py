# app/routers/user.py
from fastapi import APIRouter, Depends, Header
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from app.models.database import get_db, User
from app.services.auth_service import AuthService

router = APIRouter(prefix="/user", tags=["user"])


# ── 工具函数：从 Token 获取用户 ───────────────────────

def get_current_user_id(authorization: str, db: Session) -> int:
    if not authorization or not authorization.startswith("Bearer "):
        raise ValueError("未登录")
    token = authorization.replace("Bearer ", "")
    user = AuthService.get_current_user(db, token)
    if not user:
        raise ValueError("Token无效或已过期")
    return user.id


# ── 修改昵称 ──────────────────────────────────────────

@router.put("/nickname")
def update_nickname(
    req: dict,
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

    nickname = req.get("nickname", "").strip()
    if not nickname:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "昵称不能为空"}
        )
    if len(nickname) > 20:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "昵称不能超过20个字符"}
        )

    user = db.query(User).filter(User.id == user_id).first()
    user.nickname = nickname
    db.commit()

    return {"success": True, "message": "昵称修改成功"}


# ── 绑定邮箱 ──────────────────────────────────────────

@router.put("/email")
def update_email(
    req: dict,
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

    email = req.get("email", "").strip()
    if not email:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "邮箱不能为空"}
        )
    if "@" not in email:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "邮箱格式不正确"}
        )

    existing = db.query(User).filter(
        User.email == email,
        User.id != user_id
    ).first()
    if existing:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "该邮箱已被其他账号绑定"}
        )

    user = db.query(User).filter(User.id == user_id).first()
    user.email = email
    db.commit()

    return {"success": True, "message": "邮箱绑定成功"}


# ── 绑定手机 ──────────────────────────────────────────

@router.put("/phone")
def update_phone(
    req: dict,
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

    phone = req.get("phone", "").strip()
    if not phone:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "手机号不能为空"}
        )
    if len(phone) != 11 or not phone.isdigit():
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "手机号格式不正确"}
        )

    existing = db.query(User).filter(
        User.phone == phone,
        User.id != user_id
    ).first()
    if existing:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "该手机号已被其他账号绑定"}
        )

    user = db.query(User).filter(User.id == user_id).first()
    user.phone = phone
    db.commit()

    return {"success": True, "message": "手机号绑定成功"}


# ── 修改密码 ──────────────────────────────────────────

@router.put("/password")
def update_password(
    req: dict,
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

    old_password = req.get("old_password", "")
    new_password = req.get("new_password", "")

    if not old_password or not new_password:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "请填写完整信息"}
        )
    if len(new_password) < 6:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "新密码至少6位"}
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not AuthService.verify_password(old_password, user.password_hash):
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "原密码错误"}
        )

    user.password_hash = AuthService.hash_password(new_password)
    db.commit()

    return {"success": True, "message": "密码修改成功"}