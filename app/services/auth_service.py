# app/services/auth_service.py
import bcrypt
import jwt
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.models.database import User, UserPreference
from app.config import JWT_SECRET, JWT_EXPIRE_DAYS, JWT_EXPIRE_DAYS_LONG


class AuthService:

    # ── 密码 ──────────────────────────────────────────

    @staticmethod
    def hash_password(password: str) -> str:
        return bcrypt.hashpw(
            password.encode("utf-8"),
            bcrypt.gensalt()
        ).decode("utf-8")

    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        return bcrypt.checkpw(
            password.encode("utf-8"),
            hashed.encode("utf-8")
        )

    # ── Token ─────────────────────────────────────────

    @staticmethod
    def create_token(user_id: int, keep_login: bool = False) -> str:
        expire_days = JWT_EXPIRE_DAYS_LONG if keep_login else JWT_EXPIRE_DAYS
        payload = {
            "user_id": user_id,
            "exp": datetime.utcnow() + timedelta(days=expire_days)
        }
        return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

    @staticmethod
    def verify_token(token: str) -> int | None:
        try:
            payload = jwt.decode(
                token, JWT_SECRET, algorithms=["HS256"]
            )
            return payload.get("user_id")
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    # ── 注册 ──────────────────────────────────────────

    @staticmethod
    def register(
        db: Session,
        password: str,
        email: str = None,
        phone: str = None,
        nickname: str = None
    ) -> User:
        # 检查邮箱是否已注册
        if email:
            existing = db.query(User).filter(
                User.email == email
            ).first()
            if existing:
                raise ValueError("邮箱已被注册")

        # 检查手机号是否已注册
        if phone:
            existing = db.query(User).filter(
                User.phone == phone
            ).first()
            if existing:
                raise ValueError("手机号已被注册")

        if not email and not phone:
            raise ValueError("邮箱和手机号至少填一个")

        # 创建用户
        user = User(
            email=email,
            phone=phone,
            password_hash=AuthService.hash_password(password),
            nickname=nickname or (email.split("@")[0] if email else phone)
        )
        db.add(user)
        db.flush()

        # 创建默认偏好
        preference = UserPreference(user_id=user.id)
        db.add(preference)
        db.commit()
        db.refresh(user)
        return user

    # ── 登录 ──────────────────────────────────────────

    @staticmethod
    def login(
        db: Session,
        password: str,
        email: str = None,
        phone: str = None,
        keep_login: bool = False
    ) -> dict:
        # 查找用户
        user = None
        if email:
            user = db.query(User).filter(
                User.email == email
            ).first()
        elif phone:
            user = db.query(User).filter(
                User.phone == phone
            ).first()

        if not user:
            raise ValueError("用户不存在")

        if not user.is_active:
            raise ValueError("账号已被禁用")

        if not AuthService.verify_password(
            password, user.password_hash
        ):
            raise ValueError("密码错误")

        token = AuthService.create_token(user.id, keep_login)
        return {
            "token": token,
            "user_id": user.id,
            "nickname": user.nickname,
            "email": user.email,
            "phone": user.phone
        }

    # ── 获取当前用户 ──────────────────────────────────

    @staticmethod
    def get_current_user(
        db: Session,
        token: str
    ) -> User | None:
        user_id = AuthService.verify_token(token)
        if not user_id:
            return None
        return db.query(User).filter(
            User.id == user_id
        ).first()