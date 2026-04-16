# app/models/database.py
from sqlalchemy import (
    create_engine, Column, Integer, String,
    Float, DateTime, Boolean, Text, ForeignKey, Enum
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import enum

from app.config import DB_URL

engine = create_engine(DB_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── 用户表 ────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(100), unique=True, nullable=True)
    phone = Column(String(20), unique=True, nullable=True)
    password_hash = Column(String(255), nullable=False)
    nickname = Column(String(50), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    preference = relationship("UserPreference", back_populates="user", uselist=False)
    watchlist = relationship("WatchlistItem", back_populates="user")


# ── 投资偏好表 ─────────────────────────────────────────

class UserPreference(Base):
    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    risk_level = Column(String(10), default="medium")   # low/medium/high
    investment_style = Column(String(20), default="value")  # value/growth/trend
    holding_period = Column(String(10), default="medium")   # short/medium/long
    total_assets = Column(Float, nullable=True)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    user = relationship("User", back_populates="preference")


# ── 自选内容表 ─────────────────────────────────────────

class WatchlistItem(Base):
    __tablename__ = "watchlist"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    code = Column(String(20), nullable=False)
    name = Column(String(50), nullable=False)
    type = Column(String(10), nullable=False)    # stock/fund/bond
    data_range = Column(Integer, default=7)      # 3/7/14/30
    created_at = Column(DateTime, default=datetime.now)

    user = relationship("User", back_populates="watchlist")
    positions = relationship("Position", back_populates="watchlist_item")


# ── 持仓记录表 ─────────────────────────────────────────

class Position(Base):
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    watchlist_id = Column(Integer, ForeignKey("watchlist.id"))
    action = Column(String(10), nullable=False)   # buy/sell
    price = Column(Float, nullable=False)
    shares = Column(Float, nullable=False)
    action_date = Column(DateTime, nullable=False)
    note = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now)

    watchlist_item = relationship("WatchlistItem", back_populates="positions")