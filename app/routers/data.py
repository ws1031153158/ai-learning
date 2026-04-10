# app/routers/data.py
from fastapi import APIRouter, HTTPException
from typing import Optional
import app.state as state          # ← 从state取服务实例

router = APIRouter(prefix="/data", tags=["数据管理"])


@router.post("/update")
async def update_data(stock_code: Optional[str] = None):
    try:
        from app.config import DEFAULT_WATCH_LIST
        from practice.rag.finance_server.data_pipline import (
            update_knowledge_base
        )
        codes = [stock_code] if stock_code else DEFAULT_WATCH_LIST
        count = update_knowledge_base(
            codes,
            state.rag_service.embeddings
        )
        return {
            "status": "success",
            "updated": codes,
            "chunks": count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))