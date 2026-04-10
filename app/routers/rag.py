# app/routers/rag.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import app.state as state          # ← 从state取服务实例

router = APIRouter(prefix="/rag", tags=["RAG知识库"])


class RAGRequest(BaseModel):
    question: str
    stock_code: Optional[str] = None


@router.post("/ask")
async def rag_ask(request: RAGRequest):
    try:
        return state.rag_service.ask(
            request.question,
            request.stock_code
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask/stream")
async def rag_ask_stream(request: RAGRequest):
    try:
        async def generate():
            async for chunk in state.rag_service.ask_stream(
                request.question,
                request.stock_code
            ):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))