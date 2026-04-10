# app/routers/agent.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import app.state as state          # ← 从state取服务实例

router = APIRouter(prefix="/agent", tags=["Agent对话"])


class ChatRequest(BaseModel):
    session_id: str = "default"
    message: str


class AnalysisRequest(BaseModel):
    stock_code: str
    total_assets: Optional[float] = None


class ClearRequest(BaseModel):
    session_id: str


@router.post("/chat")
async def agent_chat(request: ChatRequest):
    try:
        response = await state.agent_service.chat_async(
            session_id=request.session_id,
            user_input=request.message
        )
        return {
            "session_id": request.session_id,
            "response": response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze")
async def agent_analyze(request: AnalysisRequest):
    try:
        result = await state.agent_service.analyze_async(
            stock_code=request.stock_code,
            total_assets=request.total_assets
        )
        return {
            "stock_code": request.stock_code,
            "report": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear")
async def clear_session(request: ClearRequest):
    state.agent_service.session_manager.clear(request.session_id)
    return {"status": "success"}


@router.get("/session/{session_id}")
async def get_session_info(session_id: str):
    return state.agent_service.session_manager.get_info(session_id)


@router.get("/memory/{user_id}")
async def get_memories(user_id: str):
    memories = state.memory_service.get_all(user_id)
    return {
        "user_id": user_id,
        "count": len(memories),
        "memories": [m["memory"] for m in memories]
    }


@router.delete("/memory/{user_id}")
async def delete_memories(user_id: str):
    state.memory_service.delete_all(user_id)
    return {"status": "success", "user_id": user_id}