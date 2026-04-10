# server/main.py
import sys
sys.path.append(".")

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from contextlib import asynccontextmanager
import uvicorn

from practice.server.agent_service import (
    chat_with_agent,
    run_crew_analysis,
    session_manager
)
from practice.rag.finance_server.data_pipline import update_knowledge_base
from practice.rag.finance_server.rag_engine import RAGEngine
from practice.server.config import DEFAULT_WATCH_LIST


# ============================================================
# 启动初始化
# ============================================================

rag_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_engine
    print("\n🚀 理财AI分析师服务启动中...")

    rag_engine = RAGEngine()

    print("⏳ 初始化知识库...")
    update_knowledge_base(DEFAULT_WATCH_LIST, rag_engine.embeddings)

    print("✅ 服务启动完成！")
    print(f"   API文档：http://localhost:8000/docs\n")

    yield
    print("👋 服务关闭")


app = FastAPI(
    title="理财AI分析师",
    description="RAG + Agent 完整投资分析服务",
    version="2.0.0",
    lifespan=lifespan
)


# ============================================================
# 数据模型
# ============================================================

class ChatRequest(BaseModel):
    session_id: str = "default"
    message: str

class AnalysisRequest(BaseModel):
    stock_code: str
    total_assets: Optional[float] = None

class RAGRequest(BaseModel):
    question: str
    stock_code: Optional[str] = None

class ClearRequest(BaseModel):
    session_id: str


# ============================================================
# 基础接口
# ============================================================

@app.get("/")
async def root():
    return {
        "service": "理财AI分析师 v2.0",
        "status": "running",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "endpoints": {
            "对话Agent": "POST /agent/chat",
            "多Agent分析": "POST /agent/analyze",
            "RAG问答": "POST /rag/ask",
            "RAG流式": "POST /rag/ask/stream",
            "更新数据": "POST /data/update"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}


# ============================================================
# Agent 接口
# ============================================================

@app.post("/agent/chat")
async def agent_chat(request: ChatRequest):
    """
    带记忆的对话Agent

    示例：
    {"session_id": "user_001", "message": "帮我分析茅台"}
    """
    try:
        response = await chat_with_agent(
            session_id=request.session_id,
            user_message=request.message
        )
        return {
            "session_id": request.session_id,
            "response": response,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/analyze")
async def agent_analyze(request: AnalysisRequest):
    """
    多Agent综合分析（耗时较长，约1~3分钟）

    示例：
    {"stock_code": "600519", "total_assets": 500000}
    """
    try:
        result = await run_crew_analysis(
            stock_code=request.stock_code,
            total_assets=request.total_assets
        )
        return {
            "stock_code": request.stock_code,
            "report": result,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/clear")
async def clear_session(request: ClearRequest):
    """清空对话历史"""
    session_manager.clear(request.session_id)
    return {
        "status": "success",
        "message": f"会话 {request.session_id} 已清空"
    }


# ============================================================
# RAG 接口
# ============================================================

@app.post("/rag/ask")
async def rag_ask(request: RAGRequest):
    """RAG知识库问答"""
    try:
        result = rag_engine.ask(
            question=request.question,
            stock_code=request.stock_code
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/ask/stream")
async def rag_ask_stream(request: RAGRequest):
    """RAG流式问答（打字机效果）"""
    async def generate():
        try:
            async for chunk in rag_engine.ask_stream(
                question=request.question,
                stock_code=request.stock_code
            ):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR]{str(e)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"}
    )


# ============================================================
# 数据管理接口
# ============================================================

@app.post("/data/update")
async def update_data(stock_code: Optional[str] = None):
    """
    手动触发数据更新

    不传stock_code则更新所有默认股票
    """
    try:
        codes = [stock_code] if stock_code else DEFAULT_WATCH_LIST
        count = update_knowledge_base(codes, rag_engine.embeddings)
        return {
            "status": "success",
            "updated": codes,
            "chunks": count,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 启动
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )