# rag/finance_server/main.py
import sys
sys.path.append(".")

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from contextlib import asynccontextmanager
import uvicorn

from practice.rag.finance_server.config import DEFAULT_WATCH_LIST
from practice.rag.finance_server.data_pipline import update_knowledge_base
from practice.rag.finance_server.rag_engine import RAGEngine

# ============================================================
# 启动和关闭事件
# ============================================================

rag_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动时初始化"""
    global rag_engine

    print("\n🚀 理财AI分析师服务启动中...")

    # 初始化RAG引擎
    rag_engine = RAGEngine()

    # 启动时更新一次知识库
    print("⏳ 初始化知识库数据...")
    update_knowledge_base(
        DEFAULT_WATCH_LIST,
        rag_engine.embeddings
    )

    print("\n✅ 服务启动完成！")
    print(f"   监控股票：{DEFAULT_WATCH_LIST}")
    print(f"   API文档：http://localhost:8000/docs\n")

    yield

    print("👋 服务关闭")


# ============================================================
# 初始化 FastAPI
# ============================================================

app = FastAPI(
    title="理财AI分析师",
    description="基于RAG的专业股票投资分析服务",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================
# 数据模型
# ============================================================

class QuestionRequest(BaseModel):
    question: str
    stock_code: Optional[str] = None
    k: Optional[int] = 4

class AnalysisRequest(BaseModel):
    stock_code: str

class AddStockRequest(BaseModel):
    stock_code: str

class AddDocRequest(BaseModel):
    content: str
    stock_code: Optional[str] = None
    doc_type: Optional[str] = "manual"


# ============================================================
# 接口
# ============================================================

@app.get("/")
async def root():
    return {
        "service": "理财AI分析师",
        "version": "1.0.0",
        "status": "running",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "watch_list": DEFAULT_WATCH_LIST,
        "docs": "http://localhost:8000/docs"
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/ask")
async def ask(request: QuestionRequest):
    """
    智能问答接口

    示例：
    {"question": "茅台今天涨跌怎么样？", "stock_code": "600519"}
    """
    try:
        result = rag_engine.ask(
            question=request.question,
            stock_code=request.stock_code,
            k=request.k
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/stream")
async def ask_stream(request: QuestionRequest):
    """
    流式问答接口（打字机效果）
    Android端用SSE接收
    """
    async def generate():
        try:
            async for chunk in rag_engine.ask_stream(
                question=request.question,
                stock_code=request.stock_code,
                k=request.k
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


@app.post("/analyze")
async def analyze(request: AnalysisRequest):
    """
    股票综合分析接口

    示例：
    {"stock_code": "600519"}
    """
    try:
        result = rag_engine.analyze_stock(request.stock_code)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update")
async def update(request: AddStockRequest):
    """
    手动触发更新指定股票数据
    """
    try:
        count = update_knowledge_base(
            [request.stock_code],
            rag_engine.embeddings
        )
        return {
            "status": "success",
            "stock_code": request.stock_code,
            "chunks_added": count,
            "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update/all")
async def update_all():
    """
    更新所有监控股票数据
    """
    try:
        count = update_knowledge_base(
            DEFAULT_WATCH_LIST,
            rag_engine.embeddings
        )
        return {
            "status": "success",
            "updated_stocks": DEFAULT_WATCH_LIST,
            "chunks_added": count,
            "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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