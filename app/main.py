# app/main.py
import sys
sys.path.append(".")

from fastapi import FastAPI
from contextlib import asynccontextmanager
from datetime import datetime
import uvicorn

from app.config import DEFAULT_WATCH_LIST
from app.services.rag_service import RAGService
from app.services.memory_service import MemoryService
from app.services.agent_service import AgentService
from app.routers import rag, agent, data
import app.state as state          # ← 导入状态容器


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("\n🚀 理财AI分析师启动中...")

    # 初始化服务，存入 state
    state.rag_service = RAGService()
    state.memory_service = MemoryService()
    state.agent_service = AgentService(state.memory_service)

    print("⏳ 更新知识库...")
    from practice.rag.finance_server.data_pipline import update_knowledge_base
    update_knowledge_base(
        DEFAULT_WATCH_LIST,
        state.rag_service.embeddings
    )

    print("\n✅ 服务启动完成！")
    print("   API文档：http://localhost:8001/docs\n")

    yield

    print("👋 服务关闭")


app = FastAPI(
    title="理财AI分析师",
    description="RAG + Agent + 长期记忆 完整投资分析服务",
    version="3.0.0",
    lifespan=lifespan
)

app.include_router(rag.router)
app.include_router(agent.router)
app.include_router(data.router)


@app.get("/")
async def root():
    return {
        "service": "理财AI分析师 v3.0",
        "status": "running",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "docs": "http://localhost:8001/docs"
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )