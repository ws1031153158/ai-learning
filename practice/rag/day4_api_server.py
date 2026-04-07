# rag/day4_api_server.py
import sys
sys.path.append(".")

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import asyncio
import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(override=True)

# ============================================================
# 初始化 FastAPI
# ============================================================

app = FastAPI(
    title="理财AI分析师 API",
    description="基于RAG的股票投资分析接口",
    version="0.1.0"
)

# ============================================================
# 初始化AI组件（启动时加载一次，不要每次请求都重新加载）
# ============================================================

print("⏳ 正在初始化AI组件...")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5"
)

llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com",
    temperature=0.3
)

llm_stream = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com",
    temperature=0.3,
    streaming=True  # 流式输出版本
)

# 加载或创建向量数据库
DB_PATH = "./rag/chroma_day3"

if os.path.exists(DB_PATH):
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )
    print("✅ 已加载现有知识库")
else:
    # 如果没有现有数据库，创建一个基础的
    sample_docs = [
        Document(
            page_content="贵州茅台2024年三季报营业收入1144亿元，净利润571亿元，市盈率约28倍",
            metadata={"type": "financial", "stock": "600519"}
        ),
        Document(
            page_content="比亚迪2024年11月销量50.68万辆，同比增长67.87%",
            metadata={"type": "sales", "stock": "002594"}
        ),
    ]
    vectorstore = Chroma.from_documents(
        documents=sample_docs,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    print("✅ 已创建新知识库")

print("✅ AI组件初始化完成\n")


# ============================================================
# 数据模型定义（请求和响应的格式）
# ============================================================

class QuestionRequest(BaseModel):
    """问答请求"""
    question: str                    # 用户问题
    stock_code: Optional[str] = None # 股票代码（可选）
    k: Optional[int] = 3            # 检索数量

class AddDocumentRequest(BaseModel):
    """添加文档请求"""
    content: str                     # 文档内容
    stock_code: Optional[str] = None
    doc_type: Optional[str] = "news"

class AnalysisRequest(BaseModel):
    """股票分析请求"""
    stock_code: str                  # 股票代码
    analysis_type: str = "comprehensive"  # 分析类型


# ============================================================
# 接口1：健康检查
# ============================================================

@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "理财AI分析师",
        "version": "0.1.0"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# ============================================================
# 接口2：问答接口（核心接口）
# ============================================================

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """
    基于知识库回答投资问题

    请求示例：
    {
        "question": "茅台的市盈率是多少？",
        "stock_code": "600519",
        "k": 3
    }
    """
    try:
        # 构建检索过滤条件
        search_kwargs = {"k": request.k}
        if request.stock_code:
            search_kwargs["filter"] = {"stock": request.stock_code}

        # 检索相关文档
        results = vectorstore.similarity_search(
            request.question,
            **search_kwargs
        )

        if not results:
            return {
                "answer": "暂无相关数据，请先添加相关股票信息",
                "sources": [],
                "question": request.question
            }

        # 拼接上下文
        context = "\n\n".join([
            f"[{r.metadata.get('type', '未知')}] {r.page_content}"
            for r in results
        ])

        # 构建Prompt
        prompt = f"""你是一个专业的投资分析助手。
请根据以下参考资料回答问题。
如果资料中没有相关信息，直接说"暂无相关数据"，不要编造任何数据。
回答要专业、客观，适当提示投资风险。

参考资料：
{context}

问题：{request.question}

专业分析："""

        response = llm.invoke([HumanMessage(content=prompt)])

        return {
            "answer": response.content,
            "sources": [r.page_content[:100] for r in results],
            "question": request.question,
            "retrieved_count": len(results)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 接口3：流式问答接口（打字机效果）
# ============================================================

@app.post("/ask/stream")
async def ask_stream(request: QuestionRequest):
    """
    流式输出版本，实现打字机效果
    Android端用SSE接收
    """

    async def generate():
        try:
            # 检索
            results = vectorstore.similarity_search(
                request.question,
                k=request.k
            )

            context = "\n\n".join([r.page_content for r in results])

            prompt = f"""你是专业投资分析助手。
根据以下资料回答问题，无相关信息说"暂无数据"。
适当提示投资风险。

资料：{context}

问题：{request.question}

分析："""

            # 流式输出
            async for chunk in llm_stream.astream(
                [HumanMessage(content=prompt)]
            ):
                if chunk.content:
                    # SSE格式：data: 内容\n\n
                    yield f"data: {chunk.content}\n\n"

            # 结束标记
            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


# ============================================================
# 接口4：添加文档到知识库
# ============================================================

@app.post("/documents/add")
async def add_document(request: AddDocumentRequest):
    """
    向知识库添加新文档
    用于实时更新财经新闻
    """
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=30
        )

        doc = Document(
            page_content=request.content,
            metadata={
                "type": request.doc_type,
                "stock": request.stock_code or "general"
            }
        )

        chunks = splitter.split_documents([doc])
        vectorstore.add_documents(chunks)

        return {
            "status": "success",
            "message": f"成功添加 {len(chunks)} 个文本块",
            "chunks_count": len(chunks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 接口5：股票综合分析
# ============================================================

@app.post("/analyze")
async def analyze_stock(request: AnalysisRequest):
    """
    对指定股票进行综合分析
    """
    try:
        # 针对同一股票问多个维度的问题
        questions = [
            f"{request.stock_code} 的最新财务数据",
            f"{request.stock_code} 的最新新闻动态",
            f"{request.stock_code} 的投资价值分析",
        ]

        all_context = []
        for q in questions:
            results = vectorstore.similarity_search(q, k=2)
            for r in results:
                if r.page_content not in all_context:
                    all_context.append(r.page_content)

        if not all_context:
            return {
                "stock_code": request.stock_code,
                "analysis": "暂无该股票相关数据，请先添加相关信息",
            }

        context = "\n\n".join(all_context)

        prompt = f"""你是专业的股票分析师。
请对以下股票进行综合分析，包含以下维度：

股票代码：{request.stock_code}

参考资料：
{context}

请按以下格式输出分析报告：

## 基本面分析
（财务数据、盈利能力）

## 市场表现
（近期走势、成交量）

## 风险提示
（主要风险点）

## 综合评级
（看多/中性/看空，并说明理由）

注意：以上仅为分析参考，不构成投资建议，投资有风险。"""

        response = llm.invoke([HumanMessage(content=prompt)])

        return {
            "stock_code": request.stock_code,
            "analysis": response.content,
            "data_sources_count": len(all_context)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 启动服务
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )