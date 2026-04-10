# app/services/rag_service.py
import sys
sys.path.append(".")

from datetime import datetime
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from app.config import *


class RAGService:

    def __init__(self):
        print("⏳ 初始化RAG服务...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )
        self.vectorstore = Chroma(
            persist_directory=RAG_DB_PATH,
            embedding_function=self.embeddings
        )
        self.llm = ChatOpenAI(
            model=DEEPSEEK_MODEL,
            openai_api_key=DEEPSEEK_API_KEY,
            openai_api_base=DEEPSEEK_BASE_URL,
            temperature=0.3
        )
        self.llm_stream = ChatOpenAI(
            model=DEEPSEEK_MODEL,
            openai_api_key=DEEPSEEK_API_KEY,
            openai_api_base=DEEPSEEK_BASE_URL,
            temperature=0.3,
            streaming=True
        )
        print("✅ RAG服务初始化完成")

    def retrieve(
        self,
        question: str,
        stock_code: str = None,
        k: int = 4
    ) -> list:
        """多类型检索，确保各维度数据都能检索到"""
        all_results = []
        seen = set()

        if stock_code:
            for data_type in [
                "price_analysis", "news",
                "fund_flow", "financial"
            ]:
                try:
                    results = self.vectorstore.similarity_search(
                        question, k=2,
                        filter={
                            "stock_code": stock_code,
                            "type": data_type
                        }
                    )
                    for r in results:
                        if r.page_content not in seen:
                            seen.add(r.page_content)
                            all_results.append(r)
                except Exception:
                    pass

        semantic = self.vectorstore.similarity_search(
            question, k=k
        )
        for r in semantic:
            if r.page_content not in seen:
                seen.add(r.page_content)
                all_results.append(r)

        return all_results[:k * 2]

    def ask(
        self,
        question: str,
        stock_code: str = None
    ) -> dict:
        results = self.retrieve(question, stock_code)
        if not results:
            return {
                "answer": "暂无相关数据",
                "sources": [],
                "retrieved_count": 0
            }

        context = "\n\n".join([
            f"[{r.metadata.get('type','未知')}]\n{r.page_content}"
            for r in results
        ])

        prompt = f"""你是专业投资分析助手。
根据以下资料回答问题，无数据说"暂无数据"，不编造。
回答专业客观，适当提示风险。

资料：
{context}

问题：{question}

分析："""

        response = self.llm.invoke(
            [HumanMessage(content=prompt)]
        )
        return {
            "answer": response.content,
            "sources": [
                {
                    "type": r.metadata.get("type"),
                    "content": r.page_content[:100],
                    "update_time": r.metadata.get("update_time")
                }
                for r in results
            ],
            "retrieved_count": len(results)
        }

    async def ask_stream(
        self,
        question: str,
        stock_code: str = None
    ):
        results = self.retrieve(question, stock_code)
        context = "\n\n".join([r.page_content for r in results])

        prompt = f"""你是专业投资分析助手。
根据资料回答问题，无数据说"暂无数据"，适当提示风险。

资料：{context}
问题：{question}
分析："""

        async for chunk in self.llm_stream.astream(
            [HumanMessage(content=prompt)]
        ):
            if chunk.content:
                yield chunk.content