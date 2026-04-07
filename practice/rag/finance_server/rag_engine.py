# rag/finance_server/rag_engine.py
import sys
sys.path.append(".")

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from practice.rag.finance_server.config import *
from datetime import datetime


class RAGEngine:
    """
    RAG核心引擎
    修复Day5的检索问题：
    用 stock_code 过滤 + 多类型检索
    """

    def __init__(self):
        print("⏳ 初始化RAG引擎...")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        self.vectorstore = Chroma(
            persist_directory=DB_PATH,
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

        print("✅ RAG引擎初始化完成")

    def retrieve(
        self,
        question: str,
        stock_code: str = None,
        k: int = 4
    ) -> list:
        """
        修复版检索：
        同时用语义检索 + 类型过滤
        确保各类数据都能被检索到
        """
        all_results = []
        seen = set()

        if stock_code:
            # 按类型分别检索，确保每种数据都有覆盖
            data_types = [
                "price_analysis",
                "news",
                "fund_flow",
                "financial"
            ]

            for data_type in data_types:
                try:
                    results = self.vectorstore.similarity_search(
                        question,
                        k=2,
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

        # 补充语义检索结果
        semantic_results = self.vectorstore.similarity_search(
            question, k=k
        )
        for r in semantic_results:
            if r.page_content not in seen:
                seen.add(r.page_content)
                all_results.append(r)

        return all_results[:k * 2]

    def ask(
        self,
        question: str,
        stock_code: str = None,
        k: int = 4
    ) -> dict:
        """普通问答"""
        results = self.retrieve(question, stock_code, k)

        if not results:
            return {
                "answer": "暂无相关数据",
                "sources": [],
                "retrieved_count": 0
            }

        context = "\n\n".join([
            f"[{r.metadata.get('type', '未知')}]\n{r.page_content}"
            for r in results
        ])

        prompt = f"""你是专业投资分析助手。
根据以下资料回答问题。
无相关数据直接说"暂无数据"，不要编造。
回答专业客观，适当提示投资风险。

资料：
{context}

问题：{question}

分析："""

        response = self.llm.invoke([HumanMessage(content=prompt)])

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
        stock_code: str = None,
        k: int = 4
    ):
        """流式问答"""
        results = self.retrieve(question, stock_code, k)

        context = "\n\n".join([
            f"[{r.metadata.get('type', '未知')}]\n{r.page_content}"
            for r in results
        ])

        prompt = f"""你是专业投资分析助手。
根据以下资料回答问题，无数据说"暂无数据"。
回答专业客观，适当提示风险。

资料：
{context}

问题：{question}

分析："""

        async for chunk in self.llm_stream.astream(
            [HumanMessage(content=prompt)]
        ):
            if chunk.content:
                yield chunk.content

    def analyze_stock(self, stock_code: str) -> dict:
        """
        股票综合分析
        强制检索所有维度数据
        """
        results = self.retrieve(
            f"{stock_code}综合分析",
            stock_code=stock_code,
            k=8
        )

        if not results:
            return {
                "stock_code": stock_code,
                "analysis": "暂无该股票数据，请先更新知识库"
            }

        context = "\n\n".join([
            f"[{r.metadata.get('type')}]\n{r.page_content}"
            for r in results
        ])

        prompt = f"""你是专业股票分析师。
对股票 {stock_code} 进行综合分析。

参考资料：
{context}

请按以下格式输出：

## 📈 行情分析
（最新价格、涨跌幅、均线趋势）

## 💰 资金动向
（主力资金流入流出情况）

## 📰 近期动态
（重要新闻和公告）

## 📊 基本面
（核心财务指标）

## ⚖️ 综合评估
（投资机会与风险）

## ⚠️ 风险提示
以上仅供参考，不构成投资建议，投资有风险。"""

        response = self.llm.invoke([HumanMessage(content=prompt)])

        return {
            "stock_code": stock_code,
            "analysis": response.content,
            "data_count": len(results),
            "update_time": datetime.now().strftime("%Y-%m-%d %H:%M")
        }