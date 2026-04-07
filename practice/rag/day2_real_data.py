# rag/day2_real_data.py
import sys
sys.path.append(".")

from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
    CSVLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
import akshare as ak
import pandas as pd
import os
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

# ============================================================
# 公共组件初始化
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5"
)

llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com",
    temperature=0.3
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30,
)

# ============================================================
# 数据源1：AKShare 获取真实股票新闻
# ============================================================

def load_stock_news(stock_code: str = "002594") -> list[Document]:
    """
    用AKShare获取股票相关新闻
    002594
    """
    print(f"\n⏳ 正在获取 {stock_code} 的新闻...")

    try:
        # 获取个股新闻
        news_df = ak.stock_news_em(symbol=stock_code)

        # 只取最新的10条
        news_df = news_df.head(10)

        documents = []
        for _, row in news_df.iterrows():
            content = f"标题：{row['新闻标题']}\n内容：{row['新闻内容']}"
            doc = Document(
                page_content=content,
                metadata={
                    "source": "eastmoney_news",
                    "stock_code": stock_code,
                    "date": str(row.get('发布时间', '')),
                    "type": "stock_news"
                }
            )
            documents.append(doc)

        print(f"✅ 获取到 {len(documents)} 条新闻")
        return documents

    except Exception as e:
        print(f"❌ 获取新闻失败：{e}")
        return []


# ============================================================
# 数据源2：AKShare 获取股票基本信息
# ============================================================

def load_stock_info(stock_code: str = "002594") -> list[Document]:
    """
    获取股票基本面数据
    """
    print(f"\n⏳ 正在获取 {stock_code} 基本信息...")

    try:
        documents = []

        # 获取股票基本信息
        stock_info = ak.stock_individual_info_em(symbol=stock_code)

        # 转成文字描述
        info_text = f"股票代码：{stock_code}\n"
        for _, row in stock_info.iterrows():
            info_text += f"{row['item']}：{row['value']}\n"

        doc = Document(
            page_content=info_text,
            metadata={
                "source": "stock_basic_info",
                "stock_code": stock_code,
                "type": "basic_info"
            }
        )
        documents.append(doc)

        print(f"✅ 基本信息获取成功")
        return documents

    except Exception as e:
        print(f"❌ 获取基本信息失败：{e}")
        return []


# ============================================================
# 数据源3：AKShare 获取财务数据
# ============================================================

def load_financial_data(stock_code: str = "002594") -> list[Document]:
    """
    获取股票财务指标
    """
    print(f"\n⏳ 正在获取 {stock_code} 财务数据...")

    try:
        # 获取主要财务指标
        financial_df = ak.stock_financial_abstract_ths(
            symbol=stock_code,
            indicator="按年度"
        )

        # 只取最近3年
        financial_df = financial_df.head(3)

        documents = []
        for _, row in financial_df.iterrows():
            content = f"财务数据（{row.get('报告期', '')}）：\n"
            for col in financial_df.columns:
                content += f"{col}：{row[col]}\n"

            doc = Document(
                page_content=content,
                metadata={
                    "source": "financial_data",
                    "stock_code": stock_code,
                    "type": "financial"
                }
            )
            documents.append(doc)

        print(f"✅ 获取到 {len(documents)} 期财务数据")
        return documents

    except Exception as e:
        print(f"❌ 获取财务数据失败：{e}")
        return []


# ============================================================
# 数据源4：加载本地PDF（如果你有研报PDF）
# ============================================================

def load_pdf(pdf_path: str) -> list[Document]:
    """
    加载PDF文件
    """
    if not os.path.exists(pdf_path):
        print(f"⚠️  PDF文件不存在：{pdf_path}，跳过")
        return []

    print(f"\n⏳ 正在加载PDF：{pdf_path}")

    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        print(f"✅ PDF加载成功，共 {len(documents)} 页")
        return documents

    except Exception as e:
        print(f"❌ PDF加载失败：{e}")
        return []


# ============================================================
# 整合所有数据，构建知识库
# ============================================================

def build_knowledge_base(stock_code: str = "002594"):
    """
    整合多个数据源，构建股票知识库
    """
    print(f"\n{'='*50}")
    print(f"开始构建 {stock_code} 知识库")
    print(f"{'='*50}")

    all_documents = []

    # 1. 获取新闻
    news_docs = load_stock_news(stock_code)
    all_documents.extend(news_docs)

    # 2. 获取基本信息
    info_docs = load_stock_info(stock_code)
    all_documents.extend(info_docs)

    # 3. 获取财务数据
    financial_docs = load_financial_data(stock_code)
    all_documents.extend(financial_docs)

    # 4. 加载PDF（如果有的话）
    pdf_docs = load_pdf("rag/sample.pdf")
    all_documents.extend(pdf_docs)

    if not all_documents:
        print("❌ 没有获取到任何数据")
        return None

    print(f"\n✅ 共收集到 {len(all_documents)} 个文档")

    # 切块
    chunks = text_splitter.split_documents(all_documents)
    print(f"✅ 切块完成，共 {len(chunks)} 个文本块")

    # 存入向量数据库
    print("\n⏳ 正在构建向量数据库...")
    db_path = f"./rag/chroma_{stock_code}"

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path
    )

    print(f"✅ 知识库构建完成，存储在：{db_path}")
    return vectorstore


# ============================================================
# 问答函数
# ============================================================

def ask(vectorstore, question: str):
    """
    基于知识库回答问题
    """
    print(f"\n问题：{question}")

    # 检索相关文段
    results = vectorstore.similarity_search(question, k=3)

    if not results:
        print("回答：暂无相关数据")
        return

    # 拼接上下文
    context = "\n\n".join([
        f"[来源：{r.metadata.get('type', '未知')}]\n{r.page_content}"
        for r in results
    ])

    # 构建Prompt
    prompt = f"""你是一个专业的投资分析助手。
请根据以下参考资料回答问题。
如果资料中没有相关信息，直接说"暂无相关数据"，不要编造。

参考资料：
{context}

问题：{question}

请给出专业、客观的分析："""

    response = llm.invoke([HumanMessage(content=prompt)])
    print(f"回答：{response.content}")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":

    # 构建知识库
    vectorstore = build_knowledge_base("002594")

    if vectorstore:
        # 测试几个问题
        questions = [
            "002594最近有什么重要新闻？",
            "002594的营业收入是多少？",
            "002594的市盈率是多少？",
        ]

        for q in questions:
            ask(vectorstore, q)
            print("-" * 50)