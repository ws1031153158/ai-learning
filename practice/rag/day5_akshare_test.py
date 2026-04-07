# rag/day5_pipeline_final.py
import sys
sys.path.append(".")

import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv(override=True)

# ============================================================
# 初始化组件
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

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30
)

DB_PATH = "./rag/chroma_finance"

today = datetime.now().strftime("%Y%m%d")
thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")


# ============================================================
# 数据源1：历史行情 → 生成技术分析文本
# ============================================================

def fetch_price_analysis(stock_code: str) -> list[Document]:
    """
    获取历史行情，计算技术指标，生成分析文本
    """
    print(f"  📈 获取 {stock_code} 行情数据...")
    docs = []

    try:
        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date=thirty_days_ago,
            end_date=today,
            adjust="qfq"
        )

        if df.empty:
            return docs

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # 计算技术指标
        ma5 = df['收盘'].tail(5).mean()
        ma10 = df['收盘'].tail(10).mean()
        ma20 = df['收盘'].tail(20).mean() if len(df) >= 20 else None

        month_start_price = df.iloc[0]['收盘']
        month_change = (latest['收盘'] - month_start_price) / month_start_price * 100

        max_price = df['最高'].max()
        min_price = df['最低'].min()
        avg_volume = df['成交量'].mean()
        latest_volume_ratio = latest['成交量'] / avg_volume

        # 判断趋势
        trend = "上涨" if ma5 > ma10 else "下跌"
        volume_status = "放量" if latest_volume_ratio > 1.5 else (
            "缩量" if latest_volume_ratio < 0.7 else "正常"
        )

        price_text = f"""【{stock_code}行情分析】
更新时间：{datetime.now().strftime("%Y-%m-%d %H:%M")}
最新收盘价：{latest['收盘']}元
今日涨跌幅：{latest['涨跌幅']}%
今日成交量：{latest['成交量']}手（{volume_status}）
近30日涨跌幅：{month_change:.2f}%
近30日最高价：{max_price}元
近30日最低价：{min_price}元
5日均线：{ma5:.2f}元
10日均线：{ma10:.2f}元
20日均线：{f"{ma20:.2f}元" if ma20 else "数据不足"}
短期趋势：{trend}（MA5{">" if ma5 > ma10 else "<"}MA10）
最近5日收盘：{df['收盘'].tail(5).tolist()}"""

        docs.append(Document(
            page_content=price_text,
            metadata={
                "stock_code": stock_code,
                "type": "price_analysis",
                "update_time": datetime.now().strftime("%Y-%m-%d %H:%M")
            }
        ))

        print(f"  ✅ 行情分析生成成功")

    except Exception as e:
        print(f"  ❌ 行情数据获取失败：{e}")

    return docs


# ============================================================
# 数据源2：个股新闻 → 舆情分析
# ============================================================

def fetch_news(stock_code: str, limit: int = 10) -> list[Document]:
    """
    获取个股新闻
    """
    print(f"  📰 获取 {stock_code} 新闻...")
    docs = []

    try:
        news_df = ak.stock_news_em(symbol=stock_code)
        news_df = news_df.head(limit)

        for _, row in news_df.iterrows():
            title = row.get('新闻标题', '')
            content = row.get('新闻内容', '')
            pub_time = str(row.get('发布时间', ''))

            # 内容太长就截断
            if len(content) > 400:
                content = content[:400] + "..."

            news_text = f"""【新闻】{title}
时间：{pub_time}
内容：{content}"""

            docs.append(Document(
                page_content=news_text,
                metadata={
                    "stock_code": stock_code,
                    "type": "news",
                    "publish_time": pub_time,
                    "update_time": datetime.now().strftime("%Y-%m-%d %H:%M")
                }
            ))

        print(f"  ✅ 获取到 {len(docs)} 条新闻")

    except Exception as e:
        print(f"  ❌ 新闻获取失败：{e}")

    return docs


# ============================================================
# 数据源3：资金流向 → 主力动向分析
# ============================================================

def fetch_fund_flow(stock_code: str) -> list[Document]:
    """
    获取主力资金流向
    """
    print(f"  💰 获取 {stock_code} 资金流向...")
    docs = []

    try:
        # 判断市场（沪市sh，深市sz）
        market = "sh" if stock_code.startswith("6") else "sz"

        df = ak.stock_individual_fund_flow(
            stock=stock_code,
            market=market
        )

        # 取最近5天
        df = df.tail(5)

        fund_text = f"""【{stock_code}资金流向分析】
更新时间：{datetime.now().strftime("%Y-%m-%d %H:%M")}
"""
        for _, row in df.iterrows():
            main_flow = row.get('主力净流入-净额', 0)
            flow_ratio = row.get('主力净流入-净占比', 0)
            flow_status = "净流入" if main_flow > 0 else "净流出"

            fund_text += f"""
日期：{row.get('日期', '')}
收盘价：{row.get('收盘价', '')}元 涨跌：{row.get('涨跌幅', '')}%
主力资金：{flow_status} {abs(main_flow/1e8):.2f}亿元（占比{flow_ratio}%）
超大单：{row.get('超大单净流入-净额', 0)/1e8:.2f}亿
大单：{row.get('大单净流入-净额', 0)/1e8:.2f}亿"""

        docs.append(Document(
            page_content=fund_text,
            metadata={
                "stock_code": stock_code,
                "type": "fund_flow",
                "update_time": datetime.now().strftime("%Y-%m-%d %H:%M")
            }
        ))

        print(f"  ✅ 资金流向获取成功")

    except Exception as e:
        print(f"  ❌ 资金流向获取失败：{e}")

    return docs


# ============================================================
# 数据源4：财务指标 → 基本面分析
# ============================================================

def fetch_financial(stock_code: str) -> list[Document]:
    """
    获取财务指标
    """
    print(f"  📊 获取 {stock_code} 财务指标...")
    docs = []

    try:
        df = ak.stock_financial_analysis_indicator(
            symbol=stock_code,
            start_year="2024"
        )

        if df.empty:
            return docs

        # 取最新一期
        latest = df.iloc[0]

        financial_text = f"""【{stock_code}财务指标】
更新时间：{datetime.now().strftime("%Y-%m-%d %H:%M")}
报告期：{latest.get('日期', 'N/A')}
每股收益(EPS)：{latest.get('摊薄每股收益(元)', 'N/A')}元
每股净资产：{latest.get('每股净资产_调整后(元)', 'N/A')}元
净资产收益率(ROE)：{latest.get('净资产收益率(%)', 'N/A')}%
总资产收益率(ROA)：{latest.get('总资产净利润率(%)', 'N/A')}%
营业利润率：{latest.get('营业利润率(%)', 'N/A')}%
销售净利率：{latest.get('销售净利率(%)', 'N/A')}%
主营收入增长率：{latest.get('主营业务收入增长率(%)', 'N/A')}%
净利润增长率：{latest.get('净利润增长率(%)', 'N/A')}%
资产负债率：{latest.get('资产负债率(%)', 'N/A')}%
流动比率：{latest.get('流动比率', 'N/A')}"""

        docs.append(Document(
            page_content=financial_text,
            metadata={
                "stock_code": stock_code,
                "type": "financial",
                "update_time": datetime.now().strftime("%Y-%m-%d %H:%M")
            }
        ))

        print(f"  ✅ 财务指标获取成功")

    except Exception as e:
        print(f"  ❌ 财务指标获取失败：{e}")

    return docs


# ============================================================
# 整合：更新知识库
# ============================================================

def update_knowledge_base(stock_codes: list[str]) -> int:
    """
    更新指定股票的知识库
    """
    print(f"\n{'='*50}")
    print(f"开始更新知识库")
    print(f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"股票：{stock_codes}")
    print(f"{'='*50}")

    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    all_docs = []

    for code in stock_codes:
        print(f"\n处理股票：{code}")
        all_docs.extend(fetch_price_analysis(code))
        all_docs.extend(fetch_news(code))
        all_docs.extend(fetch_fund_flow(code))
        all_docs.extend(fetch_financial(code))

    if not all_docs:
        print("❌ 没有获取到任何数据")
        return 0

    chunks = splitter.split_documents(all_docs)
    vectorstore.add_documents(chunks)

    print(f"\n✅ 知识库更新完成，共 {len(chunks)} 个文本块")
    return len(chunks)


# ============================================================
# 测试问答效果
# ============================================================

def test_qa(stock_codes: list[str]):
    """
    测试RAG问答效果
    """
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    questions = [
        f"{stock_codes[0]}今天涨跌情况怎么样？",
        f"{stock_codes[0]}最近主力资金是流入还是流出？",
        f"{stock_codes[0]}最近有什么重要新闻？",
        f"{stock_codes[0]}的财务状况怎么样？",
    ]

    print(f"\n{'='*50}")
    print("RAG问答测试")
    print(f"{'='*50}")

    for question in questions:
        print(f"\n问：{question}")

        results = vectorstore.similarity_search(question, k=3)
        context = "\n\n".join([
            f"[{r.metadata.get('type')}]\n{r.page_content}"
            for r in results
        ])

        prompt = f"""你是专业投资分析助手。
根据以下资料回答问题，无相关数据直接说"暂无数据"。
回答简洁专业，适当提示风险。

资料：
{context}

问题：{question}

分析："""

        response = llm.invoke([HumanMessage(content=prompt)])
        print(f"答：{response.content}")
        print("-" * 40)


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":

    WATCH_LIST = ["600519", "002594", "300750"]

    # 更新知识库
    count = update_knowledge_base(WATCH_LIST)

    if count > 0:
        # 测试问答
        test_qa(WATCH_LIST)