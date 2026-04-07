# rag/finance_server/data_pipeline.py
import sys
sys.path.append(".")

import akshare as ak
from datetime import datetime, timedelta
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from practice.rag.finance_server.config import *

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30
)

today = datetime.now().strftime("%Y%m%d")
thirty_days_ago = (
    datetime.now() - timedelta(days=HISTORY_DAYS)
).strftime("%Y%m%d")


# ============================================================
# 数据获取
# ============================================================

def fetch_price_analysis(stock_code: str) -> list[Document]:
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
        ma5 = df['收盘'].tail(5).mean()
        ma10 = df['收盘'].tail(10).mean()
        ma20 = df['收盘'].tail(20).mean() if len(df) >= 20 else None
        month_change = (
            (latest['收盘'] - df.iloc[0]['收盘'])
            / df.iloc[0]['收盘'] * 100
        )
        avg_volume = df['成交量'].mean()
        volume_ratio = latest['成交量'] / avg_volume
        trend = "上涨" if ma5 > ma10 else "下跌"
        volume_status = (
            "放量" if volume_ratio > 1.5
            else "缩量" if volume_ratio < 0.7
            else "正常"
        )

        text = f"""【{stock_code}行情分析】
更新时间：{datetime.now().strftime("%Y-%m-%d %H:%M")}
最新收盘价：{latest['收盘']}元
今日涨跌幅：{latest['涨跌幅']}%
今日成交量：{latest['成交量']}手（{volume_status}）
近30日涨跌幅：{month_change:.2f}%
近30日最高：{df['最高'].max()}元
近30日最低：{df['最低'].min()}元
5日均线：{ma5:.2f}元
10日均线：{ma10:.2f}元
20日均线：{f"{ma20:.2f}元" if ma20 else "数据不足"}
短期趋势：{trend}
最近5日收盘：{df['收盘'].tail(5).tolist()}"""

        docs.append(Document(
            page_content=text,
            metadata={
                "stock_code": stock_code,
                "type": "price_analysis",
                "update_time": datetime.now().strftime("%Y-%m-%d %H:%M")
            }
        ))
    except Exception as e:
        print(f"  ❌ 行情数据失败：{e}")
    return docs


def fetch_news(stock_code: str) -> list[Document]:
    docs = []
    try:
        df = ak.stock_news_em(symbol=stock_code)
        df = df.head(NEWS_LIMIT)

        for _, row in df.iterrows():
            title = row.get('新闻标题', '')
            content = row.get('新闻内容', '')
            pub_time = str(row.get('发布时间', ''))
            if len(content) > 400:
                content = content[:400] + "..."

            docs.append(Document(
                page_content=f"""【新闻】{title}
时间：{pub_time}
内容：{content}""",
                metadata={
                    "stock_code": stock_code,
                    "type": "news",
                    "publish_time": pub_time,
                    "update_time": datetime.now().strftime("%Y-%m-%d %H:%M")
                }
            ))
    except Exception as e:
        print(f"  ❌ 新闻获取失败：{e}")
    return docs


def fetch_fund_flow(stock_code: str) -> list[Document]:
    docs = []
    try:
        market = "sh" if stock_code.startswith("6") else "sz"
        df = ak.stock_individual_fund_flow(
            stock=stock_code,
            market=market
        )
        df = df.tail(FUND_FLOW_DAYS)

        text = f"""【{stock_code}资金流向】
更新时间：{datetime.now().strftime("%Y-%m-%d %H:%M")}\n"""

        for _, row in df.iterrows():
            main_flow = row.get('主力净流入-净额', 0)
            flow_ratio = row.get('主力净流入-净占比', 0)
            status = "净流入🟢" if main_flow > 0 else "净流出🔴"

            text += f"""
日期：{row.get('日期', '')} | 收盘：{row.get('收盘价', '')}元 | 涨跌：{row.get('涨跌幅', '')}%
主力：{status} {abs(main_flow/1e8):.2f}亿（占比{flow_ratio}%）
超大单：{row.get('超大单净流入-净额', 0)/1e8:.2f}亿 | 大单：{row.get('大单净流入-净额', 0)/1e8:.2f}亿"""

        docs.append(Document(
            page_content=text,
            metadata={
                "stock_code": stock_code,
                "type": "fund_flow",
                "update_time": datetime.now().strftime("%Y-%m-%d %H:%M")
            }
        ))
    except Exception as e:
        print(f"  ❌ 资金流向失败：{e}")
    return docs


def fetch_financial(stock_code: str) -> list[Document]:
    docs = []
    try:
        df = ak.stock_financial_analysis_indicator(
            symbol=stock_code,
            start_year="2024"
        )
        if df.empty:
            return docs

        latest = df.iloc[0]
        text = f"""【{stock_code}财务指标】
更新时间：{datetime.now().strftime("%Y-%m-%d %H:%M")}
报告期：{latest.get('日期', 'N/A')}
每股收益EPS：{latest.get('摊薄每股收益(元)', 'N/A')}元
每股净资产：{latest.get('每股净资产_调整后(元)', 'N/A')}元
净资产收益率ROE：{latest.get('净资产收益率(%)', 'N/A')}%
总资产收益率ROA：{latest.get('总资产净利润率(%)', 'N/A')}%
营业利润率：{latest.get('营业利润率(%)', 'N/A')}%
销售净利率：{latest.get('销售净利率(%)', 'N/A')}%
主营收入增长率：{latest.get('主营业务收入增长率(%)', 'N/A')}%
净利润增长率：{latest.get('净利润增长率(%)', 'N/A')}%
资产负债率：{latest.get('资产负债率(%)', 'N/A')}%
流动比率：{latest.get('流动比率', 'N/A')}"""

        docs.append(Document(
            page_content=text,
            metadata={
                "stock_code": stock_code,
                "type": "financial",
                "update_time": datetime.now().strftime("%Y-%m-%d %H:%M")
            }
        ))
    except Exception as e:
        print(f"  ❌ 财务指标失败：{e}")
    return docs


# ============================================================
# 更新知识库
# ============================================================

def update_knowledge_base(
    stock_codes: list[str],
    embeddings: HuggingFaceEmbeddings
) -> int:
    print(f"\n{'='*50}")
    print(f"更新知识库 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"股票：{stock_codes}")
    print(f"{'='*50}")

    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    all_docs = []
    for code in stock_codes:
        print(f"\n处理：{code}")
        all_docs.extend(fetch_price_analysis(code))
        all_docs.extend(fetch_news(code))
        all_docs.extend(fetch_fund_flow(code))
        all_docs.extend(fetch_financial(code))

    if not all_docs:
        print("❌ 没有获取到数据")
        return 0

    chunks = splitter.split_documents(all_docs)
    vectorstore.add_documents(chunks)
    print(f"\n✅ 更新完成，共 {len(chunks)} 个文本块")
    return len(chunks)