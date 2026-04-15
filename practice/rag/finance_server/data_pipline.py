# app/services/data_pipeline.py
import sys
sys.path.append(".")

import requests
import json
from datetime import datetime, timedelta
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import *

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30
)


# ── 公共请求头 ────────────────────────────────────────

SINA_HEADERS = {
    "Referer": "https://finance.sina.com.cn",
    "User-Agent": "Mozilla/5.0"
}

EM_HEADERS = {
    "Referer": "https://data.eastmoney.com",
    "User-Agent": "Mozilla/5.0"
}


# ── 数据获取 ──────────────────────────────────────────

def fetch_price_analysis(stock_code: str) -> list[Document]:
    docs = []
    try:
        # 新浪实时行情
        symbol = (
            f"sh{stock_code}"
            if stock_code.startswith("6")
            else f"sz{stock_code}"
        )
        url = f"http://hq.sinajs.cn/list={symbol}"
        resp = requests.get(
            url, headers=SINA_HEADERS, timeout=10
        )
        resp.encoding = "gbk"
        data_str = resp.text.split('"')[1]
        fields = data_str.split(",")

        if len(fields) < 32:
            return docs

        name = fields[0]
        price = float(fields[3])
        prev_close = float(fields[2])
        high = float(fields[4])
        low = float(fields[5])
        volume = float(fields[8])
        change_pct = round(
            (price - prev_close) / prev_close * 100, 2
        ) if prev_close > 0 else 0
        change_amount = round(price - prev_close, 2)

        text = f"""【{stock_code} {name} 行情分析】
更新时间：{datetime.now().strftime("%Y-%m-%d %H:%M")}
最新价：{price}元
涨跌幅：{change_pct}%
涨跌额：{change_amount}元
今日最高：{high}元
今日最低：{low}元
成交量：{volume}手
昨收：{prev_close}元"""

        docs.append(Document(
            page_content=text,
            metadata={
                "stock_code": stock_code,
                "type": "price_analysis",
                "update_time": datetime.now().strftime("%Y-%m-%d %H:%M")
            }
        ))
        print(f"  ✅ 行情数据获取成功")

    except Exception as e:
        print(f"  ❌ 行情数据失败：{e}")
    return docs


def fetch_news(stock_code: str) -> list[Document]:
    docs = []
    try:
        url = "https://np-listapi.eastmoney.com/comm/wap/getListInfo"
        params = {
            "cb": "callback",
            "client": "wap",
            "type": 1,
            "mTypeAndCode": f"0,{stock_code}",
            "pageSize": NEWS_LIMIT,
            "pageIndex": 1,
            "callback": "cb"
        }
        resp = requests.get(
            url, params=params,
            headers=EM_HEADERS, timeout=10
        )
        text = resp.text
        start = text.index("(") + 1
        end = text.rindex(")")
        data = json.loads(text[start:end])

        items = data.get("data", {}).get("list", [])
        for item in items[:NEWS_LIMIT]:
            title = item.get("title", "")
            pub_time = item.get("datetime", "")
            content = item.get("digest", "")[:400]

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
        print(f"  ✅ 新闻获取成功，共{len(docs)}条")

    except Exception as e:
        print(f"  ❌ 新闻获取失败：{e}")
    return docs


def fetch_fund_flow(stock_code: str) -> list[Document]:
    docs = []
    try:
        market = 1 if stock_code.startswith("6") else 0
        url = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
        params = {
            "lmt": 5,
            "klt": 101,
            "secid": f"{market}.{stock_code}",
            "fields1": "f1,f2,f3,f7",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63",
            "ut": "b2884a393a59ad64002292a3e90d46a5"
        }
        resp = requests.get(
            url, params=params,
            headers=EM_HEADERS, timeout=10
        )
        data = resp.json()
        klines = data.get("data", {}).get("klines", [])

        if not klines:
            return docs

        text = f"""【{stock_code}资金流向】
更新时间：{datetime.now().strftime("%Y-%m-%d %H:%M")}\n"""

        for kline in klines:
            fields = kline.split(",")
            if len(fields) < 7:
                continue
            main_flow = float(fields[1]) if fields[1] != "-" else 0
            status = "净流入🟢" if main_flow > 0 else "净流出🔴"
            text += f"""
日期：{fields[0]}
主力：{status} {abs(main_flow / 1e8):.2f}亿（占比{fields[6]}%）"""

        docs.append(Document(
            page_content=text,
            metadata={
                "stock_code": stock_code,
                "type": "fund_flow",
                "update_time": datetime.now().strftime("%Y-%m-%d %H:%M")
            }
        ))
        print(f"  ✅ 资金流向获取成功")

    except Exception as e:
        print(f"  ❌ 资金流向失败：{e}")
    return docs


def fetch_financial(stock_code: str) -> list[Document]:
    docs = []
    try:
        market = "SH" if stock_code.startswith("6") else "SZ"
        url = "https://emweb.securities.eastmoney.com/PC_HSF10/NewFinanceAnalysis/ZYZBAjaxNew"
        params = {
            "type": 1,
            "code": f"{market}{stock_code}"
        }
        resp = requests.get(
            url, params=params,
            headers=EM_HEADERS, timeout=10
        )
        data = resp.json()
        items = data.get("data", [])

        if not items:
            return docs

        latest = items[0]
        text = f"""【{stock_code}财务指标】
更新时间：{datetime.now().strftime("%Y-%m-%d %H:%M")}
报告期：{latest.get("REPORTDATE", "N/A")}
每股收益EPS：{latest.get("EPSBASIC", "N/A")}元
净资产收益率ROE：{latest.get("ROEJQ", "N/A")}%
总资产收益率ROA：{latest.get("ZZCJLL", "N/A")}%
销售净利率：{latest.get("XSMLL", "N/A")}%
主营收入增长率：{latest.get("YYZSRGRATE", "N/A")}%
净利润增长率：{latest.get("GSJLRGRATE", "N/A")}%
资产负债率：{latest.get("ZCFZL", "N/A")}%"""

        docs.append(Document(
            page_content=text,
            metadata={
                "stock_code": stock_code,
                "type": "financial",
                "update_time": datetime.now().strftime("%Y-%m-%d %H:%M")
            }
        ))
        print(f"  ✅ 财务指标获取成功")

    except Exception as e:
        print(f"  ❌ 财务指标失败：{e}")
    return docs


# ── 更新知识库 ────────────────────────────────────────

def update_knowledge_base(
    stock_codes: list[str],
    embeddings: HuggingFaceEmbeddings
) -> int:
    print(f"\n{'=' * 50}")
    print(f"更新知识库 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"股票：{stock_codes}")
    print(f"{'=' * 50}")

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