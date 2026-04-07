# rag/day1_basic_rag.py
import sys
sys.path.append(".")

from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv(override=True)

# ============================================================
# 第一步：准备测试数据
# 用几段财经新闻模拟知识库内容
# ============================================================

# 模拟几篇财经新闻（先用假数据，后面会接真实数据）
news_data = [
    """
    贵州茅台2024年三季报显示，公司实现营业收入1144.06亿元，
    同比增长15.04%。净利润达到571.27亿元，同比增长15.08%。
    茅台酒收入为985.93亿元，系列酒收入为158.13亿元。
    公司每股收益为45.49元，ROE为32.18%。
    分析师普遍认为茅台业绩符合预期，维持买入评级。
    """,

    """
    比亚迪2024年11月销量数据出炉，当月销量达到50.68万辆，
    同比增长67.87%，创历史新高。其中新能源乘用车销量为
    49.74万辆。海外销量为3.07万辆，同比增长2.6倍。
    比亚迪全年销量有望突破420万辆，超越特斯拉成为全球
    新能源汽车销量第一。
    """,

    """
    美联储2024年12月议息会议决定降息25个基点，
    将联邦基金利率目标区间下调至4.25%-4.5%。
    这是美联储年内第三次降息，累计降息100个基点。
    美联储主席鲍威尔表示，通胀已大幅回落，
    但仍高于2%的目标。市场预计2025年将继续降息2次。
    """,

    """
    比特币价格在2024年12月突破10万美元大关，
    创历史新高。分析人士认为，特朗普当选美国总统后
    对加密货币的友好态度，以及比特币现货ETF的获批，
    是推动本轮上涨的主要原因。
    机构投资者持续增加比特币配置，
    MicroStrategy已持有超过40万枚比特币。
    """,

    """
    A股市场2024年四季度迎来政策面重大利好，
    国务院出台一系列稳增长措施，
    包括降准降息、扩大内需等政策组合拳。
    上证指数从2700点附近快速反弹至3400点以上，
    涨幅超过25%。北向资金大幅流入，
    单月净买入超过600亿元，创近年新高。
    """
]

# ============================================================
# 第二步：把文字转成 Document 对象
# LangChain 用 Document 来统一管理文本
# ============================================================

documents = []
for i, news in enumerate(news_data):
    doc = Document(
        page_content=news.strip(),
        metadata={
            "source": f"news_{i+1}",
            "type": "financial_news"
        }
    )
    documents.append(doc)

print(f"✅ 准备了 {len(documents)} 篇新闻文档")

# ============================================================
# 第三步：文本切块
# 把长文章切成小段，方便检索
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,       # 每块最多200个字符
    chunk_overlap=20,     # 相邻块之间重叠20个字符（防止信息被切断）
    length_function=len,
)

chunks = text_splitter.split_documents(documents)
print(f"✅ 切块完成，共 {len(chunks)} 个文本块")

# 看看切出来的效果
print("\n--- 第一个文本块内容 ---")
print(chunks[0].page_content)
print(f"来源：{chunks[0].metadata}")

# ============================================================
# 第四步：向量化 + 存入向量数据库
# ============================================================

# 使用 DeepSeek 兼容的 Embedding
# 注意：这里用 OpenAI 的 Embedding 接口格式，指向 DeepSeek
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5"  # 免费、中文效果好、体积小
)

print("\n⏳ 正在向量化并存入数据库...")

# 创建向量数据库
# persist_directory 指定本地存储路径，下次启动不需要重新建库
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./rag/chroma_db"
)

print("✅ 向量数据库创建成功！")

# ============================================================
# 第五步：测试检索
# ============================================================

print("\n--- 测试检索效果 ---")

# 用问题去检索相关文段
query = "比特币两个月内的走势会怎样？"
results = vectorstore.similarity_search(query, k=2)  # 返回最相关的2个文段

print(f"问题：{query}")
print(f"检索到 {len(results)} 个相关文段：")
for i, result in enumerate(results):
    print(f"\n[文段{i+1}]")
    print(result.page_content)

# ============================================================
# 第六步：把检索结果 + 问题 → 发给 LLM 生成回答
# ============================================================

print("\n--- 生成最终回答 ---")

# 初始化 LLM
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com",
    temperature=0.3
)

# 把检索到的文段拼接成上下文
context = "\n\n".join([r.page_content for r in results])

# 构建 Prompt
prompt = f"""你是一个专业的投资分析助手。
请根据以下参考资料回答用户的问题。
如果参考资料中没有相关信息，请直接说"暂无相关数据"，不要编造。

参考资料：
{context}

用户问题：{query}

请给出专业、简洁的回答："""

# 调用 LLM
response = llm.invoke([HumanMessage(content=prompt)])

print(f"问题：{query}")
print(f"回答：{response.content}")