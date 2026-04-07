# rag/day3_retrieval_optimization.py
import sys
sys.path.append(".")

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# ============================================================
# 初始化公共组件
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

# ============================================================
# 准备测试数据（复用Day1的数据）
# ============================================================

news_data = [
    """贵州茅台2024年三季报显示，公司实现营业收入1144.06亿元，
    同比增长15.04%。净利润达到571.27亿元，同比增长15.08%。
    茅台酒收入为985.93亿元，系列酒收入为158.13亿元。
    公司每股收益为45.49元，ROE为32.18%，市盈率约为28倍。""",

    """比亚迪2024年11月销量数据出炉，当月销量达到50.68万辆，
    同比增长67.87%，创历史新高。其中新能源乘用车销量为49.74万辆。
    海外销量为3.07万辆，同比增长2.6倍。
    比亚迪全年销量有望突破420万辆。""",

    """美联储2024年12月议息会议决定降息25个基点，
    将联邦基金利率目标区间下调至4.25%-4.5%。
    这是美联储年内第三次降息，累计降息100个基点。
    市场预计2025年将继续降息2次。""",

    """比特币价格在2024年12月突破10万美元大关，创历史新高。
    特朗普当选美国总统后对加密货币的友好态度，
    以及比特币现货ETF的获批，是推动本轮上涨的主要原因。
    MicroStrategy已持有超过40万枚比特币。""",

    """A股市场2024年四季度迎来政策面重大利好，
    国务院出台一系列稳增长措施，包括降准降息、扩大内需等。
    上证指数从2700点附近快速反弹至3400点以上，涨幅超过25%。
    北向资金单月净买入超过600亿元，创近年新高。""",

    """茅台酒是中国最著名的白酒品牌之一，产地贵州省仁怀市茅台镇。
    茅台酒以其独特的酱香型口感著称，酿造工艺复杂，
    需要经过多次蒸馏和长期窖藏。
    茅台酒在中国文化中具有重要地位，常用于重要场合。""",

    """贵州茅台股票（600519）是A股市场市值最大的白酒股。
    公司护城河极深，品牌价值超过1万亿元。
    茅台的ROE长期保持在30%以上，是A股最优质的白马股之一。
    机构投资者普遍将茅台作为核心持仓。""",
]

# 构建向量数据库
def build_vectorstore():
    documents = [
        Document(
            page_content=text.strip(),
            metadata={"id": i, "type": "news"}
        )
        for i, text in enumerate(news_data)
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    chunks = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./rag/chroma_day3"
    )
    return vectorstore

print("⏳ 构建测试数据库...")
vectorstore = build_vectorstore()
print("✅ 完成\n")


# ============================================================
# 优化1：调整检索参数
# ============================================================

def demo_retrieval_params():
    print("=" * 50)
    print("优化1：调整检索参数")
    print("=" * 50)

    query = "茅台的市盈率是多少？"

    # k=2 只返回2个结果
    results_k2 = vectorstore.similarity_search(query, k=2)
    print(f"\nk=2 的检索结果：")
    for i, r in enumerate(results_k2):
        print(f"[{i+1}] {r.page_content[:60]}...")

    # k=5 返回5个结果（更多但可能有噪音）
    results_k5 = vectorstore.similarity_search(query, k=5)
    print(f"\nk=5 的检索结果：")
    for i, r in enumerate(results_k5):
        print(f"[{i+1}] {r.page_content[:60]}...")

    # 带相似度分数的检索
    results_with_score = vectorstore.similarity_search_with_score(
        query, k=5
    )
    print(f"\n带相似度分数的检索结果：")
    print("（分数越低越相似，0=完全相同）")
    for i, (doc, score) in enumerate(results_with_score):
        print(f"[{i+1}] 相似度分数：{score:.4f}")
        print(f"     内容：{doc.page_content[:60]}...")


# ============================================================
# 优化2：相似度阈值过滤
# 过滤掉相关性太低的结果
# ============================================================

def demo_score_threshold():
    print("\n" + "=" * 50)
    print("优化2：相似度阈值过滤")
    print("=" * 50)

    query = "茅台的市盈率是多少？"

    # 使用阈值过滤器
    # score_threshold：只返回相似度高于这个值的结果
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": 0.3,  # 只要相似度>0.3的结果
            "k": 5
        }
    )

    results = retriever.invoke(query)
    print(f"\n阈值过滤后的结果（只保留相关性高的）：")
    print(f"共 {len(results)} 个结果")
    for i, r in enumerate(results):
        print(f"[{i+1}] {r.page_content[:80]}...")


# ============================================================
# 优化3：MMR检索（最大边际相关性）
# 解决检索结果重复的问题
# ============================================================

def demo_mmr():
    print("\n" + "=" * 50)
    print("优化3：MMR检索（去重+多样性）")
    print("=" * 50)

    query = "茅台的投资价值"

    # 普通检索（可能返回重复内容）
    normal_results = vectorstore.similarity_search(query, k=3)
    print("\n普通检索结果：")
    for i, r in enumerate(normal_results):
        print(f"[{i+1}] {r.page_content[:80]}...")

    # MMR检索（保证多样性）
    mmr_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,           # 最终返回3个
            "fetch_k": 10,    # 先取10个候选
            "lambda_mult": 0.7  # 0=最多样性，1=最相关
        }
    )

    mmr_results = mmr_retriever.invoke(query)
    print("\nMMR检索结果（更多样，不重复）：")
    for i, r in enumerate(mmr_results):
        print(f"[{i+1}] {r.page_content[:80]}...")


# ============================================================
# 优化4：Reranker重排序
# 检索后对结果重新排序，效果最好
# ============================================================

def demo_reranker():
    print("\n" + "=" * 50)
    print("优化4：Reranker重排序")
    print("=" * 50)

    # 安装：pip install sentence-transformers
    try:
        from sentence_transformers import CrossEncoder

        # 加载Reranker模型（第一次运行会下载）
        print("⏳ 加载Reranker模型...")
        reranker = CrossEncoder(
            "BAAI/bge-reranker-base",
            max_length=512
        )
        print("✅ Reranker加载成功")

        query = "茅台的财务数据和投资价值"

        # 第一步：先用向量检索召回候选
        candidates = vectorstore.similarity_search(query, k=5)
        print(f"\n向量检索召回 {len(candidates)} 个候选")

        # 第二步：用Reranker重新打分排序
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = reranker.predict(pairs)

        # 第三步：按新分数排序
        ranked = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True  # 分数越高越相关
        )

        print("\nReranker重排序后的结果：")
        for i, (score, doc) in enumerate(ranked):
            print(f"[{i+1}] Reranker分数：{score:.4f}")
            print(f"     内容：{doc.page_content[:80]}...")

    except ImportError:
        print("请先安装：pip install sentence-transformers")


# ============================================================
# 优化5：查询改写
# 把用户的口语化问题改写成更适合检索的形式
# ============================================================

def demo_query_rewriting():
    print("\n" + "=" * 50)
    print("优化5：查询改写")
    print("=" * 50)

    # 用户的原始问题（口语化，不适合直接检索）
    original_query = "茅台最近咋样，值得买不？"

    # 用LLM改写成更适合检索的问题
    rewrite_prompt = f"""
请将以下口语化的投资问题改写为3个更专业、更适合检索的问题。
每个问题一行，直接输出问题，不要编号和解释。

原始问题：{original_query}
"""

    response = llm.invoke([HumanMessage(content=rewrite_prompt)])
    rewritten_queries = response.content.strip().split('\n')
    rewritten_queries = [q.strip() for q in rewritten_queries if q.strip()]

    print(f"\n原始问题：{original_query}")
    print(f"\n改写后的问题：")
    for q in rewritten_queries:
        print(f"  - {q}")

    # 用改写后的多个问题分别检索，合并结果
    all_results = []
    seen_contents = set()

    for q in rewritten_queries:
        results = vectorstore.similarity_search(q, k=2)
        for r in results:
            # 去重
            if r.page_content not in seen_contents:
                seen_contents.add(r.page_content)
                all_results.append(r)

    print(f"\n合并检索结果（共{len(all_results)}个不重复文段）：")
    for i, r in enumerate(all_results):
        print(f"[{i+1}] {r.page_content[:80]}...")

    # 最终生成回答
    context = "\n\n".join([r.page_content for r in all_results])
    final_prompt = f"""你是专业投资分析助手。
根据以下资料回答问题，如无相关信息说"暂无数据"。

资料：
{context}

问题：{original_query}

专业分析："""

    final_response = llm.invoke([HumanMessage(content=final_prompt)])
    print(f"\n最终回答：{final_response.content}")


# ============================================================
# 主程序：依次运行所有优化演示
# ============================================================

if __name__ == "__main__":
    demo_retrieval_params()
    demo_score_threshold()
    demo_mmr()
    demo_reranker()
    demo_query_rewriting()

    print("\n" + "=" * 50)
    print("✅ Day3 完成！")
    print("=" * 50)