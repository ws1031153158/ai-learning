# server/config.py
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# LLM配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# 向量数据库
DB_PATH = "./rag/chroma_finance_v2"

# Embedding模型
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

# 默认监控股票
DEFAULT_WATCH_LIST = [
    "600519",  # 贵州茅台
    "002594",  # 比亚迪
    "300750",  # 宁德时代
]

# CrewAI需要的环境变量
os.environ["OPENAI_API_KEY"] = DEEPSEEK_API_KEY
os.environ["OPENAI_API_BASE"] = DEEPSEEK_BASE_URL
os.environ["OPENAI_MODEL_NAME"] = DEEPSEEK_MODEL