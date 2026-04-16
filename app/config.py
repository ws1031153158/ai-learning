# app/config.py
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# ── LLM ──────────────────────────────────────────────
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# ── 数据库路径 ────────────────────────────────────────
RAG_DB_PATH = "./data/chroma_rag"
MEMORY_DB_PATH = "./data/chroma_memory"

# ── 模型 ──────────────────────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

# ── 业务配置 ──────────────────────────────────────────
DEFAULT_WATCH_LIST = ["600519", "002594", "300750"]
MAX_SESSIONS = 100
MAX_HISTORY = 20
SESSION_TIMEOUT = 3600

# ── CrewAI 需要的环境变量 ─────────────────────────────
os.environ["OPENAI_API_KEY"] = DEEPSEEK_API_KEY
os.environ["OPENAI_API_BASE"] = DEEPSEEK_BASE_URL
os.environ["OPENAI_MODEL_NAME"] = DEEPSEEK_MODEL

# ── 数据目录 ──────────────────────────────────────────
os.makedirs("./data", exist_ok=True)
os.makedirs(RAG_DB_PATH, exist_ok=True)
os.makedirs(MEMORY_DB_PATH, exist_ok=True)

# ── 数据库 ────────────────────────────────────────────
DB_URL = "mysql+pymysql://financeai:financeai123456@localhost:3306/financeai"

# ── JWT ───────────────────────────────────────────────
JWT_SECRET = "chanceforgao19991010"   # ← 改成随机字符串
JWT_EXPIRE_DAYS = 7                           # Token 有效期7天
JWT_EXPIRE_DAYS_LONG = 30                     # 保持登录30天