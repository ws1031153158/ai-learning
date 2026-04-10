# app/services/memory_service.py
import sys
sys.path.append(".")

from mem0 import Memory
from app.config import *


class MemoryService:

    def __init__(self):
        print("⏳ 初始化记忆服务...")
        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": DEEPSEEK_MODEL,
                    "api_key": DEEPSEEK_API_KEY,
                    "openai_base_url": DEEPSEEK_BASE_URL
                }
            },
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": EMBEDDING_MODEL
                }
            },
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": "user_memory",
                    "path": MEMORY_DB_PATH
                }
            }
        }
        self.memory = Memory.from_config(config)
        print("✅ 记忆服务初始化完成")

    def search(self, query: str, user_id: str) -> str:
        """搜索相关记忆"""
        try:
            results = self.memory.search(
                query=query,
                user_id=user_id,
                limit=5
            )
            memories = results.get("results", [])
            if not memories:
                return ""
            return "\n".join([
                f"- {m['memory']}" for m in memories
            ])
        except Exception:
            return ""

    def save(self, messages: list, user_id: str):
        """保存对话到记忆"""
        try:
            self.memory.add(
                messages=messages,
                user_id=user_id
            )
        except Exception as e:
            print(f"记忆保存失败：{e}")

    def get_all(self, user_id: str) -> list:
        """获取用户所有记忆"""
        try:
            results = self.memory.get_all(user_id=user_id)
            return results.get("results", [])
        except Exception:
            return []

    def delete_all(self, user_id: str):
        """清空用户记忆"""
        try:
            self.memory.delete_all(user_id=user_id)
        except Exception as e:
            print(f"清空记忆失败：{e}")