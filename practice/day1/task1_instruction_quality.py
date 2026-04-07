# day01_basic/task1_instruction_quality.py
import sys
sys.path.append('.')
from utils import chat

# ❌ 模糊的Prompt
bad_prompt = "帮我写个介绍"

# ✅ 清晰的Prompt
good_prompt = """
请为一款名为"FocusFlow"的番茄钟App写一段产品介绍。

要求：
- 目标用户：职场人士和学生
- 字数：100字以内
- 语气：简洁、专业
- 突出：专注效率、简单易用两个核心卖点
"""

print("=== 模糊Prompt的输出 ===")
print(chat(bad_prompt))

print("\n=== 清晰Prompt的输出 ===")
print(chat(good_prompt))