# day01_basic/task2_four_elements.py
from utils import chat

# 一个完整的Prompt通常包含：
# 1. 角色（Role）    - 你是谁
# 2. 背景（Context） - 情况是什么
# 3. 任务（Task）    - 要做什么
# 4. 格式（Format）  - 输出什么样

# 练习：逐步添加要素，观察输出变化

# 只有任务
prompt_v1 = "总结这段话：人工智能正在改变各行各业，从医疗到金融，从教育到制造业，AI的应用越来越广泛。"

# 添加格式要求
prompt_v2 = """
总结以下内容，用一句话概括核心观点：

内容：人工智能正在改变各行各业，从医疗到金融，从教育到制造业，AI的应用越来越广泛。

要求：
- 一句话，不超过20字
- 直接输出总结，不要有多余的话
"""

# 添加角色和背景
prompt_v3 = """
你是一名科技媒体编辑，需要为忙碌的读者提炼文章核心。

请将以下内容总结为一句话标题（不超过20字）：

内容：人工智能正在改变各行各业，从医疗到金融，从教育到制造业，AI的应用越来越广泛。

直接输出标题，不需要任何解释。
"""

for i, prompt in enumerate([prompt_v1, prompt_v2, prompt_v3], 1):
    print(f"=== Prompt v{i} ===")
    print(chat(prompt))
    print()