# day05_fewshot/task1_basic_fewshot.py
from utils import chat

# 场景：把用户的口语化反馈转换成标准Bug报告

# Zero-shot（没有例子）
zero_shot_prompt = """
把以下用户反馈转换成标准Bug报告格式：

用户反馈：点击保存按钮没反应，我点了好几次都不行
"""

# Few-shot（有例子）
few_shot_prompt = """
把用户的口语化反馈转换成标准Bug报告格式。

示例1：
用户反馈：登录的时候一直转圈，等了5分钟还没进去
Bug报告：
- 标题：登录页面加载超时
- 复现步骤：打开App → 输入账号密码 → 点击登录
- 预期结果：正常登录进入首页
- 实际结果：登录按钮点击后持续loading，超过5分钟未响应
- 严重程度：高

示例2：
用户反馈：我改了头像之后，刷新页面还是显示旧头像
Bug报告：
- 标题：头像修改后未实时更新
- 复现步骤：进入个人设置 → 修改头像 → 保存 → 刷新页面
- 预期结果：页面显示新头像
- 实际结果：页面仍显示修改前的旧头像
- 严重程度：中

现在请转换：
用户反馈：点击保存按钮没反应，我点了好几次都不行
Bug报告：
"""

print("=== Zero-shot ===")
print(chat(zero_shot_prompt, temperature=0.0))

print("\n=== Few-shot ===")
print(chat(few_shot_prompt, temperature=0.0))