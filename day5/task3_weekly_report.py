# day05_fewshot/task3_weekly_report.py
from utils import chat

def generate_weekly_report(raw_notes: str, person_name: str) -> str:
    """
    把杂乱的工作记录转换成规范的周报
    """

    system = """
你是一个专业的职场写作助手，擅长把零散的工作记录整理成规范、专业的周报。
语言风格：简洁、专业、有条理。
"""

    prompt = f"""
请将以下工作记录整理成规范的周报格式。

工作记录（原始、零散）：
{raw_notes}

请按照以下格式输出周报：

---
**{person_name} 周报**
**时间：本周**

**一、本周完成工作**
（按重要程度排列，每项用动词开头，如"完成"、"推进"、"优化"）

**二、工作成果与数据**
（量化的成果，如有数据请突出）

**三、遇到的问题**
（客观描述问题，不要抱怨）

**四、下周计划**
（具体、可执行的计划）

**五、需要支持**
（需要其他人或资源的协助，没有则写"无"）
---
"""

    return chat(prompt, system_message=system, temperature=0.4)


# 测试
raw_notes = """
周一：修了个登录的bug，搞了一上午
周二：开了好几个会，讨论新功能方案，感觉没啥结论
周三：做了用户列表页面，基本完成了，还差一个筛选功能
周四：筛选功能做完了，自测没问题，提测了
周五：帮小王review了代码，发现他有个内存泄漏，让他改了
另外这周用户反馈那个搜索慢的问题，我优化了一下SQL，快了不少
下周要做导出Excel的功能，还有个性能优化的需求
"""

print(generate_weekly_report(raw_notes, "张伟"))