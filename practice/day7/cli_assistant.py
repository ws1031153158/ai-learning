# final_project/cli_assistant.py
"""
功能：
✅ 多轮对话（记忆历史）
✅ 角色切换（不同的System Prompt）
✅ 流式输出
✅ Token统计
✅ 错误处理
✅ 对话历史保存
"""
import sys
sys.path.append('.')
import json
import os
import time
from datetime import datetime
from openai import OpenAI, RateLimitError, APITimeoutError
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)

# 预设角色
ROLES = {
    "default": {
        "name": "小AI",
        "system": "你是一个友好、专业的AI助手，回答简洁准确。"
    },
    "coder": {
        "name": "代码助手",
        "system": """
你是一个资深程序员助手，专注于帮助解决编程问题。
- 提供可运行的代码示例
- 解释代码的关键部分
- 指出潜在的问题和优化点
- 使用markdown格式展示代码
"""
    },
    "teacher": {
        "name": "学习导师",
        "system": """
你是一个耐心的学习导师。
- 用简单的类比解释复杂概念
- 循序渐进，从基础开始
- 在解释后提出一个思考问题帮助巩固
- 鼓励学习者
"""
    },
    "critic": {
        "name": "批判性思维助手",
        "system": """
你是一个批判性思维助手。
- 对任何观点都提出质疑和反驳
- 指出论点中的漏洞和假设
- 提供不同角度的看法
- 帮助用户更全面地思考问题
"""
    }
}

class CLIAssistant:
    def __init__(self):
        self.history = []
        self.current_role = "default"
        self.total_tokens = 0
        self.session_start = datetime.now()

    def stream_response(self, messages: list) -> str:
        """流式输出并返回完整响应"""
        full_response = ""
        role_name = ROLES[self.current_role]["name"]
        print(f"\n{role_name}：", end="", flush=True)

        for attempt in range(3):
            try:
                stream = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    stream=True,
                    temperature=0.7
                )

                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        print(content, end="", flush=True)
                        full_response += content

                print("\n")
                return full_response

            except RateLimitError:
                wait = 2 ** attempt
                print(f"\n[触发限流，{wait}秒后重试...]")
                time.sleep(wait)
            except APITimeoutError:
                print("\n[请求超时，正在重试...]")

        return "抱歉，请求失败，请重试"

    def chat(self, user_input: str) -> str:
        """发送消息并获取回复"""
        self.history.append({"role": "user", "content": user_input})

        messages = [
            {"role": "system", "content": ROLES[self.current_role]["system"]}
        ] + self.history[-10:]  # 只保留最近10轮，避免超出上下文

        response = self.stream_response(messages)
        self.history.append({"role": "assistant", "content": response})

        return response

    def switch_role(self, role_key: str):
        """切换角色"""
        if role_key in ROLES:
            self.current_role = role_key
            self.history = []  # 切换角色时清空历史
            print(f"✅ 已切换到【{ROLES[role_key]['name']}】模式，对话历史已清空\n")
        else:
            print(f"❌ 未知角色，可用角色：{', '.join(ROLES.keys())}")

    def save_history(self):
        """保存对话历史"""
        if not self.history:
            print("没有对话记录可保存")
            return

        filename = f"chat_{self.session_start.strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                "session_start": self.session_start.isoformat(),
                "role": self.current_role,
                "history": self.history
            }, f, ensure_ascii=False, indent=2)
        print(f"✅ 对话已保存到 {filename}")

    def show_help(self):
        print("""
╔══════════════════════════════════════╗
║           可用命令                    ║
╠══════════════════════════════════════╣
║ /role <角色>  切换角色                ║
║   可选：default/coder/teacher/critic  ║
║ /clear        清空对话历史            ║
║ /save         保存对话记录            ║
║ /help         显示此帮助              ║
║ /quit         退出                    ║
╚══════════════════════════════════════╝
""")

    def run(self):
        """主循环"""
        print("=" * 50)
        print("  🤖 AI 命令行助手")
        print(f"  当前角色：{ROLES[self.current_role]['name']}")
        print("  输入 /help 查看命令")
        print("=" * 50)

        while True:
            try:
                user_input = input("\n你：").strip()

                if not user_input:
                    continue

                # 命令处理
                if user_input.startswith("/"):
                    parts = user_input.split()
                    cmd = parts[0]

                    if cmd == "/quit":
                        self.save_history()
                        print("👋 再见！")
                        break
                    elif cmd == "/role" and len(parts) > 1:
                        self.switch_role(parts[1])
                    elif cmd == "/clear":
                        self.history = []
                        print("✅ 对话历史已清空")
                    elif cmd == "/save":
                        self.save_history()
                    elif cmd == "/help":
                        self.show_help()
                    else:
                        print(f"❓ 未知命令，输入 /help 查看帮助")
                else:
                    self.chat(user_input)

            except KeyboardInterrupt:
                print("\n\n👋 检测到中断，正在保存对话...")
                self.save_history()
                break


if __name__ == "__main__":
    assistant = CLIAssistant()
    assistant.run()