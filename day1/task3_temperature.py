# day01_basic/task3_temperature.py
from utils import chat

prompt = "给一家卖手工咖啡的小店起3个有创意的名字"

print("=== Temperature = 0.0（保守）===")
for i in range(3):
    print(f"第{i+1}次：{chat(prompt, temperature=0.0)}")

print("\n=== Temperature = 1.5（创意）===")
for i in range(3):
    print(f"第{i+1}次：{chat(prompt, temperature=1.5)}")