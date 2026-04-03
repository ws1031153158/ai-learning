# day04_cot/task2_zeroshot_cot.py
from utils import chat

# "让我们一步步思考" 这句话有神奇的效果
problems = [
    "我有一根绳子，对折3次后剪断，展开后有几段？",
    "如果今天是周三，那么100天后是周几？",
    "一个水桶装满水重10kg，装半桶水重6kg，桶本身重多少kg？"
]

for problem in problems:
    print(f"问题：{problem}")

    # 不加CoT触发词
    answer1 = chat(problem, temperature=0.0)
    print(f"直接回答：{answer1}")

    # 加上CoT触发词
    answer2 = chat(
        f"{problem}\n\n让我们一步一步地思考这个问题：",
        temperature=0.0
    )
    print(f"CoT回答：{answer2}")
    print("=" * 50)