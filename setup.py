import re

# 假设文件名是 strategy_1.json
filename = "ace/data/strategies/strategy_112.json"

# 使用正则表达式提取策略编号
match = re.search(r'strategy_(\d+)', filename)

if match:
    strategy_number = match.group(1)
    print(f"策略编号: {strategy_number}")
else:
    print("未找到策略编号")

