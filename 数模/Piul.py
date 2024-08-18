import pulp
import pandas as pd

# 假设我们已经将数据加载到一个 Pandas DataFrame 中，名为 output_data
# output_data 的列包括：['单品编码', '单品名称', '分类编码', '分类名称', '总销量', '平均销售单价', '批发价格', '损耗率', '需求']
output_data = pd.read_csv('processed_data.csv')
# 定义问题
problem = pulp.LpProblem("Vegetable_Replenishment_Optimization", pulp.LpMaximize)

# 决策变量：每个单品的订购量（连续变量）
q = pulp.LpVariable.dicts("q", output_data['单品编码'], lowBound=0, cat='Continuous')

# 二进制变量：是否选择某个单品
x = pulp.LpVariable.dicts("x", output_data['单品编码'], cat='Binary')

# 目标函数：最大化商超收益
for i in output_data.index:
    problem += pulp.lpSum([
        (output_data.iloc[i]['平均销售单价'] * q[i] * (1 - output_data.iloc[i]['损耗率']) - 
         output_data.iloc[i]['批发价格'] * q[i])
    ])


# 约束条件 1：控制选择的单品数量在 27 到 33 之间
problem += pulp.lpSum([x[i] for i in output_data.index]) >= 27, "Minimum_Items_Selected"
problem += pulp.lpSum([x[i] for i in output_data.index]) <= 33, "Maximum_Items_Selected"

# 约束条件 2：每个被选择的单品订购量至少为 2.5 千克
for i in output_data.index:
    problem += q[i] >= 2.5 * x[i], f"Min_Display_Quantity_{i}"

# 约束条件 3：订购量不能超过市场需求
for i in output_data.index:
    problem += q[i] <= output_data.loc[i, '需求'], f"Demand_Constraint_{i}"

# 求解模型
problem.solve()

# 输出结果
print("Status:", pulp.LpStatus[problem.status])

for i in output_data.index:
    print(f"单品编码: {output_data.loc[i, '单品编码']}, "
          f"单品名称: {output_data.loc[i, '单品名称']}, "
          f"订购量: {q[i].varValue:.2f} kg, "
          f"选择: {'是' if x[i].varValue == 1 else '否'}")

# 输出商超最大化的收益
print(f"最大化的收益: {pulp.value(problem.objective):.2f} 元")
