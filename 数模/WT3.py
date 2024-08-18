import pandas as pd
import pulp

# 加载数据
need_data = pd.read_csv('need_data.csv')
attachment_1 = pd.read_csv('附件1.csv')
attachment_3 = pd.read_csv('附件3.csv')
attachment_4 = pd.read_csv('附件4.csv')

# 合并need_data, attachment_1, attachment_3，这三者都有单品编码
merged_data = pd.merge(need_data, attachment_1, on='单品编码')
merged_data = pd.merge(merged_data, attachment_3[['单品编码', '批发价格(元/千克)']], on='单品编码')

# 使用分类编码和小分类编码来合并损耗率信息
merged_data = pd.merge(merged_data, attachment_4[['小分类编码', '平均损耗率(%)_小分类编码_不同值']], left_on='分类编码', right_on='小分类编码', how='left')

# 保存合并后的数据集为CSV文件
merged_data.to_csv('merged_data_with_loss_rates.csv', index=False)

# 转换损耗率为小数
merged_data['平均损耗率'] = merged_data['平均损耗率(%)_小分类编码_不同值'] / 100

# 决策变量
x = pulp.LpVariable.dicts("x", merged_data['单品编码'], cat='Binary')
q = pulp.LpVariable.dicts("q", merged_data['单品编码'], lowBound=2.5, cat='Continuous')

# 创建线性规划问题
problem = pulp.LpProblem("Maximize_Profit", pulp.LpMaximize)

# 目标函数：最大化总收益
problem += pulp.lpSum([
    (q[i] * (1 - merged_data.loc[j, '平均损耗率']) * merged_data.loc[j, '预测需求(千克)'] - q[i] * merged_data.loc[j, '批发价格(元/千克)'])
    for j, i in enumerate(merged_data['单品编码'])
])

# 约束条件 1：单品数量控制在 27 到 33 之间
problem += pulp.lpSum([x[i] for i in merged_data['单品编码']]) >= 27
problem += pulp.lpSum([x[i] for i in merged_data['单品编码']]) <= 33

# 约束条件 2：每个单品的订购量至少为 2.5 千克
for i in merged_data['单品编码']:
    problem += q[i] >= 2.5 * x[i]

# 约束条件 3：订购量不能超过需求量
for j, i in enumerate(merged_data['单品编码']):
    problem += q[i] <= merged_data.loc[j, '预测需求(千克)']

# 求解模型
problem.solve()

# 输出结果
results = pd.DataFrame({
    '单品编码': merged_data['单品编码'],
    '订购量(千克)': [q[i].varValue for i in merged_data['单品编码']],
    '选择': [x[i].varValue for i in merged_data['单品编码']],
    '定价(元/千克)': merged_data['预测需求(千克)']
})

results = results[results['选择'] == 1]  # 过滤出选择的单品
results.to_csv('supply_and_pricing_strategy.csv', index=False)

results
