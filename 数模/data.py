import pandas as pd

# 读取附件数据
attachment1 = pd.read_csv('附件1.csv')
attachment2 = pd.read_csv('附件2.csv')
attachment3 = pd.read_csv('附件3.csv')
attachment4 = pd.read_csv('附件4.csv')

# 处理附件2：计算每个单品的总销量和平均销售单价
sales_data = attachment2.groupby('单品编码').agg(
    总销量=('销量(千克)', 'sum'),
    平均销售单价=('销售单价(元/千克)', 'mean')
).reset_index()

# 处理附件3：获取最新的批发价格
wholesale_price = attachment3.groupby('单品编码').agg(
    批发价格=('批发价格(元/千克)', 'last')
).reset_index()

# 处理附件4：获取每个分类的平均损耗率
loss_rate = attachment4.rename(columns={'平均损耗率(%)_小分类编码_不同值': '损耗率'}).copy()

# 合并数据：合并销售数据、批发价格和损耗率
data = pd.merge(sales_data, attachment1, on='单品编码')
data = pd.merge(data, wholesale_price, on='单品编码')
data = pd.merge(data, loss_rate, left_on='分类编码', right_on='小分类编码')

# 计算需求（取历史总销量作为需求估计）
data['需求'] = data['总销量']

# 提取模型需要的参数并保存为CSV文件
output_data = data[['单品编码', '单品名称', '分类编码', '分类名称', '总销量', '平均销售单价', '批发价格', '损耗率', '需求']]

# 将数据保存为CSV文件
output_data.to_csv('processed_data.csv', index=False)

print("数据已成功保存为 'processed_data.csv'")
