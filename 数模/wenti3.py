import pandas as pd

# 使用你本地的文件路径
file_path = r'C:\Users\86183\Desktop\数模\wenti3.csv'
data = pd.read_csv(file_path)

# 明确指定日期格式
# 删除日期无效的行
data = data.dropna(subset=['销售日期'])

# 检查修复后的日期列
print(data['销售日期'].head())

# 筛选6月24日到6月30日的数据
filtered_data = data[(data['销售日期'] >= '2023-06-24') & (data['销售日期'] <= '2023-06-30')]

# 如果没有数据，打印出提示
if filtered_data.empty:
    print("筛选后没有数据，请检查日期是否在数据中存在。")

# 按单品编码和销售日期分组，并计算日销量
daily_sales = filtered_data.groupby(['单品编码', '销售日期'])['销量(千克)'].sum().reset_index()

# 计算前六天的平均日销量作为需求预测
average_sales = daily_sales.groupby('单品编码')['销量(千克)'].mean().reset_index()

# 重命名列
average_sales.columns = ['单品编码', '预测需求(千克)']
average_sales.to_csv('need_data.csv', index=False)
# 输出需求预测结果
print(average_sales)
