#绘制随时间变化的折线图
import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('dirty-knn-50.csv')
#df = pd.read_csv('clean.csv')
#df = pd.read_csv('dirty-mean-50.csv')

# 选取前200条数据
df = df.head(200)

# 绘制V2列的折线图
plt.figure(figsize=(10, 6))
plt.plot(df['V2'], label='data')
plt.title('M4-monthly-timeseries(knn-50%)')
#plt.title('M4-monthly-timeseries(clean)')
#plt.title('M4-monthly-timeseries(mean-50%)')
plt.xlabel('date-index')
plt.ylabel('Value')
plt.legend()
plt.show()