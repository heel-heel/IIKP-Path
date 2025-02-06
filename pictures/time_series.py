import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 生成x值，从-π到π，共10000个点
x = np.linspace(-1.05*np.pi,1.05* np.pi, 10000)
# 计算对应的y值，即sin(x)
y = np.sin(x)

# 绘制正弦曲线
fig, ax = plt.subplots()
ax.plot(x, y, color='#1b5aea', linewidth=2)

# 隐藏y轴的刻度和轴线
ax.yaxis.set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)

# 隐藏x轴的刻度和轴线
ax.xaxis.set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_position(('data',0))

# 隐藏标题
ax.title.set_visible(False)


# 读取CSV文件
# 请将 'your_file.csv' 替换为您的CSV文件路径
data = pd.read_excel('data1.xlsx')

# 假设CSV文件中有两列，分别为 'x' 和 'y'
# 请根据您的CSV文件中的列名进行调整
x1 = data['x1']
y1 = data['y1']
x2 =data['x2']
y2=data['y2']

# 绘制图形
plt.plot(x1, y1, color='#00a270', linewidth=2)
plt.plot(x2, y2, color='#00a270', linewidth=2)

# 显示图形
plt.show()

