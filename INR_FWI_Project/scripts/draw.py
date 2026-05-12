
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置中文字体，Windows系统常见的字体是 SimHei（黑体）或 Microsoft YaHei（微软雅黑）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或者 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False  # 防止负号显示乱码

# 读取 Excel 文件
df = pd.read_excel("A组记录.xlsx")

# 将 '层数' 和 '每层神经元数量' 列的数据转换为字符串并去除空格
df['层数'] = df['层数'].astype(str).str.strip()
df['每层神经元数量'] = df['每层神经元数量'].astype(str).str.strip()

# 检查是否有重复的组合
duplicates = df[df.duplicated(subset=['层数', '每层神经元数量'], keep=False)]
if not duplicates.empty:
    print("存在重复的组合：")
    print(duplicates)

# 如果有重复的组合，聚合（取平均）
df = df.groupby(['层数', '每层神经元数量'], as_index=False)['SSIM'].mean()

# 使用 pivot 创建热图数据
heatmap_data = df.pivot(index='层数', columns='每层神经元数量', values='SSIM')

# 创建热图
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, cmap='viridis')
plt.title("层数 vs 每层神经元数量 vs SSIM 热图")
plt.show()
