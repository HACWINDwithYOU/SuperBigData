import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- 文件路径 ---
csv_path = "datasets/denoised/2024-02_denoised.csv"  # 替换为你的实际路径
output_path = "datasets/denoised.png"

# --- 要绘制的列 ---
columns_to_plot = [
    "LAE71AA101ZZ.AV", "A1SPRFLOW.AV", "LAE72AA101ZZ.AV", "B1SPRFLOW.AV",
    "LAE73AA101ZZ.AV", "A2SPRFLOW.AV", "LAE74AA101ZZ.AV", "B2SPRFLOW.AV",
    "PSHAOUTTE.AV", "PSHBOUTTE.AV", "SHAOUTTE.AV", "SHBOUTTE.AV"
]

# --- 读取数据 ---
df = pd.read_csv(csv_path)

# --- 构造绘图 ---
fig, axes = plt.subplots(3, 4, figsize=(10, 10))
axes = axes.flatten()

for i, col in enumerate(columns_to_plot):
    ax = axes[i]

    # 构造横轴为样本序号
    x = np.arange(len(df))
    y = df[col].values

    # 构造非缺失掩码
    valid_mask = ~pd.isna(y)

    # 绘制散点图，只保留非缺失值
    ax.scatter(x[valid_mask], y[valid_mask], s=0.001, alpha=0.7)
    ax.set_title(col)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Value")
    ax.grid(True)

# 删除多余子图（如果列少于 12）
for j in range(len(columns_to_plot), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle("Selected Signal Trends (Scatter Plot)", fontsize=16, y=1.02)

# --- 保存图像 ---
plt.savefig(output_path, dpi=300, bbox_inches='tight')

plt.show()
