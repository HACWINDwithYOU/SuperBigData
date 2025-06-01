import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pywt
import os

# 配置
important_features = [
    'LAE71AA101ZZ.AV', 'LAE72AA101ZZ.AV', 
    'LAE73AA101ZZ.AV', 'LAE74AA101ZZ.AV', 
    'PSHAOUTTE.AV','PSHBOUTTE.AV'
    ]

window_size = 60
wavelet = 'db4'
level = 3

# 数据加载
df = pd.read_csv('./datasets/processed/2024-02_processed.csv')

# 输出目录
output_dir = 'RandomForest/result/wavelet_heatmaps'
os.makedirs(output_dir, exist_ok=True)

# 遍历每个重要特征
for feat in important_features:
    energy_matrix = []

    # 滑动窗口计算小波能量
    for i in range(0, len(df) - window_size, window_size):
        segment = df[feat].iloc[i:i+window_size].values
        if np.any(pd.isna(segment)):
            energy_matrix.append([np.nan] * (level + 1))
            continue
        coeffs = pywt.wavedec(segment, wavelet=wavelet, level=level)
        energies = [np.sum(np.square(c)) for c in coeffs]  # 包含 [A3, D3, D2, D1]
        energies = energies[::-1]  # 变为 [D1, D2, D3, A3]
        energy_matrix.append(energies)

    energy_matrix = np.array(energy_matrix).T  # shape: [4, 窗口数]

    # 热力图绘制
    plt.figure(figsize=(12, 4))
    sns.heatmap(energy_matrix, cmap='viridis', cbar=True,
                xticklabels=100, yticklabels=['D1', 'D2', 'D3', 'A3'])
    plt.title(f'Wavelet Energy Heatmap - {feat}')
    plt.xlabel('Window Index')
    plt.ylabel('Wavelet Level')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{feat}_heatmap.png'))
    plt.close()

print("✅ 所有热力图已生成并保存到:", output_dir)
