import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.fft import fft

# ---------------- 配置 ----------------
important_features = [
    'LAE71AA101ZZ.AV', 'LAE72AA101ZZ.AV', 
    'LAE73AA101ZZ.AV', 'LAE74AA101ZZ.AV', 
    'PSHAOUTTE.AV','PSHBOUTTE.AV'
    ]
window_size = 60  # 秒
sampling_rate = 1.0  # 1Hz
max_freq_display = 30  # 只显示前30个频率分量
output_dir = 'RandomForest/result/fft_amplitude_plots'
os.makedirs(output_dir, exist_ok=True)

# ---------------- 数据加载 ----------------
df = pd.read_csv('./datasets/processed/2024-02_processed.csv')

# ---------------- 幅频响应绘制 ----------------
for feat in important_features:
    plt.figure(figsize=(12, 6))
    idx = 0

    for i in range(0, len(df) - window_size, window_size):
        segment = df[feat].iloc[i:i+window_size].values
        if np.any(pd.isna(segment)):
            continue

        # FFT 幅值计算
        fft_vals = np.abs(fft(segment))[:window_size // 2]
        freqs = np.fft.fftfreq(window_size, d=1/sampling_rate)[:window_size // 2]

        # 画线，透明度降低，体现重叠曲线
        plt.plot(freqs[:max_freq_display], fft_vals[:max_freq_display], alpha=0.3, label=f'Window {idx}' if idx < 5 else "")
        idx += 1

    plt.title(f'Amplitude Spectrum - {feat}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{feat}_fft_amplitude.png'))
    plt.close()

print("✅ 所有幅频响应图已保存到:", output_dir)
