import pandas as pd

# 读取原始数据
file_path = './raw/2024-06.csv'
df = pd.read_csv(file_path)

# 1. 使用线性插值补全缺失值（按列处理）
df_interpolated = df.interpolate(method='linear', limit_direction='both')

# 2. 重采样：每5行取1行，相当于1Hz变为0.2Hz（假设数据每秒1行）
df_downsampled = df_interpolated.iloc[::5].reset_index(drop=True)

# 保存处理后的数据
output_path = './processed/2024-06_processed.csv'
df_downsampled.to_csv(output_path, index=False)

print(f"处理完成，已保存到：{output_path}")