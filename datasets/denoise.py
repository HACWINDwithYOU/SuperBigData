import pandas as pd
import os
from scipy.signal import savgol_filter

# ---------- 配置 ----------
input_train_path = 'datasets/train.csv'
input_test_path = 'datasets/test.csv'
output_train_path = 'datasets/train_denoised.csv'
output_test_path = 'datasets/test_denoised.csv'

# Savitzky-Golay 滤波参数
window_length = 101  # 必须是奇数
polyorder = 2

# ---------- 降噪函数 ----------
def denoise_dataframe(df, window_length=101, polyorder=2):
    df_denoised = df.copy()
    numeric_cols = df.select_dtypes(include='number').columns

    for col in numeric_cols:
        # 若列太短无法滤波，则跳过
        if len(df[col]) < window_length:
            continue
        try:
            df_denoised[col] = savgol_filter(df[col], window_length=window_length, polyorder=polyorder)
        except Exception as e:
            print(f"[警告] 跳过列: {col}, 错误: {e}")

    return df_denoised

# ---------- 加载并处理 ----------
print("读取并处理训练集...")
train_df = pd.read_csv(input_train_path)
train_denoised = denoise_dataframe(train_df, window_length, polyorder)
train_denoised.to_csv(output_train_path, index=False)
print(f"训练集降噪完成，保存至: {output_train_path}")

print("读取并处理测试集...")
test_df = pd.read_csv(input_test_path)
test_denoised = denoise_dataframe(test_df, window_length, polyorder)
test_denoised.to_csv(output_test_path, index=False)
print(f"测试集降噪完成，保存至: {output_test_path}")
