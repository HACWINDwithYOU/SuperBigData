import pandas as pd

# 要合并的文件列表（按顺序）
file_list = [f'./processed/2024-0{i}_processed.csv' for i in range(1, 6)]

# 用于存储所有数据的列表
df_list = []

# 逐个读取并加入列表
for file in file_list:
    df = pd.read_csv(file)
    df_list.append(df)

# 拼接所有数据
df_merged = pd.concat(df_list, ignore_index=True)

# 保存为新的合并文件
output_path = 'train.csv'
df_merged.to_csv(output_path, index=False)

print(f"已成功合并并保存为：{output_path}")
