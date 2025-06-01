import os
import joblib  # 用于保存和加载标准化器
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class ThermalDataset(Dataset):
    def __init__(self, data_folder, input_cols, output_cols, months,
                 seq_length, pred_length, step,
                 mode, save_path):
        """
        input_cols: 6个输入特征列名
        output_cols: 2个输出目标列名
        months: 训练或测试所用的月份列表
        seq_length: 输入序列长度
        pred_length: 预测长度
        step: 步长
        mode: 'train' 或 'test'
        scaler_path: 保存或加载标准化器的路径
        """
        # 读取原数据
        data_list = [os.path.join(data_folder, f"2024-0{month}_processed.csv") for month in months]
        data = None
        for data_path in data_list:
            tmp = pd.read_csv(os.path.join(data_folder, data_path))
            if data is not None:
                data = pd.concat((data, tmp), ignore_index=True)
            else:
                data = tmp
        
        headers = data.columns.tolist()
        raw_data = data.values

        # 定义输入和输出
        self.features = raw_data[:, [headers.index(col) for col in input_cols]]
        self.targets = raw_data[:, [headers.index(col) for col in output_cols]]
        
        # 定义标准化器
        self.scaler_path = os.path.join(save_path, 'scalers')
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        if mode == 'train':
            # 训练模式下，标准化器拟合数据并保存
            self.normalized_features = self.feature_scaler.fit_transform(self.features)
            self.normalized_targets = self.target_scaler.fit_transform(self.targets)
            
            # 保存标准化器
            if not os.path.exists(self.scaler_path):
                os.makedirs(self.scaler_path)
            joblib.dump(self.feature_scaler, os.path.join(self.scaler_path, 'feature_scaler.pkl'))
            joblib.dump(self.target_scaler, os.path.join(self.scaler_path, 'target_scaler.pkl'))
        
        elif mode == 'test':
            # 测试模式下，加载训练时保存的标准化器
            if not os.path.exists(self.scaler_path):
                raise FileNotFoundError(f"Scaler path {self.scaler_path} does not exist. Please run in 'train' mode first.")
            self.feature_scaler = joblib.load(os.path.join(self.scaler_path, 'feature_scaler.pkl'))
            self.target_scaler = joblib.load(os.path.join(self.scaler_path, 'target_scaler.pkl'))
            
            self.normalized_features = self.feature_scaler.transform(self.features)
            self.normalized_targets = self.target_scaler.transform(self.targets)
        else:
            raise ValueError("mode must be 'train' or 'test'")
        
        # 创建序列
        self.sequences = self._create_sequences(self.normalized_features, 
                                                self.normalized_targets, 
                                                seq_length, pred_length, step)
        print('Dataset initialized with {} sequences.'.format(len(self.sequences)))
    def _create_sequences(self, features, targets, seq_length, pred_length, step):
        sequences = []
        for i in range(0, len(features) - seq_length - pred_length + 1, step):
            seq = features[i: i + seq_length]
            label = targets[i + seq_length: i + seq_length + pred_length]  # 现在label是2维向量
            sequences.append((seq, label))
        return sequences

    def inverse_transform(self, y):
        """将标准化后的输出逆变换为原始值"""
        return self.target_scaler.inverse_transform(y)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq, label = self.sequences[idx]
        # 输入形状: [channels=6, seq_length]
        # 输出形状: [channels=2, pred_length]
        return torch.FloatTensor(seq.T), torch.FloatTensor(label.T)
    
