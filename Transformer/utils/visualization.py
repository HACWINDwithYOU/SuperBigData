import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

def inverse_transform_outputs(dataset, outputs):
    outputs_shape = outputs.shape
    real_outputs = outputs.copy()
    real_outputs.transpose(0, 2, 1)  # [batch, features, pred_len] -> [batch, pred_len, features]
    real_outputs = real_outputs.reshape(-1, outputs_shape[1])
    real_outputs = dataset.inverse_transform(real_outputs)
    real_outputs = real_outputs.reshape(outputs_shape[0], outputs_shape[2], outputs_shape[1])
    return real_outputs.transpose(0, 2, 1)  # [batch, pred_len, features] -> [batch, features, pred_len]


class ThermalVisualizer:
    def __init__(self, dataset, output_cols, plot_delay = 1):
        self.dataset = dataset
        self.output_cols = output_cols
        self.plot_delay = plot_delay
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
    def plot_predictions(self, outputs, targets, num_samples=5, save_path=None, inverse_transform=True):
        """
        绘制预测值与真实值的对比散点图
        参数:
            outputs: 模型预测输出 [n_samples, n_outputs]
            targets: 真实目标值 [n_samples, n_outputs]
            num_samples: 随机显示的样本数
            save_path: 图片保存路径
        """
        if inverse_transform:
            # 反标准化
            outputs = inverse_transform_outputs(self.dataset, outputs)[:, :, self.plot_delay]
            targets = inverse_transform_outputs(self.dataset, targets)[:, :, self.plot_delay]
        else:
            outputs = outputs[:, :, self.plot_delay]
            targets = targets[:, :, self.plot_delay]

        plt.figure(figsize=(8, 10))
        
        # 随机选择样本
        sample_indices = np.random.choice(len(outputs), size=num_samples, replace=False)
        
        for i, col in enumerate(self.output_cols):
            plt.subplot(2, 1, i+1)
            
            # 所有样本的散点
            plt.scatter(targets[:, i], outputs[:, i], 
                       alpha=0.3, label='All samples', color=self.colors[i])
            
            # 突出显示随机样本
            plt.scatter(targets[sample_indices, i], outputs[sample_indices, i],
                       s=100, edgecolors='k', label='Selected samples', color=self.colors[i])
            
            # 绘制理想线
            min_val = min(targets[:, i].min(), outputs[:, i].min())
            max_val = max(targets[:, i].max(), outputs[:, i].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
            
            # 计算R2分数
            r2 = r2_score(targets[:, i], outputs[:, i])
            plt.title(f'{col} Prediction (R2={r2:.3f})')
            plt.xlabel('True Temperature (°C)')
            plt.ylabel('Predicted Temperature (°C)')
            plt.grid(True)
            plt.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_temperature(self, outputs, targets, save_path=None, inverse_transform=True):
        """绘制温度变化曲线"""
        if inverse_transform:
            outputs = inverse_transform_outputs(self.dataset, outputs)[:, :, self.plot_delay]
            targets = inverse_transform_outputs(self.dataset, targets)[:, :, self.plot_delay]
        else:
            outputs = outputs[:, :, self.plot_delay]
            targets = targets[:, :, self.plot_delay]
        
        plt.figure(figsize=(8, 10))
        
        for i, col in enumerate(self.output_cols):
            plt.subplot(2, 1, i+1)
            # sorted_indices = np.argsort(targets[:, i])
            # plt.plot(outputs[sorted_indices, i], label='Predicted', color=self.colors[i])
            # plt.plot(targets[sorted_indices, i], label='True', color='k', alpha=0.5)
            plt.plot(outputs[:, i], label='Predicted', color=self.colors[i])
            plt.plot(targets[:, i], label='True', color='k', alpha=0.5)
            plt.title(col)
            plt.xlabel('Sample')
            plt.ylabel('Temperature (°C)')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_loss_curves(self, train_losses, val_losses, save_path=None):
        """绘制训练和验证损失曲线"""
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()