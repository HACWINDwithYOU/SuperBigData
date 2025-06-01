import os
import torch
import numpy as np
from sklearn.metrics import r2_score

from utils.dataset import ThermalDataset
from utils.model import ThermalTransformer
from utils.visualization import ThermalVisualizer, inverse_transform_outputs
from utils.config import CONFIG
from utils.filter import *

load_time = "0601-190332"

class load_and_predict:
    def __init__(self, config):
        self.config = config

    def prediction(self, model_path):
        # 加载测试数据集
        self.test_dataset = ThermalDataset(
            data_folder = self.config['data_folder'],
            input_cols = self.config['input_cols'],
            output_cols = self.config['output_cols'],
            months = self.config['test_months'],
            seq_length = self.config['seq_length'],
            pred_length = self.config['pred_length'],
            step = self.config['step'],
            mode = 'train',
            save_path = self.config['save_path']
        )
        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size = self.config['batch_size'])

        # 初始化模型
        model = ThermalTransformer(
            input_channels = len(self.config['input_cols']),
            output_channels = len(self.config['output_cols']),
            seq_length = self.config['seq_length'],
            pred_length = self.config['pred_length'],
            d_model = self.config['d_model'],
            nhead = self.config['nhead'],
            num_layers = self.config['num_layers'],
            dropout = self.config['dropout']
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        model.eval()

        # 进行预测
        self.outputs = None
        self.targets = None
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                if self.outputs is not None:
                    self.outputs = np.append(self.outputs, outputs.cpu().numpy(), axis=0)
                else:
                    self.outputs = outputs.cpu().numpy()
                
                if self.targets is not None:
                    self.targets = np.append(self.targets, targets.cpu().numpy(), axis=0)
                else:
                    self.targets = targets.cpu().numpy()

        self.outputs = inverse_transform_outputs(self.test_dataset, self.outputs)
        self.targets = inverse_transform_outputs(self.test_dataset, self.targets)
        # self.outputs_filtered = apply_savgol_filter(self.outputs, window_length=501, polyorder=3)

    def visualization(self, save_path, plot_delay=1):
        # 初始化可视化工具
        visualizer = ThermalVisualizer(self.test_dataset, self.config['output_cols'], plot_delay)

        # 绘制预测散点图
        visualizer.plot_predictions(
            self.outputs,
            self.targets,
            save_path = os.path.join(save_path, 'prediction_scatter.png'),
            inverse_transform = False  # 这里不进行反标准化，因为已经在prediction中处理过了
        )

        # 绘制温度变化曲线
        visualizer.plot_temperature(
            self.outputs,
            self.targets,
            save_path = os.path.join(save_path, 'temperature_curve.png'),
            inverse_transform = False
        )

        print(f"预测完成，结果已保存到 {save_path}")
    
    def score(self):
        for delay in range(self.config['pred_length']):
            print(f"Delay {delay*5 + 5} seconds:")

            # 计算R2分数
            r2_scores = []
            for i in range(self.outputs.shape[1]):
                r2 = r2_score(self.targets[:, i, delay], self.outputs[:, i, delay])
                r2_scores.append(r2)
                print(f"Output {self.config['output_cols'][i]} R2 Score: {r2:.4f}")
            
            # 计算MSE
            mse_scores = []
            for i in range(self.outputs.shape[1]):
                mse = np.mean((self.targets[:, i, delay] - self.outputs[:, i, delay]) ** 2)
                mse_scores.append(mse)
                print(f"Output {self.config['output_cols'][i]} MSE: {mse:.4f}")
            
            # 计算MAE
            mae_scores = []
            for i in range(self.outputs.shape[1]):
                mae = np.mean(np.abs(self.targets[:, i, delay] - self.outputs[:, i, delay]))
                mae_scores.append(mae)
                print(f"Output {self.config['output_cols'][i]} MAE: {mae:.4f}")

# 示例调用
if __name__ == "__main__":
    model_path = f'./results/{load_time}/multi_output_transformer.pth'
    save_path = f'./results/{load_time}/predictions'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    predictor = load_and_predict(CONFIG)
    predictor.prediction(model_path)
    # predictor.visualization(save_path, plot_delay=CONFIG['plot_delay'])
    predictor.score()

    