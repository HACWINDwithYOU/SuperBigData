import time
import os
from datetime import datetime

import torch
import logging
import numpy as np
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, random_split
from torch import nn

from utils.model import ThermalTransformer
from utils.dataset import ThermalDataset
from utils.visualization import *
from utils.logger import setlogger
from utils.config import CONFIG

class MammalDSB:
    def __init__(self, config):
        self.config = config

    def setup(self):
        # 初始化数据集
        train_dataset = ThermalDataset(
            data_folder = self.config['data_folder'],
            input_cols = self.config['input_cols'],
            output_cols = self.config['output_cols'],
            months = self.config['train_months'],
            seq_length = self.config['seq_length'],
            pred_length = self.config['pred_length'],
            step = self.config['step'],
            mode = 'train',
            save_path = self.config['save_path']
        )
        
        # 数据集划分
        train_size = int(len(train_dataset) * self.config['train_ratio'])
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size],
            generator = torch.Generator().manual_seed(42)
        )
        
        # 数据加载器
        self.train_loader = DataLoader(train_dataset, batch_size = self.config['batch_size'], shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size = self.config['batch_size'])
        
        # 初始化模型
        self.model = ThermalTransformer(
            input_channels = len(self.config['input_cols']),
            output_channels = len(self.config['output_cols']),
            seq_length = self.config['seq_length'],
            pred_length = self.config['pred_length'],
            d_model = self.config['d_model'],
            nhead = self.config['nhead'],
            num_layers = self.config['num_layers'],
            dropout = self.config['dropout']
        )   
        
        # 训练设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 损失函数
        if self.config['criterion'] == "MSELoss":
            self.criterion = nn.MSELoss()
        elif self.config['criterion'] == "L1Loss":
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss function: {self.config['criterion']}")

        # 优化器
        if self.config['optimizer'] == "AdamW":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr = self.config['learning_rate'], 
                weight_decay = self.config['optimizer_params']['weight_decay'],
                amsgrad = self.config['optimizer_params']['amsgrad']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config['optimizer']}")
        
        # 学习率调度器
        if self.config['lr_scheduler'] == "ReduceLROnPlateau":
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode = self.config['lr_scheduler_params']['mode'], 
                factor = self.config['lr_scheduler_params']['factor'], 
                patience = self.config['lr_scheduler_params']['patience'], 
                verbose = self.config['lr_scheduler_params']['verbose']
            )
        else:
            raise ValueError(f"Unsupported learning rate scheduler: {self.config['lr_scheduler']}")
        
        # 创建保存路径
        if not os.path.exists(self.config['save_path']):
            os.makedirs(self.config['save_path'])

    def train(self):
        train_losses = []
        val_losses = []

        # 训练循环
        for epoch in range(self.config['num_epochs']):
            logging.info(f"{'-' * 5}Epoch {epoch}/{self.config['num_epochs'] - 1}{'-' * 5}")
            total_loss = 0
            start_time = time.time()

            self.model.train()
            
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
            
            # 验证集评估
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in self.val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    val_loss += self.criterion(outputs, targets).item()
            
            # 记录loss
            train_losses.append(total_loss/len(self.train_loader) * (1-self.config['train_ratio'])/self.config['train_ratio'])
            val_losses.append(val_loss/len(self.val_loader))

            # 更新学习率
            self.lr_scheduler.step(val_loss/len(self.val_loader))

            logging.info(f"Epoch {epoch+1:03d} | "
                f"Train Loss: {train_losses[-1]:.4f} | "
                f"Val Loss: {val_losses[-1]:.4f} | "
                f"Time: {time.time() - start_time:.2f}s | "
                f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}"
                )
        
        # 保存loss历史
        self.train_losses = train_losses
        self.val_losses = val_losses

        # 保存模型和标准化器
        torch.save({
            'model': self.model.state_dict(),
        }, os.path.join(self.config['save_path'], 'multi_output_transformer.pth'))

    def test(self):
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

        self.test_loader = DataLoader(self.test_dataset, batch_size = self.config['batch_size'])
    
        # 测试集评估
        self.model.eval()
        test_loss = 0

        self.outputs = None
        self.targets = None

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                test_loss += self.criterion(outputs, targets).item()

                if self.outputs is not None:
                    self.outputs = np.append(self.outputs, outputs.cpu().numpy(), axis=0)
                else:
                    self.outputs = outputs.cpu().numpy()
                
                if self.targets is not None:
                    self.targets = np.append(self.targets, targets.cpu().numpy(), axis=0)
                else:
                    self.targets = targets.cpu().numpy()

        self.test_losses = test_loss/len(self.test_loader)
        logging.info(f"Test Loss: {self.test_losses:.4f}")

    def plot(self):
        if not hasattr(self, 'outputs'):
            raise ValueError("No validation outputs available. Run training first.")
        
        visualizer = ThermalVisualizer(self.test_dataset, self.config['output_cols'], self.config['plot_delay'])

        # 绘制预测散点图
        visualizer.plot_predictions(
            self.outputs, 
            self.targets,
            save_path = os.path.join(self.config['save_path'], 'prediction_scatter.png')
        )

        # 绘制温度变化曲线
        visualizer.plot_temperature(
            self.outputs, 
            self.targets,
            save_path = os.path.join(self.config['save_path'], 'temperature_curve.png')
        )

        # 绘制损失函数变化曲线
        visualizer.plot_loss_curves(
            self.train_losses,
            self.val_losses,
            save_path = os.path.join(self.config['save_path'], 'loss_curves.png')
        )
    
    def score(self):
        real_outputs = inverse_transform_outputs(self.test_dataset, self.outputs)
        real_targets = inverse_transform_outputs(self.test_dataset, self.targets)

        r2_scores = [[] for _ in range(real_outputs.shape[1])]
        mse_scores = [[] for _ in range(real_outputs.shape[1])]
        mae_scores = [[] for _ in range(real_outputs.shape[1])]

        for delay in range(self.config['pred_length']):
            # 计算R2分数
            for i in range(real_outputs.shape[1]):
                r2 = r2_score(real_targets[:, i, delay], real_outputs[:, i, delay])
                r2_scores[i].append(r2)
            
            # 计算MSE
            for i in range(real_outputs.shape[1]):
                mse = np.mean((real_targets[:, i, delay] - real_outputs[:, i, delay]) ** 2)
                mse_scores[i].append(mse)
            
            # 计算MAE
            for i in range(real_outputs.shape[1]):
                mae = np.mean(np.abs(real_targets[:, i, delay] - real_outputs[:, i, delay]))
                mae_scores[i].append(mae)
        
        r2_average = np.mean(r2_scores, axis=1)
        mse_average = np.mean(mse_scores, axis=1)
        mae_average = np.mean(mae_scores, axis=1)

        logging.info("Scores:")
        for i in range(len(self.config["output_cols"])):
            logging.info(f"Output {self.config['output_cols'][i]} R2 Score: {r2_average[i]:.4f} | "
                         f"MSE: {mse_average[i]:.4f} | "
                         f"MAE: {mae_average[i]:.4f}"
                         )
        
        
if __name__ == "__main__":
    CONFIG["save_path"] = os.path.join(CONFIG["save_path"], datetime.strftime(datetime.now(), "%m%d-%H%M%S"))
    if not os.path.exists(CONFIG["save_path"]):
        os.makedirs(CONFIG["save_path"])

    setlogger(os.path.join(CONFIG["save_path"], "train.log"))
    for k, v in CONFIG.items():
        logging.info(f"{k}: {v}")
    
    trainer = MammalDSB(CONFIG)
    trainer.setup() 
    trainer.train()
    trainer.test()
    trainer.plot()
    trainer.score()