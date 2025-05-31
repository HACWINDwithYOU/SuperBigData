# 文件名：train_autoregressive_model.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# ===== 配置 =====
INPUT_COLS = [
    "LAE71AA101ZZ.AV", "A1SPRFLOW.AV",
    "LAE72AA101ZZ.AV", "B1SPRFLOW.AV",
    "LAE73AA101ZZ.AV", "A2SPRFLOW.AV",
    "LAE74AA101ZZ.AV", "B2SPRFLOW.AV",
    "PSHAOUTTE.AV", "PSHBOUTTE.AV"
]
OUTPUT_COLS = ["SHAOUTTE.AV", "SHBOUTTE.AV"]
# SEQ_LEN = 1  # 每次仅使用当前时刻
BATCH_SIZE = 4096
EPOCHS = 60
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 数据集类 =====
class TimeSeriesDataset(Dataset):
    def __init__(self, df, input_cols, output_cols):
        self.x = df[input_cols].values.astype(np.float32)
        self.y = df[output_cols].values.astype(np.float32)
        # 构造 y_{t-1}，第一项填 0
        self.y_prev = np.vstack([np.zeros((1, len(output_cols))), self.y[:-1]])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_t = self.x[idx]
        y_t = self.y[idx]
        y_prev_t = self.y_prev[idx]
        x_aug = np.concatenate([x_t, y_prev_t], axis=0)
        return x_aug, y_t

# ===== 模型定义 =====
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len=1, input_dim)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out  # shape: (batch, seq_len, output_dim)

# ===== 训练函数 =====
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    losses = []
    for x_aug, y in tqdm(dataloader, desc="Training"):
        x_aug = x_aug.to(device).float()  # shape: (batch, input_dim + output_dim)
        y = y.to(device).float()

        x_aug = x_aug.unsqueeze(1)  # add seq_len dim: (batch, 1, input_dim + output_dim)
        output = model(x_aug).squeeze(1)  # (batch, output_dim)

        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    return np.mean(losses)

# ===== 自回归预测函数（测试阶段） =====
def autoregressive_predict(model, x_all, device):
    model.eval()
    x_all = torch.tensor(x_all, dtype=torch.float32).to(device)
    y_prev = torch.zeros((1, 1, len(OUTPUT_COLS)), dtype=torch.float32).to(device)

    preds = []
    with torch.no_grad():
        for t in tqdm(range(len(x_all)), desc="Autoregressive Predict"):
            x_t = x_all[t].unsqueeze(0).unsqueeze(0)  # (1, 1, input_dim)
            x_aug = torch.cat([x_t, y_prev], dim=-1)  # (1, 1, input_dim + output_dim)

            y_pred = model(x_aug)[:, -1, :]  # (1, output_dim)
            preds.append(y_pred.cpu().numpy())
            y_prev = y_pred.unsqueeze(1)  # 更新 y_prev

    preds = np.concatenate(preds, axis=0)  # (T, output_dim)
    return preds

# ===== 主程序入口 =====
def main():
    # ==== 1. 加载数据 ====
    # df_train = pd.read_csv("./datasets/processed/2024-01_processed.csv")
    df_train = pd.read_csv("./datasets/train.csv")
    df_test = pd.read_csv("./datasets/test.csv")
    x_test = df_test[INPUT_COLS].values
    y_test = df_test[OUTPUT_COLS].values

    # ==== 2. 构造训练集 ====
    train_dataset = TimeSeriesDataset(df_train, INPUT_COLS, OUTPUT_COLS)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ==== 3. 构造模型 ====
    input_dim = len(INPUT_COLS)
    output_dim = len(OUTPUT_COLS)
    model = LSTMRegressor(input_size=input_dim + output_dim, hidden_size=64, output_size=output_dim).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ==== 4. 训练模型 ====
    train_losses = []
    for epoch in range(EPOCHS):
        loss = train(model, train_loader, criterion, optimizer, DEVICE)
        train_losses.append(loss)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}")

    torch.save(model.state_dict(), './models/model.pth')

    # ==== 5. 可视化训练损失 ====
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ==== 6. 测试阶段自回归预测 ====
    preds = autoregressive_predict(model, x_test, DEVICE)

    # ==== 7. 可视化预测结果 ====
    for i, col in enumerate(OUTPUT_COLS):
        plt.figure(figsize=(10, 4))
        plt.plot(y_test[:, i], label=f"True {col}", linestyle="--")
        plt.plot(preds[:, i], label=f"Predicted {col}")
        plt.xlabel("Time Step")
        plt.ylabel(col)
        plt.title(f"Autoregressive Prediction for {col}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
