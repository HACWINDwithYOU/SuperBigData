# 超参数设置
WINDOW_SIZE = 8      # 时间窗口长度（例如24个时间步）
INPUT_DIM = 10        # x的维度
OUTPUT_DIM = 2        # y的维度
HIDDEN_DIM = 64       # LSTM隐藏层维度
NUM_LAYERS = 3        # LSTM层数
BATCH_SIZE = 4096
LEARNING_RATE = 2e-3
EPOCHS = 30

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# 自定义时间窗口数据集类
class WindowDataset(Dataset):
    def __init__(self, x_data, y_data, window_size):
        self.x = x_data
        self.y = y_data
        self.window = window_size

    def __len__(self):
        return len(self.x) - self.window * 2 + 1

    def __getitem__(self, idx):
        x_window = self.x[idx + self.window : idx + self.window * 2]
        y_past   = self.y[idx       : idx + self.window]
        y_target = self.y[idx + self.window : idx + self.window * 2]
        return (
            torch.tensor(x_window, dtype=torch.float32),
            torch.tensor(y_past, dtype=torch.float32),
            torch.tensor(y_target, dtype=torch.float32)
        )

# 数据读取和处理
INPUT_COLS = [
    "LAE71AA101ZZ.AV", "A1SPRFLOW.AV",
    "LAE72AA101ZZ.AV", "B1SPRFLOW.AV",
    "LAE73AA101ZZ.AV", "A2SPRFLOW.AV",
    "LAE74AA101ZZ.AV", "B2SPRFLOW.AV",
    "PSHAOUTTE.AV", "PSHBOUTTE.AV"
]
OUTPUT_COLS = ["SHAOUTTE.AV", "SHBOUTTE.AV"]

# 读取csv文件
df_train = pd.read_csv("./datasets/train.csv")
df_test = pd.read_csv("./datasets/test.csv")

# 归一化
from sklearn.preprocessing import StandardScaler
x_scaler = StandardScaler()
y_scaler = StandardScaler()

x_train = x_scaler.fit_transform(df_train[INPUT_COLS].values)
y_train = y_scaler.fit_transform(df_train[OUTPUT_COLS].values)

x_test = x_scaler.transform(df_test[INPUT_COLS].values)
y_test = y_scaler.transform(df_test[OUTPUT_COLS].values)

# 构建数据集
train_dataset = WindowDataset(x_train, y_train, WINDOW_SIZE)
test_dataset = WindowDataset(x_test, y_test, WINDOW_SIZE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 模型定义：Encoder-Decoder LSTM
class EncoderDecoderLSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super().__init__()
        self.encoder_x = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.encoder_y = nn.LSTM(output_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(output_dim, hidden_dim, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_seq, y_past):
        _, (hx_x, cx_x) = self.encoder_x(x_seq)
        _, (hx_y, cx_y) = self.encoder_y(y_past)

        hx = hx_x + hx_y
        cx = cx_x + cx_y

        decoder_input = y_past[:, -1:, :]
        outputs = []
        for _ in range(x_seq.size(1)):
            out, (hx, cx) = self.decoder(decoder_input, (hx, cx))
            pred = self.output_layer(out)
            outputs.append(pred)
            decoder_input = pred

        return torch.cat(outputs, dim=1)

# 训练函数
def train_model(model, train_loader, epochs, lr):
    model = model.to(device := torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    best_loss = float('inf')
    best_model = None
    train_losses = []
    lrs = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for x, y_past, y_true in tqdm(train_loader, desc="Training"):
            x, y_past, y_true = x.to(device), y_past.to(device), y_true.to(device)
            optimizer.zero_grad()
            y_pred = model(x, y_past)
            loss = criterion(y_pred, y_true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * x.size(0)

        scheduler.step()
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        lrs.append(optimizer.param_groups[0]['lr'])

        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.8f} | LR: {lrs[-1]:.6f}")

        if train_loss < best_loss:
            best_loss = train_loss
            best_model = model.state_dict()

    print("Training complete. Best train loss:", best_loss)
    model.load_state_dict(best_model)
    return model, train_losses, lrs

# 滚动预测函数
def rolling_forecast(model, x_all, window_size):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    total_len = len(x_all)
    pred_y_all = torch.zeros((total_len, OUTPUT_DIM), dtype=torch.float32).to(device)
    weight = torch.zeros((total_len, OUTPUT_DIM), dtype=torch.float32).to(device)
    y_past = torch.zeros((1, window_size, OUTPUT_DIM), dtype=torch.float32).to(device)

    for t in tqdm(range(window_size, total_len - window_size), desc="Rolling Forecast"):
        x_window = torch.tensor(x_all[t:t+window_size], dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred_seq = model(x_window, y_past)

        for i in range(window_size):
            pred_y_all[t+i, :] = pred_y_all[t+i, :] * weight[t+i, :] + y_pred_seq[0, i, :]
            weight[t+i, :] += 1
            pred_y_all[t+i, :] /= weight[t+i, :]

        y_past = pred_y_all[t-window_size+1:t+1, :].unsqueeze(0)

    return y_scaler.inverse_transform(pred_y_all.cpu().numpy())

# =============== 主程序入口 ================
if __name__ == "__main__":
    model = EncoderDecoderLSTM(INPUT_DIM, OUTPUT_DIM, HIDDEN_DIM, NUM_LAYERS)
    model, train_losses, lrs = train_model(model, train_loader, EPOCHS, LEARNING_RATE)

    torch.save(model.state_dict(), './models/EDLSTM-model.pth')

    # 可视化训练过程
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(train_losses[1:], label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss")

    plt.subplot(1,2,2)
    plt.plot(lrs, label="Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.title("Learning Rate Schedule")
    plt.tight_layout()
    plt.show()

    # 测试集滚动预测
    y_pred = rolling_forecast(model, x_test, WINDOW_SIZE)
    y_true = y_scaler.inverse_transform(y_test)

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np



    # 可视化预测结果
    plt.figure(figsize=(12, 6))
    for i in range(OUTPUT_DIM):
        plt.subplot(OUTPUT_DIM, 1, i+1)
        plt.plot(y_true[:, i], label=f"True Y{i}")
        plt.plot(y_pred[:, i], label=f"Pred Y{i}")
        plt.legend()
        plt.title(f"Prediction vs Ground Truth for Y{i}")
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true[:, i], y_pred[:, i])

        # 输出结果
        print(f"MAE:  {mae:.4f}")
        print(f"MSE:  {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2:   {r2:.4f}")
    plt.tight_layout()
    plt.show()
