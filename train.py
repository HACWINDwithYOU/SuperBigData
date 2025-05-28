import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import random
from tqdm import tqdm

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# 1. Load data
print("Loading dataset...")
data = pd.read_csv("./datasets/train.csv")

# Select specific profile_ids as test set
print("Splitting test/train-validation datasets...")

# Split train/val/test based on profile_id
test_data = pd.read_csv("./datasets/test.csv")
train_val_data = data

# Normalize
target_cols = ['SHAOUTTE.AV','SHBOUTTE.AV']
scaler = MinMaxScaler()
train_val_data = train_val_data.copy()
test_data = test_data.copy()
train_val_data.loc[:, target_cols] = scaler.fit_transform(train_val_data[target_cols])
test_data.loc[:, target_cols] = scaler.transform(test_data[target_cols])

# Hyperparameters
window_size = 60
future_steps = 10
batch_size = 1024
epochs = 10
lr = 1e-3

# Dataset class
class SequenceDataset(Dataset):
    def __init__(self, df):
        self.inputs = []
        self.targets = []
        grouped = [df]
        for group in grouped:
            values = group[target_cols].values
            for i in range(len(values) - window_size - future_steps):
                self.inputs.append(values[i:i+window_size])
                self.targets.append(values[i+window_size:i+window_size+future_steps])
        self.inputs = torch.tensor(np.array(self.inputs), dtype=torch.float32)
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# Split train/val
print("Preparing train/val splits...")
train_ratio = 0.8
split_idx = int(len(train_val_data['date']) * train_ratio)
profile_ids = train_val_data['date']
train_ids = profile_ids[:split_idx]
val_ids = profile_ids[split_idx:]

train_df = train_val_data[train_val_data['date'].isin(train_ids)]
val_df = train_val_data[train_val_data['date'].isin(val_ids)]

train_dataset = SequenceDataset(train_df)
val_dataset = SequenceDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Autoregressive LSTM
class AutoregressiveLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x, future_steps):
        outputs = []
        _, (h, c) = self.lstm(x)
        input_seq = x[:, -1:, :]
        for _ in range(future_steps):
            out, (h, c) = self.lstm(input_seq, (h, c))
            out_step = self.linear(out[:, -1, :]).unsqueeze(1)
            outputs.append(out_step)
            input_seq = out_step
        return torch.cat(outputs, dim=1)

# Training preparation
model = AutoregressiveLSTM(input_size=len(target_cols), hidden_size=64, num_layers=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

train_losses, val_losses, val_mse, lrs, gaps = [], [], [], [], []

# Train
print("Starting training...")
for epoch in tqdm(range(epochs), desc="Training Epochs"):
    model.train()
    epoch_train_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x, future_steps)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    train_loss = epoch_train_loss / len(train_loader)

    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            output = model(x, future_steps)
            loss = criterion(output, y)
            epoch_val_loss += loss.item()
    val_loss = epoch_val_loss / len(val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_mse.append(val_loss)
    lrs.append(scheduler.get_last_lr()[0])
    gaps.append(train_loss - val_loss)
    scheduler.step()

# Plot training curves
print("Plotting training curves...")
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Val")
plt.title("Loss")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(val_mse)
plt.title("Val MSE")

plt.subplot(2, 2, 3)
plt.plot(lrs)
plt.title("Learning Rate")

plt.subplot(2, 2, 4)
plt.plot(gaps)
plt.title("Generalization Gap")
plt.tight_layout()
plt.show()

# Test set evaluation with EWA
print("Evaluating on test set...")
test_dataset = SequenceDataset(test_data)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
model.eval()
preds, trues = [], []
alpha = 0.6  # EMA smoothing

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        output = model(x, future_steps)  # (B, T, D)
        output = output.cpu().numpy()    # 移到CPU一次性处理
        y = y[:, -1].numpy()             # 获取每个样本的最终真实值 (B, D)

        # 批量 EWA
        B, T, D = output.shape
        weights = np.array([(1 - alpha) ** i for i in range(T)][::-1])
        weights /= weights.sum()
        weights = weights.reshape(1, T, 1)  # (1, T, 1) for broadcasting
        ewa = (output * weights).sum(axis=1)  # (B, D)

        preds.append(ewa)
        trues.append(y)

preds = np.concatenate(preds, axis=0)
trues = np.concatenate(trues, axis=0)


for i, label in enumerate(target_cols):
    plt.figure()
    plt.plot([p[i] for p in preds], label='Predicted')
    plt.plot([t[i] for t in trues], label='True')
    plt.title(f"{label} Prediction vs True")
    plt.legend()
    plt.show()
