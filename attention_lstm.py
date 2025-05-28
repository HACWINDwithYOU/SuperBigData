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
data = pd.read_csv("./datasets/train.csv")
test_data = pd.read_csv("./datasets/test.csv")

# Define input and output columns
all_columns = data.columns.tolist()
target_cols = ['SHAOUTTE.AV', 'SHBOUTTE.AV']
input_cols = [col for col in all_columns if col not in ['date'] + target_cols]

# Normalize
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
data[input_cols] = scaler_x.fit_transform(data[input_cols])
data[target_cols] = scaler_y.fit_transform(data[target_cols])
test_data[input_cols] = scaler_x.transform(test_data[input_cols])
test_data[target_cols] = scaler_y.transform(test_data[target_cols])

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
        values_x = df[input_cols].values
        values_y = df[target_cols].values
        for i in range(len(df) - window_size - future_steps):
            self.inputs.append(values_x[i:i+window_size])
            self.targets.append(values_y[i+window_size:i+window_size+future_steps])
        self.inputs = torch.tensor(np.array(self.inputs), dtype=torch.float32)
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# Split train/val
train_ratio = 0.8
split_idx = int(len(data) * train_ratio)
train_df = data.iloc[:split_idx]
val_df = data.iloc[split_idx:]

train_dataset = SequenceDataset(train_df)
val_dataset = SequenceDataset(val_df)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Model
class AutoregressiveLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, len(target_cols))

    def forward(self, x, future_steps):
        outputs = []
        _, (h, c) = self.lstm(x)
        input_seq = x[:, -1:, :]
        for _ in range(future_steps):
            out, (h, c) = self.lstm(input_seq, (h, c))
            out_step = self.linear(out[:, -1, :]).unsqueeze(1)
            outputs.append(out_step)
            input_seq = out_step.repeat(1, 1, x.shape[2] // len(target_cols))
        return torch.cat(outputs, dim=1)

# Training setup
model = AutoregressiveLSTM(input_size=len(input_cols), hidden_size=64, num_layers=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

train_losses, val_losses = [], []

# Training loop
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
    preds, trues = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            output = model(x, future_steps)
            loss = criterion(output, y)
            epoch_val_loss += loss.item()
            preds.append(output[:, -1].cpu().numpy())
            trues.append(y[:, -1].cpu().numpy())

    val_loss = epoch_val_loss / len(val_loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    scheduler.step()

print("Plotting predictions vs ground truth...")
preds = np.concatenate(preds, axis=0)
trues = np.concatenate(trues, axis=0)
preds = scaler_y.inverse_transform(preds)
trues = scaler_y.inverse_transform(trues)

for i, label in enumerate(target_cols):
    plt.figure(figsize=(10, 4))
    plt.plot(preds[:, i], label='Predicted')
    plt.plot(trues[:, i], label='True')
    plt.title(f"{label} Prediction vs True")
    plt.legend()
    plt.tight_layout()
    plt.show()
