import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from captum.attr import Saliency, IntegratedGradients
from attention_lstm import AutoregressiveLSTM
from sklearn.preprocessing import MinMaxScaler

# Load data and model
print("Loading data and model...")
data = pd.read_csv("./datasets/train.csv")
model = torch.load("./artifacts/model.pt", map_location=torch.device('cpu'))
model.eval()

# Define input/target columns
input_cols = ['LAE71AA101ZZ.AV', 'A1SPRFLOW.AV', 'LAE72AA101ZZ.AV', 'B1SPRFLOW.AV',
              'LAE73AA101ZZ.AV', 'A2SPRFLOW.AV', 'LAE74AA101ZZ.AV', 'B2SPRFLOW.AV',
              'PSHAOUTTE.AV', 'PSHBOUTTE.AV']
target_cols = ['SHAOUTTE.AV', 'SHBOUTTE.AV']

# Preprocess
scaler = MinMaxScaler()
data[input_cols + target_cols] = scaler.fit_transform(data[input_cols + target_cols])

# Extract window for explanation
window_size = 60
future_steps = 10
sample_df = data[input_cols + target_cols].iloc[:(window_size + future_steps)]
input_seq = sample_df[input_cols].values[:window_size]  # shape (60, 10)
input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)  # shape (1, 60, 10)
input_tensor.requires_grad = True

# Choose method
method = 'saliency'  # Change to 'ig' for IntegratedGradients
if method == 'saliency':
    explainer = Saliency(model)
elif method == 'ig':
    explainer = IntegratedGradients(model)
else:
    raise ValueError("Unsupported method")

# Attribution
print("Computing feature attributions...")
attributions = explainer.attribute(input_tensor, target=0)  # shape: (1, 60, 10)
attributions = attributions.squeeze(0).detach().numpy()  # shape: (60, 10)

# Visualize feature importances
plt.figure(figsize=(14, 6))
for i, col in enumerate(input_cols):
    plt.plot(range(window_size), attributions[:, i], label=col)
plt.title(f"Feature Attributions using {method.title()} (target=SHAOUTTE.AV[0])")
plt.xlabel("Timestep")
plt.ylabel("Attribution")
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
plt.tight_layout()
plt.show()
