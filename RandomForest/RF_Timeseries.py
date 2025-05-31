import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.fftpack import fft
import pywt
import matplotlib.pyplot as plt

# ------------------- 参数配置 ------------------- #
WINDOW_SIZE = 60  # 60秒窗口
IMPORTANT_FEATURES = ['PSHBOUTTE.AV', 'PSHAOUTTE.AV', 'LAE72AA101ZZ.AV', 'LAE71AA101ZZ.AV', 'LAE74AA101ZZ.AV']
ALL_FEATURES = [
    'LAE71AA101ZZ.AV', 'A1SPRFLOW.AV', 'LAE72AA101ZZ.AV', 'B1SPRFLOW.AV',
    'LAE73AA101ZZ.AV', 'A2SPRFLOW.AV', 'LAE74AA101ZZ.AV', 'B2SPRFLOW.AV',
    'PSHAOUTTE.AV', 'PSHBOUTTE.AV'
]
TARGET = 'SHAOUTTE.AV'

# ------------------- 特征提取函数 ------------------- #
def extract_features_for_window(window: pd.DataFrame, important_features: list):
    feats = []
    for col in window.columns:
        x = window[col].values
        feats.extend([np.mean(x), np.std(x), np.min(x), np.max(x)])

        if col in important_features:
            fft_vals = np.abs(fft(x))[:len(x)//2]
            feats.extend(np.sort(fft_vals)[-3:])

            coeffs = pywt.wavedec(x, wavelet='db4', level=3)
            for c in coeffs:
                feats.extend([np.mean(c), np.std(c)])
    return feats

# ------------------- 数据加载与处理 ------------------- #
data = pd.read_csv("../datasets/train.csv")
# data = pd.read_csv("../datasets/processed/2024-01_processed.csv")
testdata = pd.read_csv("../datasets/test.csv")

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
data[ALL_FEATURES] = scaler_X.fit_transform(data[ALL_FEATURES])
data[TARGET] = scaler_y.fit_transform(data[[TARGET]])



X_feat, y_feat = [], []
for i in range(0, len(data) - WINDOW_SIZE, WINDOW_SIZE):
    window = data.iloc[i:i+WINDOW_SIZE]
    feat = extract_features_for_window(window[ALL_FEATURES], IMPORTANT_FEATURES)
    target = data.iloc[i + WINDOW_SIZE - 1][TARGET]  # 取窗口末值作为目标
    X_feat.append(feat)
    y_feat.append(target)

X = np.array(X_feat)
y = np.array(y_feat)

testdata[ALL_FEATURES] = scaler_X.fit_transform(testdata[ALL_FEATURES])
testdata[TARGET] = scaler_y.fit_transform(testdata[[TARGET]])



X_test_feat, y_test_feat = [], []
for i in range(0, len(testdata) - WINDOW_SIZE, WINDOW_SIZE):
    window = testdata.iloc[i:i+WINDOW_SIZE]
    feat = extract_features_for_window(window[ALL_FEATURES], IMPORTANT_FEATURES)
    target = testdata.iloc[i + WINDOW_SIZE - 1][TARGET]  # 取窗口末值作为目标
    X_test_feat.append(feat)
    y_test_feat.append(target)

X_test = np.array(X_test_feat)
y_test = np.array(y_test_feat)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------- 训练随机森林 ------------------- #

model = RandomForestRegressor(random_state=42,n_estimators=50,min_samples_split=2,min_samples_leaf=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ------------------- 评估与可视化 ------------------- #
y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1))

print(f"RMSE: {np.sqrt(mean_squared_error(y_test_inv, y_pred_inv)):.4f}")
print(f"R2 Score: {r2_score(y_test_inv, y_pred_inv):.4f}")

plt.figure(figsize=(10,4))
plt.plot(y_test_inv, label='True')
plt.plot(y_pred_inv, label='Pred')
plt.title('Random Forest Prediction')
plt.legend()
plt.show()
