import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import graphviz
from sklearn.tree import export_graphviz

# 1. Load dataset
# data = pd.read_csv("./datasets/train.csv")
data = pd.read_csv("./datasets/processed/2024-01_processed.csv")

# 2. Define features and targets
feature_cols = [
    'LAE71AA101ZZ.AV', 'A1SPRFLOW.AV', 'LAE72AA101ZZ.AV', 'B1SPRFLOW.AV',
    'LAE73AA101ZZ.AV', 'A2SPRFLOW.AV', 'LAE74AA101ZZ.AV', 'B2SPRFLOW.AV'
]
target_cols = ['SHAOUTTE.AV', 'SHBOUTTE.AV']

# 3. Preprocess data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X = scaler_X.fit_transform(data[feature_cols])
y = scaler_y.fit_transform(data[target_cols])

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train RandomForestRegressor
model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# 6. Predict
y_pred = model.predict(X_test)

# 7. Inverse transform predictions and targets for visualization
y_pred_inv = scaler_y.inverse_transform(y_pred)
y_test_inv = scaler_y.inverse_transform(y_test)

# 8. Evaluate
mse = mean_squared_error(y_test_inv, y_pred_inv)
print(f"Test MSE: {mse:.4f}")

# 9. Plot predictions vs true values
for i, label in enumerate(target_cols):
    plt.figure(figsize=(10, 4))
    plt.plot(y_test_inv[:, i], label='True')
    plt.plot(y_pred_inv[:, i], label='Predicted')
    plt.title(f'{label} - Random Forest Prediction vs True')
    plt.xlabel('Sample Index')
    plt.ylabel(label)
    plt.legend()
    plt.tight_layout()
    plt.show()



def visualize_regression_tree(tree, feature_names, filename='regression_tree'):
    """
    可视化随机森林回归模型中的一棵回归树。

    :param tree: 训练好的单棵回归树 (rf.estimators_[i])
    :param feature_names: 特征名称列表
    :param filename: 输出文件名（不包含扩展名）
    """
    dot_data = export_graphviz(
        tree,
        out_file=None,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        special_characters=True
    )
    graph = graphviz.Source(dot_data)
    graph.render(filename)  # 保存为 .pdf 文件
    graph.view()  # 自动打开查看图形
    

from sklearn.ensemble import RandomForestRegressor


visualize_regression_tree(model.estimators_[0], feature_names=feature_cols, filename='rf_regression_tree_0')