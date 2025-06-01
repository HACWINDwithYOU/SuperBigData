CONFIG = {
    # 数据部分
    'data_folder': 'E:/Data/BigDataLecture/processed',
    'seq_length': 64,
    'pred_length': 20,
    'step': 20,
    'input_cols': ['LAE71AA101ZZ.AV', 'LAE72AA101ZZ.AV', 'LAE73AA101ZZ.AV', 'LAE71AA101ZZ.AV', 'PSHAOUTTE.AV', 'PSHBOUTTE.AV'],
    'output_cols': ['SHAOUTTE.AV', 'SHBOUTTE.AV'],
    'train_months': [1, 2, 3, 4, 5],
    'test_months': [6],

    # 模型部分
    'd_model': 32,
    'nhead': 2,
    'num_layers': 2,
    'dropout': 0.15,

    # 训练部分
    'batch_size': 32,
    'num_epochs': 5,
    'learning_rate': 1e-3,
    'train_ratio': 0.8,
    'criterion': "MSELoss",  # 损失函数
    
    'optimizer': "AdamW",  # 优化器
    'optimizer_params': {
        'weight_decay': 1e-4,
        'amsgrad': True
    },
    
    'lr_scheduler': "ReduceLROnPlateau",  # 学习率调度器
    'lr_scheduler_params': {
        'mode': 'min',
        'factor': 0.5,
        'patience': 5,
        'verbose': True
    },

    # 记录部分
    'save_path': './results',
    'plot_delay': 1
}