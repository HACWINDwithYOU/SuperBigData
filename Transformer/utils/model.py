import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class ThermalTransformer(nn.Module):
    def __init__(self, input_channels=6, output_channels=2, seq_length=12, pred_length=3,
                 d_model=64, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.pred_length = pred_length
        self.output_channels = output_channels
        
        # 输入处理
        self.input_proj = nn.Linear(input_channels, d_model)
        self.pos_encoder = PositionalEncoding(d_model, seq_length)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
            dropout=dropout, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 多输出头
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, output_channels*pred_length)  # 最终输出2个维度的温度
        )

    def forward(self, x):
        x = x.permute(2, 0, 1) # [batch, features, seq] -> [seq, batch, features]
        x = self.input_proj(x) * math.sqrt(self.d_model) # -> [seq, batch, d_model]
        x = self.pos_encoder(x) 
        x = self.transformer_encoder(x)
        x = self.output_layer(x[-1]) # 取最后一个时间步的输出 -> [batch, output_dim]
        out = x.view(-1, self.output_channels, self.pred_length) # reshape 
        return out # -> (batch, output_dim=2, pred_length)