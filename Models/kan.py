import torch
import torch.nn as nn


# 定义KAN层
class KANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(KANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Univariate functions for each input dimension
        self.univariate_funcs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(input_dim)
        ])
        self.combiner = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Ensure input x is on the same device as the model
        device = next(self.parameters()).device
        x = x.to(device)

        # Apply univariate functions to each feature
        univariate_outputs = [func(x[:, i].unsqueeze(1)) for i, func in enumerate(self.univariate_funcs)]
        univariate_outputs = torch.cat(univariate_outputs, dim=1)
        return self.combiner(univariate_outputs)


# 定义自定义的KAN-LSTM单元
class KANLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kan_hidden_dim):
        super(KANLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kan_hidden_dim = kan_hidden_dim

        # KAN layers for LSTM gates
        self.kan_forget = KANLayer(input_dim + hidden_dim, hidden_dim, kan_hidden_dim)
        self.kan_input = KANLayer(input_dim + hidden_dim, hidden_dim, kan_hidden_dim)
        self.kan_output = KANLayer(input_dim + hidden_dim, hidden_dim, kan_hidden_dim)
        self.kan_cell = KANLayer(input_dim + hidden_dim, hidden_dim, kan_hidden_dim)

    def forward(self, x, h_prev, c_prev):
        # Ensure all inputs are on the same device
        device = next(self.parameters()).device
        x, h_prev, c_prev = x.to(device), h_prev.to(device), c_prev.to(device)

        combined = torch.cat([x, h_prev], dim=1)
        f_t = torch.sigmoid(self.kan_forget(combined))  # AscendingTensor(self.kan_input(combined))   # Input gate
        i_t = torch.sigmoid(self.kan_input(combined))   # Input gate
        o_t = torch.sigmoid(self.kan_output(combined))  # Output gate
        c_tilde = torch.tanh(self.kan_cell(combined))   # Cell candidate
        c_t = f_t * c_prev + i_t * c_tilde              # Update cell state
        h_t = o_t * torch.tanh(c_t)                     # Update hidden state
        return h_t, c_t


# 定义KAN-LSTM的前向传播函数
def kan_lstm_forward(inputs, kan_lstm_cell, initial_h, initial_c):
    device = next(kan_lstm_cell.parameters()).device
    initial_h, initial_c = initial_h.to(device), initial_c.to(device)
    inputs = inputs.to(device)

    h = initial_h
    c = initial_c
    outputs = []
    for t in range(inputs.size(0)):
        x = inputs[t]
        h, c = kan_lstm_cell(x, h, c)
        outputs.append(h)
    return torch.stack(outputs, dim=0), h, c


# 示例用法
if __name__ == "__main__":
    # 设置参数
    input_dim = 10
    hidden_dim = 20
    kan_hidden_dim = 16
    seq_len = 5
    batch_size = 32

    # 初始化模型
    kan_lstm_cell = KANLSTMCell(input_dim, hidden_dim, kan_hidden_dim)

    # 生成随机输入数据
    inputs = torch.randn(seq_len, batch_size, input_dim)
    h0 = torch.zeros(batch_size, hidden_dim)
    c0 = torch.zeros(batch_size, hidden_dim)

    # 前向传播
    outputs, h_n, c_n = kan_lstm_forward(inputs, kan_lstm_cell, h0, c0)

    print("输出形状:", outputs.shape)  # 应为 (seq_len, batch_size, hidden_dim)
    print("最终隐藏状态形状:", h_n.shape)  # 应为 (batch_size, hidden_dim)
    print("最终细胞状态形状:", c_n.shape)  # 应为 (batch_size, hidden_dim)
