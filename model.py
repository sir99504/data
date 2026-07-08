import torch
import config
import torch.nn as nn
import torch.nn.functional as F
from components.convlstm import ConvLSTM


class SE3DTimeNet(nn.Module):
    def __init__(self, time_steps, reduction=16):
        super(SE3DTimeNet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(time_steps, time_steps // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(time_steps // reduction, time_steps, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, t, c, h, w = x.size()
        y = self.avg_pool(x)
        y = y.view(b, t)
        y = self.fc(y)
        y = y.view(b, t, 1, 1, 1)
        return x * y.expand_as(x)


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim, seq_len):
        super(TemporalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False)
        )

    def forward(self, x):
        b, t, c, h, w = x.size()
        spatial_avg = x.mean(dim=[3, 4])
        scores = self.attn(spatial_avg)
        weights = torch.softmax(scores, dim=1)
        weights = weights.view(b, t, 1, 1, 1)
        context_vector = torch.sum(x * weights, dim=1)
        return context_vector


class LSTMModel(nn.Module):
    def __init__(self, cfg):
        super(LSTMModel, self).__init__()
        in_channels = cfg["input_size"]
        hidden_channels = cfg["hidden_size"]
        out_channels = 1

        self.drop = nn.Dropout(p=cfg["dropout_rate"])
        self.lstm = nn.LSTM(in_channels, hidden_channels, batch_first=True)
        self.liner1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.liner2 = nn.Linear(hidden_channels // 2, hidden_channels // 2)
        self.liner3 = nn.Linear(hidden_channels // 2, out_channels)

    def forward(self, inputs):
        inputs_new = inputs
        x, _ = self.lstm(inputs_new.float())
        x = self.drop(x)
        x = F.relu(self.liner1(x[:, -1]))
        x = F.dropout(x)
        x = self.liner2(x).relu()
        x = F.dropout(x)
        x = self.liner3(x).relu()
        return x


class GraphLSTMModel(nn.Module):
    def __init__(self, cfg):
        super(GraphLSTMModel, self).__init__()
        H = 2 * cfg["kernel_size"] + 1
        W = 2 * cfg["kernel_size"] + 1
        out_channels = 1
        in_channels = cfg["input_size"] * H * W
        hidden_channels = cfg["hidden_size"]

        self.drop = nn.Dropout(p=cfg["dropout_rate"])
        self.lstm = nn.LSTM(in_channels, hidden_channels, batch_first=True)
        self.liner1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.liner2 = nn.Linear(hidden_channels // 2, hidden_channels // 2)
        self.liner3 = nn.Linear(hidden_channels // 2, out_channels * H * W)

    def forward(self, inputs):
        inputs_new = inputs
        x, _ = self.lstm(inputs_new.float())
        x = self.drop(x)
        x = F.relu(self.liner1(x[:, -1]))
        x = F.dropout(x)
        x = self.liner2(x).relu()
        x = F.dropout(x)
        x = self.liner3(x).relu()
        return x


class ConvLSTMModel(nn.Module):
    def __init__(self, cfg):
        super(ConvLSTMModel, self).__init__()
        spatial_size = int(2 * cfg["spatial_offset"] + 1)
        hidden_size_half = int(cfg["hidden_size"] / 2)

        self.ConvLSTM_net = ConvLSTM(
            input_size=(spatial_size, spatial_size),
            input_dim=int(cfg["input_size"]),
            hidden_dim=[int(cfg["hidden_size"]), hidden_size_half],
            kernel_size=(int(cfg["kernel_size"]), int(cfg["kernel_size"])),
            num_layers=2,
            cfg=cfg,
            batch_first=True
        )
        self.drop = nn.Dropout(p=cfg["dropout_rate"])
        dense_input_size = hidden_size_half * spatial_size * spatial_size
        self.dense = nn.Linear(dense_input_size, 1)

    def forward(self, inputs):
        threshold = torch.nn.Threshold(0., 0.0)
        hidden = self.ConvLSTM_net.get_init_states(inputs.shape[0])
        last_state, encoder_state = self.ConvLSTM_net(
            inputs.clone(), hidden
        )
        last_state = self.drop(last_state)
        Convout = last_state[:, -1, :, :, :]
        shape = Convout.shape[0]
        Convout = Convout.reshape(shape, -1)
        Convout = torch.flatten(Convout, 1)
        Convout = threshold(Convout)
        predictions = self.dense(Convout)
        return predictions

class GraphConvLSTMModel(nn.Module):
    def __init__(self, cfg):
        super(GraphConvLSTMModel, self).__init__()
        spatial_size = int(2 * cfg["spatial_offset"] + 1)
        hidden_size_half = int(cfg["hidden_size"] / 2)

        self.ConvLSTM_net = ConvLSTM(
            input_size=(spatial_size, spatial_size),
            input_dim=int(cfg["input_size"]),
            hidden_dim=[int(cfg["hidden_size"]), hidden_size_half],
            kernel_size=(int(cfg["kernel_size"]), int(cfg["kernel_size"])),
            num_layers=2,
            cfg=cfg,
            batch_first=True
        )
        self.drop = nn.Dropout(p=cfg["dropout_rate"])
        self.conv = nn.Conv2d(hidden_size_half, 1, kernel_size=1, padding=0)
        
        # ========== 使用Sigmoid（因为标签在0-0.6之间） ==========
        self.sigmoid = nn.Sigmoid()
        
        # ========== 初始化权重（重要！） ==========
        self._initialize_weights()
        
    def _initialize_weights(self):
        """
        初始化模型权重，防止梯度消失/爆炸
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    # 对卷积层和全连接层使用Kaiming初始化
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                else:
                    # 其他权重使用小随机值初始化
                    nn.init.normal_(param, mean=0, std=0.01)
            elif 'bias' in name:
                # 偏置初始化为小正值，避免ReLU死亡
                nn.init.constant_(param, 0.1)
        
        print("DEBUG: 模型权重初始化完成")
        
    def forward(self, inputs):
        
        # 初始化ConvLSTM的隐藏状态
        hidden = self.ConvLSTM_net.get_init_states(inputs.shape[0])
        
        # ConvLSTM前向传播
        last_state, encoder_state = self.ConvLSTM_net(inputs.clone(), hidden)
        
        # 取最后一个时间步 [batch, channels, height, width]
        last_time_step = last_state[:, -1, :, :, :]
        
        # Dropout和激活
        last_time_step = self.drop(last_time_step)
        
        # 最终的卷积层
        predictions = self.conv(last_time_step)
        
        # Sigmoid激活，输出在(0,1)之间
        predictions = self.sigmoid(predictions)
        
        # 调试信息（只在第一次前向传播时打印）
        if not hasattr(self, 'debug_printed'):
            self.debug_printed = True
            print(f"DEBUG: inputs shape: {inputs.shape}")
            print(f"DEBUG: last_state shape: {last_state.shape}")
            print(f"DEBUG: last_time_step shape: {last_time_step.shape}")
            print(f"DEBUG: predictions shape: {predictions.shape}")
            print(f"DEBUG: predictions range: [{predictions.min().item():.6f}, {predictions.max().item():.6f}]")
            
            # 检查ConvLSTM输出范围
            print(f"DEBUG: ConvLSTM output range: [{last_state.min().item():.6f}, {last_state.max().item():.6f}]")
            print(f"DEBUG: conv weight range: [{self.conv.weight.min().item():.6f}, {self.conv.weight.max().item():.6f}]")
        
        return predictions

if __name__ == '__main__':
    args = config.get_args()
    seq_len = args['seq_len']
    batch_size = args['batch_size']
    hidden_channels = args['hidden_size']
    in_channels = 5

    model = GraphConvLSTMModel(args).to(args['device'])

    inputs = torch.randn(size=(batch_size, seq_len, in_channels, 7, 7))
    inputs = inputs.to(args['device'])
    output = model(inputs)
    print(output.shape)