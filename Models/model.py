import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from Models.AbsolutePositionalEncoding import tAPE, AbsolutePositionalEncoding, LearnablePositionalEncoding
from Models.Attention import Attention, Attention_Rel_Scl, Attention_Rel_Vec
from Models.kan import KANLSTMCell, kan_lstm_forward


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Permute(nn.Module):
    def forward(self, x):
        return x.permute(1, 0, 2)


class _Permute(nn.Module):
    def __init__(self, *dims):
        super(_Permute, self).__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor: return x.permute(self.dims)

    def __repr__(self): return f"{self.__class__.__name__}(dims={', '.join([str(d) for d in self.dims])})"


def model_factory(config):
    if config['Net_Type'][0] == 'T':
        model = Transformer(config, num_classes=config['num_labels'])
    elif config['Net_Type'][0] == 'C':
        model = CasualConvTran(config, num_classes=config['num_labels'])
    # elif config['Net_Type'][0] == 'F':
    #     model = MLSTMfcn(config, num_classes=config['num_labels'])
    elif config['Net_Type'][0] == 'F':
        model = EnhancedLSTMfcn(config, num_classes=config['num_labels'])
    elif config['Net_Type'][0] == 'E':
        model = EnhancedMLSTMfcn(config, num_classes=config['num_labels'])
    elif config['Net_Type'][0] == 'k':
        model = EnhancedLSTMfcnKan(config, num_classes=config['num_labels'])
    # elif config['Net_Type'][0] == 'M':
    #     model = OwnMLSTMfcn(config, num_classes=config['num_labels'])
    # elif config['Net_Type'][0] == 'O':
    #     model = NewOMLSTMfcn(config, num_classes=config['num_labels'])
    elif config['Net_Type'][0] == 'K':
        model = OMKANfcn(config, num_classes=config['num_labels'])
    elif config['Net_Type'][0] == 'L':
        model = OMLKANfcn(config, num_classes=config['num_labels'])
    else:
        model = ConvTran(config, num_classes=config['num_labels'])
    return model


class Transformer(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        # Parameters Initialization -----------------------------------------------
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        emb_size = config['emb_size']
        num_heads = config['num_heads']
        dim_ff = config['dim_ff']
        self.Fix_pos_encode = config['Fix_pos_encode']
        self.Rel_pos_encode = config['Rel_pos_encode']
        # Embedding Layer -----------------------------------------------------------
        self.embed_layer = nn.Sequential(
            nn.Linear(channel_size, emb_size),
            nn.LayerNorm(emb_size, eps=1e-5)
        )

        if self.Fix_pos_encode == 'Sin':
            self.Fix_Position = tAPE(emb_size, dropout=config['dropout'], max_len=seq_len)
        elif config['Fix_pos_encode'] == 'Learn':
            self.Fix_Position = LearnablePositionalEncoding(emb_size, dropout=config['dropout'], max_len=seq_len)

        self.LayerNorm1 = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)
        if self.Rel_pos_encode == 'Scalar':
            self.attention_layer = Attention_Rel_Scl(emb_size, num_heads, seq_len, config['dropout'])
        elif self.Rel_pos_encode == 'Vector':
            self.attention_layer = Attention_Rel_Vec(emb_size, num_heads, seq_len, config['dropout'])
        else:
            self.attention_layer = Attention(emb_size, num_heads, config['dropout'])

        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(config['dropout']))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x_src = self.embed_layer(x.permute(0, 2, 1))
        if self.Fix_pos_encode != 'None':
            x_src = self.Fix_Position(x_src)
        att = x_src + self.attention_layer(x_src)
        att = self.LayerNorm1(att)
        out = att + self.FeedForward(att)
        out = self.LayerNorm2(out)

        out = out.permute(0, 2, 1)
        out = self.gap(out)
        out = self.flatten(out)
        out = self.out(out)
        # out = out.permute(1, 0, 2)
        # out = self.out(out[-1])

        return out


class ConvTran(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        # Parameters Initialization -----------------------------------------------
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        emb_size = config['emb_size']
        num_heads = config['num_heads']
        dim_ff = config['dim_ff']
        self.Fix_pos_encode = config['Fix_pos_encode']
        self.Rel_pos_encode = config['Rel_pos_encode']
        # Embedding Layer -----------------------------------------------------------
        self.embed_layer = nn.Sequential(nn.Conv2d(1, emb_size * 4, kernel_size=[1, 8], padding='same'),
                                         nn.BatchNorm2d(emb_size * 4),
                                         nn.GELU())

        self.embed_layer2 = nn.Sequential(
            nn.Conv2d(emb_size * 4, emb_size, kernel_size=[channel_size, 1], padding='valid'),
            nn.BatchNorm2d(emb_size),
            nn.GELU())

        if self.Fix_pos_encode == 'tAPE':
            self.Fix_Position = tAPE(emb_size, dropout=config['dropout'], max_len=seq_len)
        elif self.Fix_pos_encode == 'Sin':
            self.Fix_Position = AbsolutePositionalEncoding(emb_size, dropout=config['dropout'], max_len=seq_len)
        elif config['Fix_pos_encode'] == 'Learn':
            self.Fix_Position = LearnablePositionalEncoding(emb_size, dropout=config['dropout'], max_len=seq_len)

        if self.Rel_pos_encode == 'eRPE':
            self.attention_layer = Attention_Rel_Scl(emb_size, num_heads, seq_len, config['dropout'])
        elif self.Rel_pos_encode == 'Vector':
            self.attention_layer = Attention_Rel_Vec(emb_size, num_heads, seq_len, config['dropout'])
        else:
            self.attention_layer = Attention(emb_size, num_heads, config['dropout'])

        self.LayerNorm = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)

        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(config['dropout']))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x_src = self.embed_layer(x)
        x_src = self.embed_layer2(x_src).squeeze(2)
        x_src = x_src.permute(0, 2, 1)
        if self.Fix_pos_encode != 'None':
            x_src_pos = self.Fix_Position(x_src)
            att = x_src + self.attention_layer(x_src_pos)
        else:
            att = x_src + self.attention_layer(x_src)
        att = self.LayerNorm(att)
        out = att + self.FeedForward(att)
        out = self.LayerNorm2(out)
        out = out.permute(0, 2, 1)
        out = self.gap(out)
        out = self.flatten(out)
        out = self.out(out)
        return out


class CasualConvTran(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        # Parameters Initialization -----------------------------------------------
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        emb_size = config['emb_size']
        num_heads = config['num_heads']
        dim_ff = config['dim_ff']
        self.Fix_pos_encode = config['Fix_pos_encode']
        self.Rel_pos_encode = config['Rel_pos_encode']
        # Embedding Layer -----------------------------------------------------------
        self.causal_Conv1 = nn.Sequential(CausalConv1d(channel_size, emb_size, kernel_size=8, stride=2, dilation=1),
                                          nn.BatchNorm1d(emb_size), nn.GELU())

        self.causal_Conv2 = nn.Sequential(CausalConv1d(emb_size, emb_size, kernel_size=5, stride=2, dilation=2),
                                          nn.BatchNorm1d(emb_size), nn.GELU())

        self.causal_Conv3 = nn.Sequential(CausalConv1d(emb_size, emb_size, kernel_size=3, stride=2, dilation=2),
                                          nn.BatchNorm1d(emb_size), nn.GELU())

        if self.Fix_pos_encode == 'tAPE':
            self.Fix_Position = tAPE(emb_size, dropout=config['dropout'], max_len=seq_len)
        elif self.Fix_pos_encode == 'Sin':
            self.Fix_Position = tAPE(emb_size, dropout=config['dropout'], max_len=seq_len)
        elif config['Fix_pos_encode'] == 'Learn':
            self.Fix_Position = LearnablePositionalEncoding(emb_size, dropout=config['dropout'], max_len=seq_len)

        if self.Rel_pos_encode == 'eRPE':
            self.attention_layer = Attention_Rel_Scl(emb_size, num_heads, seq_len, config['dropout'])
        elif self.Rel_pos_encode == 'Vector':
            self.attention_layer = Attention_Rel_Vec(emb_size, num_heads, seq_len, config['dropout'])
        else:
            self.attention_layer = Attention(emb_size, num_heads, config['dropout'])

        self.LayerNorm = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)

        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(config['dropout']))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x_src = self.embed_layer(x)
        x_src = self.embed_layer2(x_src).squeeze(2)
        x_src = x_src.permute(0, 2, 1)
        if self.Fix_pos_encode != 'None':
            x_src_pos = self.Fix_Position(x_src)
            att = x_src + self.attention_layer(x_src_pos)
        else:
            att = x_src + self.attention_layer(x_src)
        att = self.LayerNorm(att)
        out = att + self.FeedForward(att)
        out = self.LayerNorm2(out)
        out = out.permute(0, 2, 1)
        out = self.gap(out)
        out = self.flatten(out)
        out = self.out(out)
        return out


class CausalConv1d(nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)

        self.__padding = (kernel_size - 1) * dilation

    def forward(self, x):
        return super(CausalConv1d, self).forward(nn.functional.pad(x, (self.__padding, 0)))


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


# class MLSTMfcn(nn.Module):
#     def __init__(self, config, num_classes,
#                  num_lstm_out=128, num_lstm_layers=1,
#                  conv1_nf=128, conv2_nf=256, conv3_nf=128,
#                  lstm_drop_p=0.8, fc_drop_p=0.3):
#         super(MLSTMfcn, self).__init__()
#
#         if config['permute_data']:
#             self.max_seq_len = config['Data_shape'][1]
#             self.num_features = config['Data_shape'][2]
#         else:
#             self.max_seq_len = config['Data_shape'][2]
#             self.num_features = config['Data_shape'][1]
#
#         noop = nn.Sequential()
#         self.num_classes = num_classes
#         self.shuffle = _Permute(0, 2, 1) if config['permute_data'] else noop
#
#         self.num_lstm_out = num_lstm_out
#         self.num_lstm_layers = num_lstm_layers
#
#         self.conv1_nf = conv1_nf
#         self.conv2_nf = conv2_nf
#         self.conv3_nf = conv3_nf
#
#         self.lstm_drop_p = lstm_drop_p
#         self.fc_drop_p = fc_drop_p
#
#         self.lstm = nn.LSTM(input_size=self.num_features,
#                             hidden_size=self.num_lstm_out,
#                             num_layers=self.num_lstm_layers,
#                             batch_first=True)
#
#         self.conv1 = CausalConv1d(self.num_features, self.conv1_nf, kernel_size=8, stride=1, dilation=1)
#         self.conv2 = CausalConv1d(self.conv1_nf, self.conv2_nf, kernel_size=5, stride=1, dilation=1)
#         self.conv3 = CausalConv1d(self.conv2_nf, self.conv3_nf, kernel_size=3, stride=1, dilation=1)
#
#         # self.conv1 = nn.Conv1d(self.num_features, self.conv1_nf, 8)
#         # self.conv2 = nn.Conv1d(self.conv1_nf, self.conv2_nf, 5)
#         # self.conv3 = nn.Conv1d(self.conv2_nf, self.conv3_nf, 3)
#
#         self.bn1 = nn.BatchNorm1d(self.conv1_nf, momentum=0.99, eps=0.001)
#         self.bn2 = nn.BatchNorm1d(self.conv2_nf, momentum=0.99, eps=0.001)
#         self.bn3 = nn.BatchNorm1d(self.conv3_nf, momentum=0.99, eps=0.001)
#
#         self.se1 = SELayer(self.conv1_nf)  # ex 128
#         self.se2 = SELayer(self.conv2_nf)  # ex 256
#
#         self.relu = nn.GELU()
#         self.lstmDrop = nn.Dropout(self.lstm_drop_p)
#         self.convDrop = nn.Dropout(self.fc_drop_p)
#         self.fc = nn.Linear(self.conv3_nf + self.num_lstm_out, self.num_classes)
#
#     def forward(self, x):
#         ''' input x should be in size [B,T,F], where
#             B = Batch size
#             T = Time samples
#             F = features
#         '''
#         # x = torch.permute(x, (0, 2, 1))
#         x = self.shuffle(x)
#         x1, (ht, ct) = self.lstm(x)
#         x1 = x1[:, -1, :]
#         x1 = self.lstmDrop(x1)
#         x2 = x.transpose(2, 1)
#         x2 = self.convDrop(self.relu(self.bn1(self.conv1(x2))))
#         x2 = self.se1(x2)
#         x2 = self.convDrop(self.relu(self.bn2(self.conv2(x2))))
#         x2 = self.se2(x2)
#         x2 = self.convDrop(self.relu(self.bn3(self.conv3(x2))))
#         x2 = torch.mean(x2, 2)
#
#         x_all = torch.cat((x1, x2), dim=1)
#         x_all = self.convDrop(x_all)
#         x_out = self.fc(x_all)
#         x_out = F.log_softmax(x_out, dim=1)
#
#         return x_out


def rbf_kernel(X, Y, gamma=5.5):
    """
    Compute the RBF kernel matrix between two sets of data points.

    Args:
    - X (torch.Tensor): Tensor of shape (N1, D) representing the first set of data points.
    - Y (torch.Tensor): Tensor of shape (N2, D) representing the second set of data points.
    - gamma (float): Parameter for the RBF kernel.

    Returns:
    - kernel_matrix (torch.Tensor): Computed RBF kernel matrix of shape (N1, N2).
    """
    # Compute squared Euclidean distance
    dist_matrix = torch.cdist(X, Y, p=2) ** 2

    # Compute RBF kernel matrix
    kernel_matrix = torch.exp(-gamma * dist_matrix)

    return kernel_matrix


def cluster_hidden_logit(hidden):
    seq = hidden.shape[1]
    last = hidden[:, -1, :]
    cosine_sim_matrix = F.cosine_similarity(hidden.unsqueeze(2), hidden.unsqueeze(1), dim=-1)
    # cosine_sim_matrix = rbf_kernel(hidden, hidden)
    cosine_sim_matrix = torch.exp(-5.5 * (1 - cosine_sim_matrix))
    cosine_sim_matrix = torch.stack(
        [cosine_sim_matrix[i] - torch.diag(cosine_sim_matrix[i].diagonal()) for i in range(cosine_sim_matrix.shape[0])])
    dist = torch.stack([cosine_sim_matrix[:, :i, :i].mean(dim=(1, 2)) + cosine_sim_matrix[:, i:, i:].mean(dim=(1, 2)) -
                        cosine_sim_matrix[:, :i, i:].mean(dim=(1, 2)) - cosine_sim_matrix[:, i:, :i].mean(dim=(1, 2)) \
                        for i in range(1, seq)])  # seq - 1 * batch
    cuts = torch.argmax(dist, dim=0) + 1
    corr_hid = torch.stack(
        [torch.mean(hidden[index, :cut, :], dim=0) for index, cut in enumerate(cuts)])
    ortho_loss = torch.mean(torch.abs(torch.matmul(corr_hid, last.T)))
    return last, corr_hid, ortho_loss


# class OwnMLSTMfcn(nn.Module):
#     def __init__(self, config, num_classes,
#                  num_lstm_out=128, num_lstm_layers=1,
#                  conv1_nf=128, conv2_nf=256, conv3_nf=128,
#                  lstm_drop_p=0.8, fc_drop_p=0.3):
#         super(OwnMLSTMfcn, self).__init__()
#         emb_size = config['emb_size']
#         if config['permute_data']:
#             self.max_seq_len = config['Data_shape'][1]
#             self.num_features = config['Data_shape'][2]
#         else:
#             self.max_seq_len = config['Data_shape'][2]
#             self.num_features = config['Data_shape'][1]
#
#         noop = nn.Sequential()
#         self.num_classes = num_classes
#         self.shuffle = _Permute(0, 2, 1) if config['permute_data'] else noop
#
#         self.num_lstm_out = num_lstm_out
#         self.num_lstm_layers = num_lstm_layers
#
#         self.conv1_nf = conv1_nf
#         self.conv2_nf = conv2_nf
#         self.conv3_nf = conv3_nf
#
#         self.lstm_drop_p = lstm_drop_p
#         self.fc_drop_p = fc_drop_p
#
#         self.lstm = nn.LSTM(input_size=self.num_features,
#                             hidden_size=self.num_lstm_out,
#                             num_layers=self.num_lstm_layers,
#                             batch_first=True)
#
#         self.conv1 = CausalConv1d(self.num_features, self.conv1_nf, kernel_size=8, stride=1, dilation=1)
#         self.conv2 = CausalConv1d(self.conv1_nf, self.conv2_nf, kernel_size=5, stride=1, dilation=1)
#         self.conv3 = CausalConv1d(self.conv2_nf, self.conv3_nf, kernel_size=3, stride=1, dilation=1)
#
#         # self.conv1 = nn.Conv1d(self.num_features, self.conv1_nf, 8)
#         # self.conv2 = nn.Conv1d(self.conv1_nf, self.conv2_nf, 5)
#         # self.conv3 = nn.Conv1d(self.conv2_nf, self.conv3_nf, 3)
#
#         self.bn1 = nn.BatchNorm1d(self.conv1_nf, momentum=0.99, eps=0.001)
#         self.bn2 = nn.BatchNorm1d(self.conv2_nf, momentum=0.99, eps=0.001)
#         self.bn3 = nn.BatchNorm1d(self.conv3_nf, momentum=0.99, eps=0.001)
#
#         self.se1 = SELayer(self.conv1_nf)  # ex 128
#         self.se2 = SELayer(self.conv2_nf)  # ex 256
#
#         self.relu = nn.GELU()
#         self.lstmDrop = nn.Dropout(self.lstm_drop_p)
#         self.convDrop = nn.Dropout(self.fc_drop_p)
#
#         self.fc_mu = nn.Linear(self.num_lstm_out, self.num_lstm_out)
#         self.fc_std = nn.Linear(self.num_lstm_out, self.num_lstm_out)
#
#         self.fc_mu_cat = nn.Linear(self.num_lstm_out * 2, self.num_lstm_out)
#         self.fc_std_cat = nn.Linear(self.num_lstm_out * 2, self.num_lstm_out)
#
#         self.fc = nn.Linear(self.conv3_nf + self.num_lstm_out, self.num_classes)
#
#     def reparameterise(self, mu, std):
#         """
#         mu : [batch_size,z_dim]
#         std : [batch_size,z_dim]
#         """
#         # get epsilon from standard normal
#         eps = torch.randn_like(std)
#         return mu + std * eps
#
#     def forward(self, x):
#         ''' input x should be in size [B,T,F], where
#             B = Batch size
#             T = Time samples
#             F = features
#         '''
#         # x1 = nn.utils.rnn.pack_padded_sequence(x, seq_lens,
#         #                                        batch_first=True,
#         #                                        enforce_sorted=False)
#         x = torch.permute(x, (0, 2, 1))
#         # x = self.shuffle(x)
#         x1, (ht, ct) = self.lstm(x)
#         # x1, _ = nn.utils.rnn.pad_packed_sequence(x1, batch_first=True,
#         #                                          padding_value=0.0)
#         out, corr_hid = cluster_hidden_logit(x1)
#         mu, std = self.fc_mu(corr_hid), F.softplus(self.fc_std(corr_hid) - 5, beta=1)
#         z = self.reparameterise(mu, std)
#         x_z = torch.cat((out, z), 1)
#         mu_cat, std_cat = self.fc_mu_cat(x_z), F.softplus(self.fc_std_cat(x_z) - 5, beta=1)
#         x1 = self.reparameterise(mu_cat, std_cat)
#         x1 = self.lstmDrop(x1)
#         x2 = x.transpose(2, 1)
#         x2 = self.convDrop(self.relu(self.bn1(self.conv1(x2))))
#         x2 = self.se1(x2)
#         x2 = self.convDrop(self.relu(self.bn2(self.conv2(x2))))
#         x2 = self.se2(x2)
#         x2 = self.convDrop(self.relu(self.bn3(self.conv3(x2))))
#         x2 = torch.mean(x2, 2)
#
#         x_all = torch.cat((x1, x2), dim=1)
#         x_all = self.convDrop(x_all)
#         x_out = self.fc(x_all)
#         x_out = F.log_softmax(x_out, dim=1)
#
#         return x_out, mu, std, mu_cat, std_cat
#
#
# class OMLSTMfcn(nn.Module):
#     def __init__(self, config, num_classes,
#                  num_lstm_out=128, num_lstm_layers=1,
#                  conv1_nf=128, conv2_nf=256, conv3_nf=128,
#                  lstm_drop_p=0.8, fc_drop_p=0.3):
#         super(OMLSTMfcn, self).__init__()
#         emb_size = config['emb_size']
#
#         if config['permute_data']:
#             self.max_seq_len = config['Data_shape'][1]
#             self.num_features = config['Data_shape'][2]
#         else:
#             self.max_seq_len = config['Data_shape'][2]
#             self.num_features = config['Data_shape'][1]
#         noop = nn.Sequential()
#         self.num_classes = num_classes
#         self.shuffle = _Permute(0, 2, 1) if config['permute_data'] else noop
#
#         self.embed_layer = nn.Sequential(nn.Conv2d(1, emb_size * 4, kernel_size=[1, 8], padding='same'),
#                                          nn.BatchNorm2d(emb_size * 4),
#                                          nn.GELU())
#         self.embed_layer2 = nn.Sequential(
#             nn.Conv2d(emb_size * 4, emb_size, kernel_size=[self.num_features, 1], padding='valid'),
#             nn.BatchNorm2d(emb_size),
#             nn.GELU())
#
#         self.num_lstm_out = num_lstm_out
#         self.num_lstm_layers = num_lstm_layers
#
#         self.conv1_nf = conv1_nf
#         self.conv2_nf = conv2_nf
#         self.conv3_nf = conv3_nf
#
#         self.lstm_drop_p = lstm_drop_p
#         self.fc_drop_p = fc_drop_p
#
#         self.lstm = nn.LSTM(input_size=emb_size,
#                             hidden_size=self.num_lstm_out,
#                             num_layers=self.num_lstm_layers,
#                             batch_first=True)
#
#         self.conv1 = CausalConv1d(self.max_seq_len, self.conv1_nf, kernel_size=8, stride=1, dilation=1)
#         self.conv2 = CausalConv1d(self.conv1_nf, self.conv2_nf, kernel_size=5, stride=1, dilation=1)
#         self.conv3 = CausalConv1d(self.conv2_nf, self.conv3_nf, kernel_size=3, stride=1, dilation=1)
#
#         # self.conv1 = nn.Conv1d(self.num_features, self.conv1_nf, 8)
#         # self.conv2 = nn.Conv1d(self.conv1_nf, self.conv2_nf, 5)
#         # self.conv3 = nn.Conv1d(self.conv2_nf, self.conv3_nf, 3)
#
#         self.bn1 = nn.BatchNorm1d(self.conv1_nf, momentum=0.99, eps=0.001)
#         self.bn2 = nn.BatchNorm1d(self.conv2_nf, momentum=0.99, eps=0.001)
#         self.bn3 = nn.BatchNorm1d(self.conv3_nf, momentum=0.99, eps=0.001)
#
#         self.se1 = SELayer(self.conv1_nf)  # ex 128
#         self.se2 = SELayer(self.conv2_nf)  # ex 256
#
#         self.relu = nn.GELU()
#         self.lstmDrop = nn.Dropout(self.lstm_drop_p)
#         self.convDrop = nn.Dropout(self.fc_drop_p)
#
#         self.fc_mu = nn.Linear(self.num_lstm_out, self.num_lstm_out)
#         self.fc_std = nn.Linear(self.num_lstm_out, self.num_lstm_out)
#
#         self.fc_mu_cat = nn.Linear(self.num_lstm_out * 2, self.num_lstm_out)
#         self.fc_std_cat = nn.Linear(self.num_lstm_out * 2, self.num_lstm_out)
#
#         self.fc = nn.Linear(self.conv3_nf + self.num_lstm_out, self.num_classes)
#
#     def reparameterise(self, mu, std):
#         """
#         mu : [batch_size,z_dim]
#         std : [batch_size,z_dim]
#         """
#         # get epsilon from standard normal
#         eps = torch.randn_like(std)
#         return mu + std * eps
#
#     def forward(self, x):
#         ''' input x should be in size [B,T,F], where
#             B = Batch size
#             T = Time samples
#             F = features
#         '''
#         x = self.shuffle(x)
#         x_ = x.unsqueeze(1)
#         x_src = self.embed_layer(x_)
#         x_src = self.embed_layer2(x_src).squeeze(2)
#         x_src = x_src.permute(0, 2, 1)
#         x1, (ht, ct) = self.lstm(x_src)
#         out, corr_hid = cluster_hidden_logit(x1)
#         mu, std = self.fc_mu(corr_hid), F.softplus(self.fc_std(corr_hid) - 5, beta=1)
#         z = self.reparameterise(mu, std)
#         x_z = torch.cat((out, z), 1)
#         mu_cat, std_cat = self.fc_mu_cat(x_z), F.softplus(self.fc_std_cat(x_z) - 5, beta=1)
#         x1 = self.reparameterise(mu_cat, std_cat)
#         x1 = self.lstmDrop(x1)
#         # x2 = x_src.transpose(2, 1)
#         x2 = x.transpose(2, 1)
#         x2 = self.convDrop(self.relu(self.bn1(self.conv1(x2))))
#         x2 = self.se1(x2)
#         x2 = self.convDrop(self.relu(self.bn2(self.conv2(x2))))
#         x2 = self.se2(x2)
#         x2 = self.convDrop(self.relu(self.bn3(self.conv3(x2))))
#         x2 = torch.mean(x2, 2)
#
#         x_all = torch.cat((x1, x2), dim=1)
#         x_all = self.convDrop(x_all)
#         x_out = self.fc(x_all)
#         x_out = F.log_softmax(x_out, dim=1)
#
#         return x_out, mu, std, mu_cat, std_cat


def phi(x, w1, w2, b1, b2, n_sin):
    """
    phi function that integrates sinusoidal embeddings with MLP layers.

    Args:
        x (torch.Tensor): Input tensor.
        w1 (torch.Tensor): Weight matrix for the first linear transformation.
        w2 (torch.Tensor): Weight matrix for the second linear transformation.
        b1 (torch.Tensor): Bias vector for the first linear transformation.
        b2 (torch.Tensor): Bias vector for the second linear transformation.
        n_sin (int): Number of sinusoidal functions to generate.

    Returns:
        torch.Tensor: Transformed tensor.
    """
    omega = (2 ** torch.arange(0, n_sin)).float().reshape(-1, 1).to('cuda')
    omega_x = F.linear(x, omega, bias=None)
    x = torch.cat([x, torch.sin(omega_x), torch.cos(omega_x)], dim=-1)

    x = F.linear(x, w1, bias=b1)
    x = F.silu(x)
    x = F.linear(x, w2, bias=b2)
    return x


class KANLayer(nn.Module):
    """
    A layer in a Kolmogorov–Arnold Networks (KAN).

    Attributes:
        dim_in (int): Dimensionality of the input.
        dim_out (int): Dimensionality of the output.
        fcn_hidden (int): Number of hidden units in the feature transformation.
        fcn_n_sin (torch.tensor): Number of sinusoidal functions to be used in phi.
    """

    def __init__(self, dim_in, dim_out, fcn_hidden=32, fcn_n_sin=3):
        """
        Initializes the KANLayer with specified dimensions and sinusoidal function count.

        Args:
            dim_in (int): Dimension of the input.
            dim_out (int): Dimension of the output.
            fcn_hidden (int): Number of hidden neurons in the for the learned non-linear transformation.
            fcn_n_sin (int): Number of sinusoidal embedding frequencies.
        """
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(dim_in, dim_out, fcn_hidden, 1 + fcn_n_sin * 2))
        self.W2 = nn.Parameter(torch.randn(dim_in, dim_out, 1, fcn_hidden))
        self.B1 = nn.Parameter(torch.randn(dim_in, dim_out, fcn_hidden))
        self.B2 = nn.Parameter(torch.randn(dim_in, dim_out, 1))

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.fcn_hidden = fcn_hidden
        self.fcn_n_sin = torch.tensor(fcn_n_sin).long()

        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_normal_(self.W1)
        nn.init.xavier_normal_(self.W2)
        # apply zero bias
        nn.init.zeros_(self.B1)
        nn.init.zeros_(self.B2)

    def map(self, x):
        """
        Maps input tensor x through phi function in a vectorized manner.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after mapping through phi.
        """
        F = torch.vmap(
            # take dim_in out, -> dim_in x (dim_out, *)(1)
            torch.vmap(phi, (None, 0, 0, 0, 0, None), 0),  # take dim_out out, -> dim_out x (*)
            (0, 0, 0, 0, 0, None), 0
        )
        return F(x.unsqueeze(-1), self.W1, self.W2, self.B1, self.B2, self.fcn_n_sin).squeeze(-1)

    def forward(self, x):
        """
        Forward pass of the KANLayer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Summed output after mapping each dimensions through phi.
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        batch, dim_in = x.shape
        assert dim_in == self.dim_in

        batch_f = torch.vmap(self.map, 0, 0)
        phis = batch_f(x)  # [batch, dim_in, dim_out]

        return phis.sum(dim=1)

    def take_function(self, i, j):
        """
        Returns a phi function specific to the (i, j)-th elements of parameters.

        Args:
            i (int): Row index in parameter tensors.
            j (int): Column index in parameter tensors.

        Returns:
            function: A function that computes phi for specific parameters.
        """

        def activation(x):
            return phi(x, self.W1[i, j], self.W2[i, j], self.B1[i, j], self.B2[i, j], self.fcn_n_sin)

        return activation


class OMKANfcn(nn.Module):
    def __init__(self, config, num_classes,
                 num_lstm_out=128, num_lstm_layers=1,
                 conv1_nf=128, conv2_nf=256, conv3_nf=128,
                 lstm_drop_p=0.8, fc_drop_p=0.3):
        super(OMKANfcn, self).__init__()
        emb_size = config['emb_size']

        if config['permute_data']:
            self.max_seq_len = config['Data_shape'][1]
            self.num_features = config['Data_shape'][2]
        else:
            self.max_seq_len = config['Data_shape'][2]
            self.num_features = config['Data_shape'][1]
        noop = nn.Sequential()
        self.num_classes = num_classes
        self.shuffle = _Permute(0, 2, 1) if config['permute_data'] else noop

        self.embed_layer = nn.Sequential(nn.Conv2d(1, emb_size * 4, kernel_size=[1, 8], padding='same'),
                                         nn.BatchNorm2d(emb_size * 4),
                                         nn.GELU())
        self.embed_layer2 = nn.Sequential(
            nn.Conv2d(emb_size * 4, emb_size, kernel_size=[self.num_features, 1], padding='valid'),
            nn.BatchNorm2d(emb_size),
            nn.GELU())

        self.num_lstm_out = num_lstm_out
        self.num_lstm_layers = num_lstm_layers

        self.conv1_nf = conv1_nf
        self.conv2_nf = conv2_nf
        self.conv3_nf = conv3_nf

        self.lstm_drop_p = lstm_drop_p
        self.fc_drop_p = fc_drop_p

        self.lstm = nn.LSTM(input_size=emb_size,
                            hidden_size=self.num_lstm_out,
                            num_layers=self.num_lstm_layers,
                            batch_first=True)

        self.conv1 = CausalConv1d(self.max_seq_len, self.conv1_nf, kernel_size=8, stride=1, dilation=1)
        self.conv2 = CausalConv1d(self.conv1_nf, self.conv2_nf, kernel_size=5, stride=1, dilation=1)
        self.conv3 = CausalConv1d(self.conv2_nf, self.conv3_nf, kernel_size=3, stride=1, dilation=1)

        # self.conv1 = nn.Conv1d(self.num_features, self.conv1_nf, 8)
        # self.conv2 = nn.Conv1d(self.conv1_nf, self.conv2_nf, 5)
        # self.conv3 = nn.Conv1d(self.conv2_nf, self.conv3_nf, 3)

        self.bn1 = nn.BatchNorm1d(self.conv1_nf, momentum=0.99, eps=0.001)
        self.bn2 = nn.BatchNorm1d(self.conv2_nf, momentum=0.99, eps=0.001)
        self.bn3 = nn.BatchNorm1d(self.conv3_nf, momentum=0.99, eps=0.001)

        self.se1 = SELayer(self.conv1_nf)  # ex 128
        self.se2 = SELayer(self.conv2_nf)  # ex 256

        self.relu = nn.GELU()
        self.lstmDrop = nn.Dropout(self.lstm_drop_p)
        self.convDrop = nn.Dropout(self.fc_drop_p)

        self.fc_mu = KANLayer(self.num_lstm_out, self.num_lstm_out)
        self.fc_std = KANLayer(self.num_lstm_out, self.num_lstm_out)

        self.fc_mu_cat = KANLayer(self.num_lstm_out * 2, self.num_lstm_out)
        self.fc_std_cat = KANLayer(self.num_lstm_out * 2, self.num_lstm_out)

        self.fc = KANLayer(self.conv3_nf + self.num_lstm_out, self.num_classes)

    def reparameterise(self, mu, std):
        """
        mu : [batch_size,z_dim]
        std : [batch_size,z_dim]
        """
        # get epsilon from standard normal
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        ''' input x should be in size [B,T,F], where
            B = Batch size
            T = Time samples
            F = features
        '''
        x = self.shuffle(x)
        x_ = x.unsqueeze(1)
        x_src = self.embed_layer(x_)
        x_src = self.embed_layer2(x_src).squeeze(2)
        x_src = x_src.permute(0, 2, 1)
        x1, (ht, ct) = self.lstm(x_src)
        out, corr_hid = cluster_hidden_logit(x1)
        mu, std = self.fc_mu(corr_hid), F.softplus(self.fc_std(corr_hid) - 5, beta=1)
        z = self.reparameterise(mu, std)
        x_z = torch.cat((out, z), 1)
        mu_cat, std_cat = self.fc_mu_cat(x_z), F.softplus(self.fc_std_cat(x_z) - 5, beta=1)
        x1 = self.reparameterise(mu_cat, std_cat)
        x1 = self.lstmDrop(x1)
        # x2 = x_src.transpose(2, 1)
        x2 = x.transpose(2, 1)
        x2 = self.convDrop(self.relu(self.bn1(self.conv1(x2))))
        x2 = self.se1(x2)
        x2 = self.convDrop(self.relu(self.bn2(self.conv2(x2))))
        x2 = self.se2(x2)
        x2 = self.convDrop(self.relu(self.bn3(self.conv3(x2))))
        x2 = torch.mean(x2, 2)

        x_all = torch.cat((x1, x2), dim=1)
        x_all = self.convDrop(x_all)
        x_out = self.fc(x_all)
        x_out = F.log_softmax(x_out, dim=1)

        return x_out, mu, std, mu_cat, std_cat


class KLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(KLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 权重矩阵
        self.Wf = KANLayer(input_size + hidden_size, hidden_size)  # 遗忘门
        self.Wi = KANLayer(input_size + hidden_size, hidden_size)  # 输入门
        self.Wc = KANLayer(input_size + hidden_size, hidden_size)  # 候选细胞状态
        self.Wo = KANLayer(input_size + hidden_size, hidden_size)  # 输出门

    def forward(self, x, hidden):
        h, c = hidden
        combined = torch.cat((x, h), dim=1)

        ft = torch.sigmoid(self.Wf(combined))  # 遗忘门
        it = torch.sigmoid(self.Wi(combined))  # 输入门
        ct_tilde = torch.tanh(self.Wc(combined))  # 候选细胞状态
        c = ft * c + it * ct_tilde  # 更新细胞状态
        ot = torch.sigmoid(self.Wo(combined))  # 输出门
        h = ot * torch.tanh(c)  # 更新隐藏状态

        return h, c


class KLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True):
        super(KLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # 创建多个LSTM单元
        self.cells = nn.ModuleList(
            [KLSTMCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)]
        )

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)

        # 初始化隐藏状态和细胞状态
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)
        c = torch.zeros(batch_size, self.hidden_size).to(x.device)

        outputs = []
        for t in range(seq_len):
            input_t = x[:, t, :]  # 当前时间步的输入

            for layer in range(self.num_layers):
                h, c = self.cells[layer](input_t if layer == 0 else h, (h, c))
                input_t = h  # 当前层的输出作为下一层的输入

            outputs.append(h.unsqueeze(1))  # 保存当前时间步的输出

        outputs = torch.cat(outputs, dim=1)  # 拼接所有时间步的输出
        return outputs  # 输出形状: (batch_size, seq_length, hidden_size)


class OMLKANfcn(nn.Module):
    def __init__(self, config, num_classes,
                 num_lstm_out=128, num_lstm_layers=1,
                 conv1_nf=128, conv2_nf=256, conv3_nf=128,
                 lstm_drop_p=0.8, fc_drop_p=0.3):
        super(OMLKANfcn, self).__init__()
        emb_size = config['emb_size']

        if config['permute_data']:
            self.max_seq_len = config['Data_shape'][1]
            self.num_features = config['Data_shape'][2]
        else:
            self.max_seq_len = config['Data_shape'][2]
            self.num_features = config['Data_shape'][1]
        noop = nn.Sequential()
        self.num_classes = num_classes
        self.shuffle = _Permute(0, 2, 1) if config['permute_data'] else noop

        self.embed_layer = nn.Sequential(nn.Conv2d(1, emb_size * 4, kernel_size=[1, 8], padding='same'),
                                         nn.BatchNorm2d(emb_size * 4),
                                         nn.GELU())
        self.embed_layer2 = nn.Sequential(
            nn.Conv2d(emb_size * 4, emb_size, kernel_size=[self.num_features, 1], padding='valid'),
            nn.BatchNorm2d(emb_size),
            nn.GELU())

        self.num_lstm_out = num_lstm_out
        self.num_lstm_layers = num_lstm_layers

        self.conv1_nf = conv1_nf
        self.conv2_nf = conv2_nf
        self.conv3_nf = conv3_nf

        self.lstm_drop_p = lstm_drop_p
        self.fc_drop_p = fc_drop_p

        self.lstm = KLSTM(input_size=emb_size,
                          hidden_size=self.num_lstm_out,
                          num_layers=self.num_lstm_layers,
                          batch_first=True)

        self.conv1 = CausalConv1d(self.max_seq_len, self.conv1_nf, kernel_size=8, stride=1, dilation=1)
        self.conv2 = CausalConv1d(self.conv1_nf, self.conv2_nf, kernel_size=5, stride=1, dilation=1)
        self.conv3 = CausalConv1d(self.conv2_nf, self.conv3_nf, kernel_size=3, stride=1, dilation=1)

        # self.conv1 = nn.Conv1d(self.num_features, self.conv1_nf, 8)
        # self.conv2 = nn.Conv1d(self.conv1_nf, self.conv2_nf, 5)
        # self.conv3 = nn.Conv1d(self.conv2_nf, self.conv3_nf, 3)

        self.bn1 = nn.BatchNorm1d(self.conv1_nf, momentum=0.99, eps=0.001)
        self.bn2 = nn.BatchNorm1d(self.conv2_nf, momentum=0.99, eps=0.001)
        self.bn3 = nn.BatchNorm1d(self.conv3_nf, momentum=0.99, eps=0.001)

        self.se1 = SELayer(self.conv1_nf)  # ex 128
        self.se2 = SELayer(self.conv2_nf)  # ex 256

        self.relu = nn.GELU()
        self.lstmDrop = nn.Dropout(self.lstm_drop_p)
        self.convDrop = nn.Dropout(self.fc_drop_p)

        self.fc_mu = nn.Linear(self.num_lstm_out, self.num_lstm_out)
        self.fc_std = nn.Linear(self.num_lstm_out, self.num_lstm_out)

        self.fc_mu_cat = nn.Linear(self.num_lstm_out * 2, self.num_lstm_out)
        self.fc_std_cat = nn.Linear(self.num_lstm_out * 2, self.num_lstm_out)

        self.fc = nn.Linear(self.conv3_nf + self.num_lstm_out, self.num_classes)

    def reparameterise(self, mu, std):
        """
        mu : [batch_size,z_dim]
        std : [batch_size,z_dim]
        """
        # get epsilon from standard normal
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        ''' input x should be in size [B,T,F], where
            B = Batch size
            T = Time samples
            F = features
        '''
        x = self.shuffle(x)
        x_ = x.unsqueeze(1)
        x_src = self.embed_layer(x_)
        x_src = self.embed_layer2(x_src).squeeze(2)
        x_src = x_src.permute(0, 2, 1)
        x1 = self.lstm(x_src)
        out, corr_hid = cluster_hidden_logit(x1)
        mu, std = self.fc_mu(corr_hid), F.softplus(self.fc_std(corr_hid) - 5, beta=1)
        z = self.reparameterise(mu, std)
        x_z = torch.cat((out, z), 1)
        mu_cat, std_cat = self.fc_mu_cat(x_z), F.softplus(self.fc_std_cat(x_z) - 5, beta=1)
        x1 = self.reparameterise(mu_cat, std_cat)
        x1 = self.lstmDrop(x1)
        # x2 = x_src.transpose(2, 1)
        x2 = x.transpose(2, 1)
        x2 = self.convDrop(self.relu(self.bn1(self.conv1(x2))))
        x2 = self.se1(x2)
        x2 = self.convDrop(self.relu(self.bn2(self.conv2(x2))))
        x2 = self.se2(x2)
        x2 = self.convDrop(self.relu(self.bn3(self.conv3(x2))))
        x2 = torch.mean(x2, 2)

        x_all = torch.cat((x1, x2), dim=1)
        x_all = self.convDrop(x_all)
        x_out = self.fc(x_all)
        x_out = F.log_softmax(x_out, dim=1)

        return x_out, mu, std, mu_cat, std_cat


class EnhancedLSTMfcn(nn.Module):
    def __init__(self, config, num_classes,
                 num_lstm_out=128, num_lstm_layers=1,
                 conv1_nf=128, conv2_nf=256, conv3_nf=128,
                 lstm_drop_p=0.5, fc_drop_p=0.3):
        super(EnhancedLSTMfcn, self).__init__()
        self.conv1_nf = conv1_nf
        self.conv2_nf = conv2_nf
        self.conv3_nf = conv3_nf
        emb_size = config['emb_size']

        # 数据维度设置
        if config['permute_data']:
            self.max_seq_len = config['Data_shape'][1]
            self.num_features = config['Data_shape'][2]
        else:
            self.max_seq_len = config['Data_shape'][2]
            self.num_features = config['Data_shape'][1]
        noop = nn.Sequential()
        self.num_classes = num_classes
        self.shuffle = _Permute(0, 2, 1) if config['permute_data'] else noop

        # 嵌入层
        self.embed_layer = nn.Sequential(
            nn.Conv2d(1, emb_size * 4, kernel_size=[1, 8], padding='same'),
            nn.BatchNorm2d(emb_size * 4),
            nn.GELU()
        )
        self.embed_layer2 = nn.Sequential(
            nn.Conv2d(emb_size * 4, emb_size, kernel_size=[self.num_features, 1], padding='valid'),
            nn.BatchNorm2d(emb_size),
            nn.GELU()
        )

        # LSTM 层
        self.num_lstm_out = num_lstm_out
        self.num_lstm_layers = num_lstm_layers
        self.lstm = nn.LSTM(input_size=emb_size,
                            hidden_size=self.num_lstm_out,
                            num_layers=self.num_lstm_layers,
                            batch_first=True)

        # 因果卷积层（加入残差连接和多尺度卷积）
        self.conv1 = CausalConv1d(self.max_seq_len, self.conv1_nf, kernel_size=8, stride=1, dilation=1)
        self.conv1_res = CausalConv1d(self.max_seq_len, self.conv1_nf, kernel_size=1, stride=1, dilation=1)
        self.conv2 = CausalConv1d(self.conv1_nf, self.conv2_nf, kernel_size=5, stride=1, dilation=1)
        self.conv2_res = CausalConv1d(self.conv1_nf, self.conv2_nf, kernel_size=1, stride=1, dilation=1)
        self.conv3 = CausalConv1d(self.conv2_nf, self.conv3_nf, kernel_size=3, stride=1, dilation=1)
        self.conv3_multi = CausalConv1d(self.conv2_nf, self.conv3_nf, kernel_size=5, stride=1, dilation=1)

        # 批归一化与 SE 层
        self.bn1 = nn.BatchNorm1d(self.conv1_nf, momentum=0.1, eps=0.001)
        self.bn2 = nn.BatchNorm1d(self.conv2_nf, momentum=0.1, eps=0.001)
        self.bn3 = nn.BatchNorm1d(self.conv3_nf, momentum=0.1, eps=0.001)
        self.se1 = SELayer(self.conv1_nf)
        self.se2 = SELayer(self.conv2_nf)

        # Dropout 和激活
        self.lstm_drop_p = lstm_drop_p
        self.fc_drop_p = fc_drop_p
        self.lstmDrop = nn.Dropout(self.lstm_drop_p)
        self.convDrop = nn.Dropout(self.fc_drop_p)
        self.relu = nn.GELU()

        # 注意力机制（动态特征融合）
        self.attention = nn.MultiheadAttention(embed_dim=self.num_lstm_out, num_heads=8)

        # 门控机制
        self.gate = nn.Linear(self.num_lstm_out + self.conv3_nf, self.num_lstm_out + self.conv3_nf)

        # 全连接层
        self.fc_mu = nn.Linear(self.num_lstm_out, self.num_lstm_out)
        self.fc_std = nn.Linear(self.num_lstm_out, self.num_lstm_out)
        self.fc_mu_cat = nn.Linear(self.num_lstm_out * 2, self.num_lstm_out)
        self.fc_std_cat = nn.Linear(self.num_lstm_out * 2, self.num_lstm_out)
        self.fc = nn.Linear(2 * self.num_lstm_out, self.num_classes)

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(2 * self.num_lstm_out)

    def reparameterise(self, mu, std):
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        # 输入处理 [B, T, F]
        x = self.shuffle(x)
        x_ = x.unsqueeze(1)  # [B, 1, T, F]

        # 嵌入
        x_src = self.embed_layer(x_)
        x_src = self.embed_layer2(x_src).squeeze(2)  # [B, emb_size, T]
        x_src = x_src.permute(0, 2, 1)  # [B, T, emb_size]

        # LSTM 路径
        x1, (ht, ct) = self.lstm(x_src)
        out, corr_hid = cluster_hidden_logit(x1)  # 假设此函数已定义
        mu, std = self.fc_mu(corr_hid), F.softplus(self.fc_std(corr_hid))  # 使用 softplus 确保 std > 0
        corr_hid_ = self.reparameterise(mu, std)

        # 连接 LSTM 输出
        x1 = torch.cat((corr_hid_, out), 1)
        mu_out, std_out = self.fc_mu_cat(x1), F.softplus(self.fc_std_cat(x1))  # 同样对 std_out 应用 softplus
        x1 = self.reparameterise(mu_out, std_out)
        x1 = self.lstmDrop(x1)

        # 卷积路径（加入残差连接和多尺度卷积）
        x2 = x.transpose(2, 1)  # [B, F, T]
        x2_res = self.conv1_res(x2)
        x2 = self.convDrop(self.relu(self.bn1(self.conv1(x2) + x2_res)))
        x2 = self.se1(x2)
        x2_res = self.conv2_res(x2)
        x2 = self.convDrop(self.relu(self.bn2(self.conv2(x2) + x2_res)))
        x2 = self.se2(x2)
        x2_multi = self.conv3_multi(x2)
        x2 = self.convDrop(self.relu(self.bn3(self.conv3(x2) + x2_multi)))
        x2 = torch.mean(x2, 2)  # [B, conv3_nf]

        # 动态特征融合（注意力机制）
        x1_attn, _ = self.attention(x1.unsqueeze(0), x1.unsqueeze(0), x1.unsqueeze(0))
        x1_attn = x1_attn.squeeze(0)

        # 门控机制
        gate_input = torch.cat((x1_attn, x2), dim=1)
        gate_weight = torch.sigmoid(self.gate(gate_input))
        x_all = gate_weight * torch.cat((x1_attn, x2), dim=1)

        # Layer Normalization
        x_all = self.layer_norm(x_all)

        # 分类
        x_all = self.convDrop(x_all)
        x_out = self.fc(x_all)
        x_out = F.log_softmax(x_out, dim=1)

        return x_out, mu, std, mu_out, std_out


class EnhancedMLSTMfcn(nn.Module):
    def __init__(self, config, num_classes,
                 num_lstm_out=128, num_lstm_layers=1,
                 conv1_nf=128, conv2_nf=256, conv3_nf=128,
                 lstm_drop_p=0.5, fc_drop_p=0.3):
        super(EnhancedMLSTMfcn, self).__init__()
        self.conv1_nf = conv1_nf
        self.conv2_nf = conv2_nf
        self.conv3_nf = conv3_nf
        emb_size = config['emb_size']

        # 数据维度设置
        if config['permute_data']:
            self.max_seq_len = config['Data_shape'][1]
            self.num_features = config['Data_shape'][2]
        else:
            self.max_seq_len = config['Data_shape'][2]
            self.num_features = config['Data_shape'][1]
        noop = nn.Sequential()
        self.num_classes = num_classes
        self.shuffle = _Permute(0, 2, 1) if config['permute_data'] else noop

        # 嵌入层
        self.embed_layer = nn.Sequential(
            nn.Conv2d(1, emb_size * 4, kernel_size=[1, 8], padding='same'),
            nn.BatchNorm2d(emb_size * 4),
            nn.GELU()
        )
        self.embed_layer2 = nn.Sequential(
            nn.Conv2d(emb_size * 4, emb_size, kernel_size=[self.num_features, 1], padding='valid'),
            nn.BatchNorm2d(emb_size),
            nn.GELU()
        )

        # LSTM 层
        self.num_lstm_out = num_lstm_out
        self.num_lstm_layers = num_lstm_layers
        self.lstm = nn.LSTM(input_size=emb_size,
                            hidden_size=self.num_lstm_out,
                            num_layers=self.num_lstm_layers,
                            batch_first=True)

        # 因果卷积层（加入残差连接和多尺度卷积）
        self.conv1 = CausalConv1d(self.max_seq_len, self.conv1_nf, kernel_size=8, stride=1, dilation=1)
        self.conv1_res = CausalConv1d(self.max_seq_len, self.conv1_nf, kernel_size=1, stride=1, dilation=1)
        self.conv2 = CausalConv1d(self.conv1_nf, self.conv2_nf, kernel_size=5, stride=1, dilation=1)
        self.conv2_res = CausalConv1d(self.conv1_nf, self.conv2_nf, kernel_size=1, stride=1, dilation=1)
        self.conv3 = CausalConv1d(self.conv2_nf, self.conv3_nf, kernel_size=3, stride=1, dilation=1)
        self.conv3_multi = CausalConv1d(self.conv2_nf, self.conv3_nf, kernel_size=5, stride=1, dilation=1)

        # 批归一化与 SE 层
        self.bn1 = nn.BatchNorm1d(self.conv1_nf, momentum=0.1, eps=0.001)
        self.bn2 = nn.BatchNorm1d(self.conv2_nf, momentum=0.1, eps=0.001)
        self.bn3 = nn.BatchNorm1d(self.conv3_nf, momentum=0.1, eps=0.001)
        self.se1 = SELayer(self.conv1_nf)
        self.se2 = SELayer(self.conv2_nf)

        # Dropout 和激活
        self.lstm_drop_p = lstm_drop_p
        self.fc_drop_p = fc_drop_p
        self.lstmDrop = nn.Dropout(self.lstm_drop_p)
        self.convDrop = nn.Dropout(self.fc_drop_p)
        self.relu = nn.GELU()

        # 注意力机制（动态特征融合）
        self.attention = nn.MultiheadAttention(embed_dim=self.num_lstm_out, num_heads=8)

        # 门控机制
        self.gate = nn.Linear(self.num_lstm_out + self.conv3_nf, self.num_lstm_out + self.conv3_nf)

        # 全连接层
        self.fc_mu = nn.Linear(self.num_lstm_out, self.num_lstm_out)
        self.fc_std = nn.Linear(self.num_lstm_out, self.num_lstm_out)
        self.fc_mu_cat = nn.Linear(self.num_lstm_out * 2, self.num_lstm_out)
        self.fc_std_cat = nn.Linear(self.num_lstm_out * 2, self.num_lstm_out)
        self.fc = nn.Linear(2 * self.num_lstm_out, self.num_classes)

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(2 * self.num_lstm_out)

    def reparameterise(self, mu, std):
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        # 输入处理 [B, T, F]
        x = self.shuffle(x)
        x_ = x.unsqueeze(1)  # [B, 1, T, F]

        # 嵌入
        x_src = self.embed_layer(x_)
        x_src = self.embed_layer2(x_src).squeeze(2)  # [B, emb_size, T]
        x_src = x_src.permute(0, 2, 1)  # [B, T, emb_size]

        # LSTM 路径
        x1, (ht, ct) = self.lstm(x_src)
        out, corr_hid,ortho_loss = cluster_hidden_logit(x1)  # 假设此函数已定义
        mu, std = self.fc_mu(corr_hid), F.softplus(self.fc_std(corr_hid))  # 使用 softplus 确保 std > 0
        corr_hid_ = self.reparameterise(mu, std)

        # 连接 LSTM 输出
        x1 = torch.cat((corr_hid_, out), 1)
        mu_out, std_out = self.fc_mu_cat(x1), F.softplus(self.fc_std_cat(x1))  # 同样对 std_out 应用 softplus
        x1 = self.reparameterise(mu_out, std_out)
        x1 = self.lstmDrop(x1)

        # 卷积路径（加入残差连接和多尺度卷积）
        x2 = x.transpose(2, 1)  # [B, F, T]
        x2_res = self.conv1_res(x2)
        x2 = self.convDrop(self.relu(self.bn1(self.conv1(x2) + x2_res)))
        x2 = self.se1(x2)
        x2_res = self.conv2_res(x2)
        x2 = self.convDrop(self.relu(self.bn2(self.conv2(x2) + x2_res)))
        x2 = self.se2(x2)
        x2_multi = self.conv3_multi(x2)
        x2 = self.convDrop(self.relu(self.bn3(self.conv3(x2) + x2_multi)))
        x2 = torch.mean(x2, 2)  # [B, conv3_nf]

        # 动态特征融合（注意力机制）
        x1_attn, _ = self.attention(x1.unsqueeze(0), x1.unsqueeze(0), x1.unsqueeze(0))
        x1_attn = x1_attn.squeeze(0)

        # 门控机制
        gate_input = torch.cat((x1_attn, x2), dim=1)
        gate_weight = torch.sigmoid(self.gate(gate_input))
        x_all = gate_weight * torch.cat((x1_attn, x2), dim=1)

        # Layer Normalization
        x_all = self.layer_norm(x_all)

        # 分类
        x_all = self.convDrop(x_all)
        x_out = self.fc(x_all)
        x_out = F.log_softmax(x_out, dim=1)

        return x_out, mu, std, mu_out, std_out,ortho_loss


class EnhancedLSTMfcnKan(nn.Module):
    def __init__(self, config, num_classes,
                 num_lstm_out=128, num_lstm_layers=1,
                 conv1_nf=128, conv2_nf=256, conv3_nf=128,
                 lstm_drop_p=0.5, fc_drop_p=0.3):
        super(EnhancedLSTMfcnKan, self).__init__()
        self.conv1_nf = conv1_nf
        self.conv2_nf = conv2_nf
        self.conv3_nf = conv3_nf

        # Data dimension settings (dynamic adaptation)
        self.config = config
        self.num_classes = num_classes
        self.permute_data = config.get('permute_data', False)
        self.shuffle = _Permute(0, 2, 1) if self.permute_data else nn.Sequential()

        # Placeholders, dimensions set dynamically in forward
        self.num_features = None
        self.max_seq_len = None

        # KAN-LSTM layer (replacing original LSTM)
        self.num_lstm_out = num_lstm_out
        self.num_lstm_layers = num_lstm_layers  # Kept for compatibility, but only 1 layer is used in KAN-LSTM here
        self.kan_lstm_cell = None  # Delayed initialization, awaiting input dims

        # Causal convolution layers (placeholder for conv1)
        self.conv1 = None
        self.conv2 = CausalConv1d(self.conv1_nf, self.conv2_nf, kernel_size=5, stride=1, dilation=1)
        self.conv3 = CausalConv1d(self.conv2_nf, self.conv3_nf, kernel_size=3, stride=1, dilation=1)

        # Batch normalization and SE layers
        self.bn1 = nn.BatchNorm1d(self.conv1_nf, momentum=0.1, eps=0.001)
        self.bn2 = nn.BatchNorm1d(self.conv2_nf, momentum=0.1, eps=0.001)
        self.bn3 = nn.BatchNorm1d(self.conv3_nf, momentum=0.1, eps=0.001)
        self.se1 = SELayer(self.conv1_nf)
        self.se2 = SELayer(self.conv2_nf)

        # Dropout and activation
        self.lstm_drop_p = lstm_drop_p
        self.fc_drop_p = fc_drop_p
        self.lstmDrop = nn.Dropout(self.lstm_drop_p)
        self.convDrop = nn.Dropout(self.fc_drop_p)
        self.relu = nn.GELU()

        # Variational layers (for DVIB), replaced with KAN
        self.fc_mu = None
        self.fc_std = None
        self.fc_mu_cat = None
        self.fc_std_cat = None

        # Final fully connected layer, replaced with KAN
        self.fc = None  # Delayed initialization, awaiting input dims

    def reparameterise(self, mu, std):
        eps = torch.randn_like(std)
        return mu + std * eps

    def initialize_layers(self, num_features):
        """Dynamically initialize KAN-LSTM and conv layers based on input dims"""
        device = next(self.parameters()).device
        self.num_features = num_features
        if self.kan_lstm_cell is None:
            self.kan_lstm_cell = KANLSTMCell(input_dim=self.num_features,
                                             hidden_dim=self.num_lstm_out,
                                             kan_hidden_dim=16).to(device)  # kan_hidden_dim adjustable
        if self.conv1 is None:
            self.conv1 = CausalConv1d(self.num_features, self.conv1_nf, kernel_size=8, stride=1, dilation=1).to(device)
        if self.fc_mu is None:
            self.fc_mu = nn.Linear(self.num_lstm_out, self.num_lstm_out).to(device)
        if self.fc_std is None:
            self.fc_std = nn.Linear(self.num_lstm_out, self.num_lstm_out).to(device)
        if self.fc_mu_cat is None:
            self.fc_mu_cat = nn.Linear(self.num_lstm_out * 2, self.num_lstm_out).to(device)
        if self.fc_std_cat is None:
            self.fc_std_cat = nn.Linear(self.num_lstm_out * 2, self.num_lstm_out).to(device)
        if self.fc is None:
            self.fc = nn.Linear(self.num_lstm_out + self.conv3_nf, self.num_classes).to(device)

    def forward(self, x):
        device = next(self.parameters()).device
        # Input processing [B, T, F]
        batch_size, seq_len, features = x.shape  # e.g., [16, 4, 65]
        if self.num_features is None or self.num_features != features:
            self.initialize_layers(features)  # Dynamic initialization
        self.max_seq_len = seq_len

        # Adjust input based on permute_data
        x = self.shuffle(x)  # [16, 4, 65] or [16, 65, 4] (if permute)

        # KAN-LSTM path (replacing original LSTM)
        x_lstm = x.transpose(0, 1)  # [seq_len, batch_size, num_features]
        h0 = torch.zeros(batch_size, self.num_lstm_out).to(x.device)
        c0 = torch.zeros(batch_size, self.num_lstm_out).to(x.device)
        x1, h_n, c_n = kan_lstm_forward(x_lstm, self.kan_lstm_cell, h0, c0)  # [seq_len, batch_size, num_lstm_out]
        x1 = x1.transpose(0, 1)  # [batch_size, seq_len, num_lstm_out]

        out, corr_hid = cluster_hidden_logit(x1)  # Assuming this function is defined
        mu = self.fc_mu(corr_hid.to(device))
        std = F.softplus(self.fc_std(corr_hid))
        z = self.reparameterise(mu, std)
        x_z = torch.cat((out, z), 1)
        mu_cat = self.fc_mu_cat(x_z)
        std_cat = F.softplus(self.fc_std_cat(x_z))
        x1 = self.reparameterise(mu_cat, std_cat)
        x1 = self.lstmDrop(x1)

        # Convolution path
        x2 = x.transpose(1, 2)  # [16, 4, 65] -> [16, 65, 4]
        x2 = self.convDrop(self.relu(self.bn1(self.conv1(x2))))
        x2 = self.se1(x2)
        x2 = self.convDrop(self.relu(self.bn2(self.conv2(x2))))
        x2 = self.se2(x2)
        x2 = self.convDrop(self.relu(self.bn3(self.conv3(x2))))
        x2 = torch.mean(x2, 2)  # [16, conv3_nf]

        # Feature fusion
        x_all = torch.cat((x1, x2), dim=1)
        x_all = self.convDrop(x_all)
        x_out = self.fc(x_all)
        x_out = F.log_softmax(x_out, dim=1)

        return x_out, mu, std, mu_cat, std_cat

# class NewOMLSTMfcn(nn.Module):
#     def __init__(self, config, num_classes,
#                  num_lstm_out=128, num_lstm_layers=1,
#                  conv1_nf=128, conv2_nf=256, conv3_nf=128,
#                  lstm_drop_p=0.8, fc_drop_p=0.3):
#         super(NewOMLSTMfcn, self).__init__()
#         emb_size = config['emb_size']
#
#         if config['permute_data']:
#             self.max_seq_len = config['Data_shape'][1]
#             self.num_features = config['Data_shape'][2]
#         else:
#             self.max_seq_len = config['Data_shape'][2]
#             self.num_features = config['Data_shape'][1]
#         noop = nn.Sequential()
#         self.num_classes = num_classes
#         self.shuffle = _Permute(0, 2, 1) if config['permute_data'] else noop
#
#         self.embed_layer = nn.Sequential(nn.Conv2d(1, emb_size * 4, kernel_size=[1, 8], padding='same'),
#                                          nn.BatchNorm2d(emb_size * 4),
#                                          nn.GELU())
#         self.embed_layer2 = nn.Sequential(
#             nn.Conv2d(emb_size * 4, emb_size, kernel_size=[self.num_features, 1], padding='valid'),
#             nn.BatchNorm2d(emb_size),
#             nn.GELU())
#
#         self.num_lstm_out = num_lstm_out
#         self.num_lstm_layers = num_lstm_layers
#
#         self.conv1_nf = conv1_nf
#         self.conv2_nf = conv2_nf
#         self.conv3_nf = conv3_nf
#
#         self.lstm_drop_p = lstm_drop_p
#         self.fc_drop_p = fc_drop_p
#
#         self.lstm = nn.LSTM(input_size=emb_size,
#                             hidden_size=self.num_lstm_out,
#                             num_layers=self.num_lstm_layers,
#                             batch_first=True)
#
#         self.conv1 = CausalConv1d(self.max_seq_len, self.conv1_nf, kernel_size=8, stride=1, dilation=1)
#         self.conv2 = CausalConv1d(self.conv1_nf, self.conv2_nf, kernel_size=5, stride=1, dilation=1)
#         self.conv3 = CausalConv1d(self.conv2_nf, self.conv3_nf, kernel_size=3, stride=1, dilation=1)
#
#         # self.conv1 = nn.Conv1d(self.num_features, self.conv1_nf, 8)
#         # self.conv2 = nn.Conv1d(self.conv1_nf, self.conv2_nf, 5)
#         # self.conv3 = nn.Conv1d(self.conv2_nf, self.conv3_nf, 3)
#
#         self.bn1 = nn.BatchNorm1d(self.conv1_nf, momentum=0.99, eps=0.001)
#         self.bn2 = nn.BatchNorm1d(self.conv2_nf, momentum=0.99, eps=0.001)
#         self.bn3 = nn.BatchNorm1d(self.conv3_nf, momentum=0.99, eps=0.001)
#
#         self.se1 = SELayer(self.conv1_nf)  # ex 128
#         self.se2 = SELayer(self.conv2_nf)  # ex 256
#
#         self.relu = nn.GELU()
#         self.lstmDrop = nn.Dropout(self.lstm_drop_p)
#         self.convDrop = nn.Dropout(self.fc_drop_p)
#
#         self.fc_mu = nn.Linear(self.num_lstm_out, self.num_lstm_out)
#         self.fc_std = nn.Linear(self.num_lstm_out, self.num_lstm_out)
#
#         self.fc_mu_cat = nn.Linear(self.num_lstm_out * 2, self.num_lstm_out)
#         self.fc_std_cat = nn.Linear(self.num_lstm_out * 2, self.num_lstm_out)
#
#         self.fc_mu_2 = nn.Linear(self.num_lstm_out, self.num_lstm_out)
#         self.fc_std_2 = nn.Linear(self.num_lstm_out, self.num_lstm_out)
#
#         self.fc = nn.Linear(2 * self.num_lstm_out, self.num_classes)
#
#     def reparameterise(self, mu, std):
#         """
#         mu : [batch_size,z_dim]
#         std : [batch_size,z_dim]
#         """
#         # get epsilon from standard normal
#         eps = torch.randn_like(std)
#         return mu + std * eps
#
#     def forward(self, x):
#         ''' input x should be in size [B,T,F], where
#             B = Batch size
#             T = Time samples
#             F = features
#         '''
#         x = self.shuffle(x)
#         x_ = x.unsqueeze(1)
#         x_src = self.embed_layer(x_)
#         x_src = self.embed_layer2(x_src).squeeze(2)
#         x_src = x_src.permute(0, 2, 1)
#         x1, (ht, ct) = self.lstm(x_src)
#         out, corr_hid = cluster_hidden_logit(x1)
#         # 遗忘部分
#         mu, std = self.fc_mu(corr_hid), F.softplus(self.fc_std(corr_hid) - 5, beta=1)
#         corr_hid_ = self.reparameterise(mu, std)
#         # 记忆部分
#         # mu_out, std_out = self.fc_mu_out(out), F.softplus(self.fc_std_out(out) - 5, beta=1)
#         # out_ = self.reparameterise(mu_out, std_out)
#
#         # f_ = torch.mean(corr_hid_, dim=0)
#         # r_ = torch.mean(out_, dim=0)
#         # n = out_.shape[0]  # 样本数
#         # Sigma = torch.matmul((corr_hid_ - f_).T, (out_ - r_)) / (n - 1)
#         # l0 = torch.logsumexp(torch.flatten(torch.abs(Sigma)), dim=0, keepdim=False)
#         x1 = torch.cat((corr_hid_, out), 1)
#         mu_out, std_out = self.fc_mu_cat(x1), F.softplus(self.fc_std_cat(x1) - 5, beta=1)
#         x1 = self.reparameterise(mu_out, std_out)
#
#         x1 = self.lstmDrop(x1)
#         x2 = x.transpose(2, 1)
#         x2 = self.convDrop(self.relu(self.bn1(self.conv1(x2))))
#         x2 = self.se1(x2)
#         x2 = self.convDrop(self.relu(self.bn2(self.conv2(x2))))
#         x2 = self.se2(x2)
#         x2 = self.convDrop(self.relu(self.bn3(self.conv3(x2))))
#         x2 = torch.mean(x2, 2)
#
#         # mu_2, std_2 = self.fc_mu_2(x2), F.softplus(self.fc_std_2(x2) - 5, beta=1)
#         # x2_ = self.reparameterise(mu_2, std_2)
#
#         x_all = torch.cat((x1, x2), dim=1)
#
#         # mu_cat, std_cat = self.fc_mu_2(x_all), F.softplus(self.fc_std_2(x_all) - 5, beta=1)
#         # x_all = self.reparameterise(mu_cat, std_cat)
#
#         x_all = self.convDrop(x_all)
#         x_out = self.fc(x_all)
#         x_out = F.log_softmax(x_out, dim=1)
#
#         return x_out, mu, std, mu_out, std_out
