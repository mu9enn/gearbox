import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

import numpy as np
import os
import random
import torch.nn.functional as F

seed = 49
random.seed(seed)
# 设置NumPy的随机种子
np.random.seed(seed)
# 设置操作系统层面的随机数生成器（部分库会用到）
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
# 设置所有GPU的随机种子
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# 确保CUDA卷积操作的确定性
torch.backends.cudnn.deterministic = True
# 禁用cuDNN的自动优化（可能会降低速度，但保证确定性）
torch.backends.cudnn.benchmark = False

class SEBlock(nn.Module):
    def __init__(self, channel, reduction):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_features=channel, out_features=channel // reduction),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=channel // reduction, out_features=channel),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=channel, out_features=channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y_weights = self.fc(y).view(b, c, 1, 1)
        # 移除：self.weights = y_weights.view(b, c)  # 注释掉权重保存，避免钩子触发
        return x * y_weights


class SCAttention2D(nn.Module):
    def __init__(self, channel=6, kernel_size=7, weight_decay=1e-4):
        super().__init__()
        self.weight_decay = weight_decay

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            weight_norm(nn.Conv2d(channel, channel // 2, 1, bias=False)),
            nn.ReLU(inplace=False),
            weight_norm(nn.Conv2d(channel // 2, channel, 1, bias=False))
        )

        self.conv = weight_norm(nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False))

        self.sigmoid = nn.Sigmoid()

        self.channel_weight = None
        self.spatial_weight = None

    def forward(self, x):
        residual = x
        B, C, H, W = x.shape

        # 通道注意力 + 归一化（和为1）
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_weight = self.sigmoid(avg_out + max_out)
        channel_weight = channel_weight / (channel_weight.sum(dim=1, keepdim=True) + 1e-8)

        # 空间注意力
        avg_out_s = torch.mean(x, dim=1, keepdim=True)
        max_out_s, _ = torch.max(x, dim=1, keepdim=True)
        spatial_weight = self.sigmoid(self.conv(torch.cat([avg_out_s, max_out_s], dim=1)))

        # 加权
        x = x * channel_weight
        x = x * spatial_weight
        x = x + residual

        # 保存权重
        self.channel_weight = channel_weight.detach().clone()
        self.spatial_weight = spatial_weight.detach().clone()

        return x



class CNNNetWorkWithAttention(nn.Module):
    def __init__(self, class_number):
        super().__init__()
        self.class_number = class_number
        if isinstance(class_number, torch.Tensor):
            self.class_number = class_number.item()

        self.att = SCAttention2D()

        # 卷积层
        self.conv_layers = nn.Sequential(
            nn.Conv2d(6, 12, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d((1, 4), stride=(1, 4))
        )

        # 分类层
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12 * 6 * 256, 256),
            nn.ReLU(),
            nn.Linear(256, self.class_number)
        )

    def forward(self, x):
        x = self.att(x)

        x = self.conv_layers(x)

        x = self.classifier(x)
        return x

    def get_attention_weights(self):
        weights = {
            'channel_weight': self.att.channel_weight,
            'spatial_weight': self.att.spatial_weight
        }
        return weights

class CNNNetWorkNoAttention(nn.Module):
    def __init__(self, class_number):
        super(CNNNetWorkNoAttention, self).__init__()
        self.class_number = class_number
        if isinstance(class_number, torch.Tensor):
            self.class_number = class_number.item()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(3, 3),padding=(1, 1), stride=(1, 1)),#(b,6,6,512)
            nn.BatchNorm2d(num_features=12),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4), padding=(0, 0)),#(b,12,6,128)
            nn.Flatten(),
            nn.Linear(in_features=12 * 6 * 128, out_features=256),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=256, out_features=self.class_number)
        )

    def forward(self, x):
        return self.feature_extractor(x)

class BaseModel4Channel(nn.Module):
    def __init__(self, class_number):
        super(BaseModel4Channel, self).__init__()
        self.class_number = class_number
        if isinstance(class_number, torch.Tensor):
            self.class_number = class_number.item()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            # (b,8,6,1024)
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4), padding=(0, 0)),  # (b,8,6,256)
            nn.Flatten(),
            nn.Linear(in_features=8 * 6 * 256, out_features=256),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=256, out_features=self.class_number)
        )

    def forward(self, x):
        return self.feature_extractor(x)

class BaseModel(nn.Module):
    def __init__(self, class_number, layer_number):
        super(BaseModel, self).__init__()
        self.class_number = class_number
        self.layer_number = layer_number
        if isinstance(class_number, torch.Tensor):
            self.class_number = class_number.item()
        if isinstance(layer_number, torch.Tensor):
            self.layer_number = layer_number.item()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            # (b,8,layer_number,1024)
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4), padding=(0, 0)),  # (b,8,layer_number,256)
            nn.Flatten(),
            nn.Linear(in_features=8 * self.layer_number * 256, out_features=256),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=256, out_features=self.class_number)
        )

    def forward(self, x):
        return self.feature_extractor(x)

class BaseModelFreq(nn.Module):
    def __init__(self, class_number):
        super(BaseModelFreq, self).__init__()
        self.class_number = class_number
        if isinstance(class_number, torch.Tensor):
            self.class_number = class_number.item()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            # (b,12,4,1024)
            nn.BatchNorm2d(num_features=12),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4), padding=(0, 0)),  # (b,12,4,256)
            nn.Flatten(),
            nn.Linear(in_features=12 * 4 * 256, out_features=256),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=256, out_features=self.class_number)
        )

    def forward(self, x):
        return self.feature_extractor(x)

class DynamicMIRegularizer(nn.Module):
    def __init__(self, non_causal_pairs: list, temp: float = 0.1, device: str = "cpu"):
        """
        动态互信息正则器，训练中持续解耦非因果节点
        :param non_causal_pairs: 预划分的非因果节点对列表 [(u,v)]
        :param temp: InfoNCE 温度系数，控制分布平滑度
        :param device: 训练设备
        """
        super().__init__()
        self.non_causal_pairs = non_causal_pairs
        self.temp = temp
        self.device = device

    def forward(self, node_emb: torch.Tensor):
        """
        :param node_emb: 节点嵌入 [num_nodes, hidden_dim]
        :return: mi_loss: 互信息正则损失 (标量)
        """
        if len(self.non_causal_pairs) == 0:
            return torch.tensor(0.0, device=self.device)

        total_mi = 0.0
        num_pairs = len(self.non_causal_pairs)

        for u, v in self.non_causal_pairs:
            # 取出两个非因果节点的嵌入 [1, dim]
            emb_u = node_emb[u:u+1]
            emb_v = node_emb[v:v+1]

            # InfoNCE 计算互信息下界
            sim = torch.matmul(emb_u, emb_v.t()) / self.temp
            log_prob = F.log_softmax(sim, dim=1)
            # 单对节点的MI损失
            pair_mi = -torch.mean(log_prob)
            total_mi += pair_mi

        # 平均所有非因果对的MI损失
        mi_loss = total_mi / num_pairs
        return mi_loss

class StatNodeEncoder(nn.Module):
    """
    输入：单节点4维统计特征 [RMS, mean, var, kurt]
    输出：d维隐表征，用于内积相似度计算
    """
    def __init__(self, in_dim=4, hidden=32, out_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim)
        )
        # 输出归一化，内积等价余弦相似度，值域[-1,1]
        self.temp = nn.Parameter(torch.tensor(0.1))

    def forward(self, node_4d_feat):
        # node_4d_feat: [batch, 18, 4]
        bsz, num_node, _ = node_4d_feat.shape
        feat_flat = node_4d_feat.reshape(-1, 4)
        h = self.mlp(feat_flat)
        h = F.normalize(h, p=2, dim=-1) / self.temp  # 归一化+温度缩放
        return h.reshape(bsz, num_node, -1)  # [batch, 18, out_dim]


class MINET(nn.Module):
    def __init__(self, d=32, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x, y):
        # x,y shape: [N,32]
        xy = torch.cat([x, y], dim=-1)
        return self.net(xy)

class GNNCausalSEU(nn.Module):
    def __init__(self, class_number=5, causal_pairs=None, non_causal_pairs=None, causal_link=None):
        super(GNNCausalSEU, self).__init__()
        if isinstance(class_number, torch.Tensor):
            self.class_number = class_number.item()
        else:
            self.class_number = class_number

        # self.reg_ratio = reg_ratio
        self.causal_pairs = causal_pairs
        self.non_causal_pairs = non_causal_pairs
        self.causal_link = causal_link
        self.node_encoder = StatNodeEncoder()

        num_causal_edge = len(self.causal_link)
        # 原始参数保留，不修改外部传参
        self.causal_ratio1 = nn.Parameter(torch.ones(num_causal_edge)) if num_causal_edge > 0 else None
        self.causal_ratio2 = nn.Parameter(torch.ones(num_causal_edge)) if num_causal_edge > 0 else None
        self.causal_ratio3 = nn.Parameter(torch.ones(num_causal_edge)) if num_causal_edge > 0 else None
        self.causal_ratio4 = nn.Parameter(torch.ones(num_causal_edge)) if num_causal_edge > 0 else None
        self.causal_ratio5 = nn.Parameter(torch.ones(num_causal_edge)) if num_causal_edge > 0 else None


        self.link_ratio = nn.Parameter(torch.ones(3, 30)).reshape(3, 1, 30, 1)

        # 卷积头不变，自动计算维度
        self.cov = nn.Sequential(
            nn.Conv1d(in_channels=30, out_channels=36, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(36),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Conv1d(in_channels=36, out_channels=48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(48),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )
        dummy_feat = torch.randn(1, 30, 32)
        conv_out = self.cov(dummy_feat)
        flat_dim = conv_out.numel()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(120, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, self.class_number)
        )

        self.regularization_loss = None
        self.common_norm = nn.InstanceNorm1d(30, affine=True)
        # self.common_norm1 = nn.InstanceNorm1d(120, affine=True)

        self.T_net = MINET(d=4)

    # 计算一对节点Z_i,Z_j的互信息下界估计值
    def mine_estimate(self,zi, zj):
        N = zi.shape[0]
        # 联合分布样本 T(x,y)
        t_joint = self.T_net(zi, zj)  # [N,1]
        # 打乱zj构造边缘独立样本 P(x)P(y)
        perm = torch.randperm(N)
        zj_shuffle = zj[perm]
        t_marginal = self.T_net(zi, zj_shuffle)  # [N,1]

        # MINE下界 I >= E[T_joint] - log(E[exp(T_marginal)])
        exp_t_marg = torch.exp(t_marginal)
        mi_lower = torch.mean(t_joint) - torch.log(torch.mean(exp_t_marg) + 1e-8)
        # 我们要最小化互信息，所以loss取mi_lower本身
        return mi_lower

    def forward(self, x):
        out1 = torch.zeros_like(x)
        for idx, (u, v) in enumerate(self.causal_link):
            out1[:, v, :] += x[:, u, :] * self.causal_ratio1[idx]

        out1 = self.common_norm(out1)

        out2 = torch.zeros_like(x)
        for idx, (u, v) in enumerate(self.causal_link):
            out2[:, v, :] += out1[:, u, :] * self.causal_ratio2[idx]

        out2 = self.common_norm(out2)

        out3 = torch.zeros_like(x)
        for idx, (u, v) in enumerate(self.causal_link):
            out3[:, v, :] += out2[:, u, :] * self.causal_ratio3[idx]

        out3 = self.common_norm(out3)
#
#         out4 = torch.zeros_like(x)
#         for idx, (u, v) in enumerate(self.causal_link):
#             out4[:, v, :] += out3[:, u, :] * self.causal_ratio4[idx]
#
# #         out4 = self.common_norm(out4)
#
#         out5 = torch.zeros_like(x)
#         for idx, (u, v) in enumerate(self.causal_link):
#             out5[:, v, :] += out4[:, u, :] * self.causal_ratio5[idx]

#         out5 = self.common_norm(out5)

        out = torch.sum(self.link_ratio * torch.stack([out1, out2, out3], dim=0), dim=0)#(n,30,4)
        # out = self.common_norm(out1)
        # out = torch.relu(out)
        # out = self.common_norm(out)
        # out = torch.relu(out)
        out = out + x
        # MI正则损失（作用融合后特征，对齐因果聚合空间）
        # if self.training:
        #     self.regularization_loss = 0.0
        #     for u, v in self.non_causal_pairs:
        #         zi = out[:, u, :]
        #         zj = out[:, v, :]
        #         mi_loss = self.mine_estimate(zi, zj)
        #         self.regularization_loss += mi_loss
        # out = torch.relu(out)
        # out = self.node_encoder(out)

        # out = torch.cat([out1, out2, out3, out4], dim=1)#(n,120,4)
        # out = self.common_norm1(out)
        logits = self.classifier(x)
        return logits


class GNNCausalCWRU(nn.Module):
    def __init__(self, class_number=5, unique_index=None, common_index=None, common_ratio=None, reg_ratio=None):
        super(GNNCausalCWRU, self).__init__()
        if isinstance(class_number, torch.Tensor):
            self.class_number = class_number.item()
        else:
            self.class_number = class_number
        if unique_index is not None:
            if isinstance(unique_index, np.ndarray):
                unique_index = torch.from_numpy(unique_index)
            self.register_buffer('unique_index', unique_index.detach().clone().long())
        else:
            self.register_buffer('unique_index', torch.empty(2, 0, dtype=torch.long))

        if common_index is not None:
            if isinstance(common_index, np.ndarray):
                common_index = torch.from_numpy(common_index)
            self.register_buffer('common_index', common_index.detach().clone().long())
        else:
            self.register_buffer('common_index', torch.empty(2, 0, dtype=torch.long))

        # Ratio 参数
        self.unique_ratio = nn.Parameter(torch.randn(self.unique_index.shape[1]))
        self.common_ratio = nn.Parameter(common_ratio)

        self.classifier = nn.Sequential(
            nn.Conv1d(in_channels=18, out_channels=36, kernel_size=7, stride=3, padding=1),#(n,36,170)
            nn.BatchNorm1d(36),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=10, stride=10, padding=0),#(n,36,17)
            nn.Flatten(),
            nn.Linear(612, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, class_number)
        )
        self.regularization_loss = None
        self.reg_ratio = reg_ratio
        self.common_norm = nn.InstanceNorm1d(18, affine=True)
        if self.common_index.numel() > 0:
            self.common_nodes = torch.unique(torch.cat([self.common_index[0], self.common_index[1]]))

    def forward(self, x):
        out = x.clone()

        # 独有因果边传播
        for idx, (i, j) in enumerate(zip(self.unique_index[0], self.unique_index[1])):
            msg = x[:, i, :] * self.unique_ratio[idx]
            out[:, j, :] = out[:, j, :] + msg

        # 公共因果边传播
        for idx, (i, j) in enumerate(zip(self.common_index[0], self.common_index[1])):
            msg = x[:, i, :] * self.common_ratio[idx]
            out[:, j, :] = out[:, j, :] + msg

        if self.common_nodes.numel() > 0:
            out_final = out.clone()
            full_normed_out = self.common_norm(out_final)
            out = full_normed_out

        if self.training:
            unique_reg = torch.sum(torch.abs(self.unique_ratio))

            all_nodes = torch.arange(18, device=x.device)
            used_nodes = torch.cat([self.unique_index[0], self.unique_index[1],
                                    self.common_index[0], self.common_index[1]])
            isolated_nodes = torch.tensor([n for n in all_nodes if n not in used_nodes], device=x.device)

            isolated_reg = 0.0
            if len(isolated_nodes) > 0:
                isolated_reg = torch.mean(torch.abs(out[:, isolated_nodes, :]))


            self.regularization_loss = self.reg_ratio[0] * unique_reg + self.reg_ratio[1] * isolated_reg

        return self.classifier(out)