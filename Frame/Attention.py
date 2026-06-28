import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class SCAttention2D(nn.Module):
    def __init__(self, channel=6, kernel_size=7, weight_decay=1e-4):
        super().__init__()
        self.weight_decay = weight_decay

        # 注意：这里我们注释掉了原来的池化层，改在 forward 中使用算子
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)

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

        # 使用 torch.mean 和 torch.amax 替代原来的自适应池化
        # dim=(2, 3) 代表在 Height 和 Width 维度上操作，keepdim=True 保持形状为 (B, C, 1, 1)
        avg_pool_x = torch.mean(x, dim=(2, 3), keepdim=True)
        max_pool_x = torch.amax(x, dim=(2, 3), keepdim=True)

        # 通道注意力 + 归一化（和为1）
        avg_out = self.fc(avg_pool_x)
        max_out = self.fc(max_pool_x)

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

# 2. 实例化模型
model = CNNNetWorkWithAttention(class_number=5)

# 3. 创建一个符合你输入维度的虚拟张量 (Batch_Size=1, Channels=6, H=6, W=1024)
dummy_input = torch.randn(1, 6, 6, 1024)

# 4. 导出为 ONNX 文件
torch.onnx.export(
    model,
    dummy_input,
    "CNNNetWorkWithAttention.onnx",
    export_params=True,
    opset_version=18,
    input_names=['Input'],
    output_names=['Output']
)
print("模型已成功导出为 CNNNetWorkWithAttention.onnx")