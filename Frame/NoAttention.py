import torch
import torch.nn as nn

class CNNNetWorkNoAttention(nn.Module):
    def __init__(self, class_number):
        super(CNNNetWorkNoAttention, self).__init__()
        self.class_number = class_number
        if isinstance(class_number, torch.Tensor):
            self.class_number = class_number.item()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(3, 3),padding=(1, 1), stride=(1, 1)),#(b,6,6,1024)
            nn.BatchNorm2d(num_features=12),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4), padding=(0, 0)),#(b,12,6,256)
            nn.Flatten(),
            nn.Linear(in_features=12 * 6 * 256, out_features=256),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=256, out_features=self.class_number)
        )

    def forward(self, x):
        return self.feature_extractor(x)

# 2. 实例化模型 (假设类别数为 10)
model = CNNNetWorkNoAttention(class_number=5)

# 3. 创建一个符合你输入维度的虚拟张量 (Batch_Size=1, Channels=6, H=6, W=1024)
dummy_input = torch.randn(1, 6, 6, 1024)

# 4. 导出为 ONNX 文件
torch.onnx.export(
    model,
    dummy_input,
    "CNNNetWorkNoAttention.onnx",
    export_params=True,
    opset_version=18,
    input_names=['Input'],
    output_names=['Output']
)
print("模型已成功导出为 CNNNetWorkNoAttention.onnx")