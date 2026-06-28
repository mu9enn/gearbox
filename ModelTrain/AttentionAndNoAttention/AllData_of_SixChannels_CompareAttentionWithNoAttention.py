import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from GearBox.Common.NetWorkFrame import CNNNetWorkWithAttention,CNNNetWorkNoAttention
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from torchsummary import summary

random.seed(42)
# 设置NumPy的随机种子
np.random.seed(42)
# 设置操作系统层面的随机数生成器（部分库会用到）
os.environ['PYTHONHASHSEED'] = str(42)
torch.manual_seed(42)
# 设置所有GPU的随机种子
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

# 确保CUDA卷积操作的确定性
torch.backends.cudnn.deterministic = True
# 禁用cuDNN的自动优化（可能会降低速度，但保证确定性）
torch.backends.cudnn.benchmark = False
#加载小波变换预处理数据集
wavelet_path = '../../wavelet_dataset/wavelet_dataset.npz'
wavelet_datasets = np.load(wavelet_path)
print(wavelet_datasets.keys())
train_data = wavelet_datasets['train_data']
train_labels = wavelet_datasets['train_labels']
test_data = wavelet_datasets['test_data']
test_labels = wavelet_datasets['test_labels']
class_num = len(wavelet_datasets['class_names'])


def param_init(model):
    """
    针对SCAttention2D注意力网络的参数初始化函数
    适配结构：
    - CNNNetWorkWithAttention: conv_layers + attention(SCAttention2D) + classifier
    - CNNNetWorkNoAttention: feature_extractor
    初始化策略：
    1. 所有Conv2d（含SCAttention2D内的卷积）：Kaiming Normal（适配ReLU）
    2. 所有BatchNorm2d：gamma=1，beta=0
    3. 所有Linear（仅分类头）：Kaiming Normal（适配ReLU）
    """
    for name, module in model.named_modules():
        # 1. 卷积层初始化（包括SCAttention2D内的Conv2d）
        if isinstance(module, nn.Conv2d):
            # Kaiming初始化（fan_in模式，适配ReLU激活）
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            # 卷积层偏置初始化为0（如果有）
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

        # 2. 批归一化层初始化
        elif isinstance(module, nn.BatchNorm2d):
            # BN层权重(gamma)初始化为1，偏置(beta)初始化为0
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

        # 3. 全连接层初始化（仅分类头的Linear）
        elif isinstance(module, nn.Linear):
            # Kaiming初始化（适配ReLU激活）
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            # Linear层偏置初始化为0（如果有）
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

#定义关键通道选择函数
def core_channel_selection(data, index):
    return data[:,index,:,:]

#ndarray转换为张量
if isinstance(train_data, np.ndarray):
    train_data = torch.from_numpy(train_data).float()
    train_labels = torch.from_numpy(train_labels).long()
if isinstance(test_data, np.ndarray):
    test_data = torch.from_numpy(test_data).float()
    test_labels = torch.from_numpy(test_labels).long()

# print(train_data.shape, train_labels.shape)

#定义超参数
learning_rate = 0.00015
batch_size = 256
epochs = 30

#定义模型
cnn_attention = CNNNetWorkWithAttention(class_number=class_num)
cnn_no_attention = CNNNetWorkNoAttention(class_number=class_num)

#定义设备，优化器和损失函数并初始化模型参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
cnn_model_attention = cnn_attention.to(device)
cnn_model_no_attention = cnn_no_attention.to(device)
param_init(cnn_attention)
param_init(cnn_no_attention)
optimizer_attention = optim.Adam(cnn_attention.parameters(), lr=learning_rate)
optimizer_no_attention = optim.Adam(cnn_no_attention.parameters(), lr=learning_rate)



#选择关键通道数据并按照batch_size拆分数据集
core_channel_data_train = core_channel_selection(data=train_data, index=[1,2,3,5,6,7])
core_channel_data_test = core_channel_selection(data=test_data, index=[1,2,3,5,6,7])
train_datasets = TensorDataset(core_channel_data_train, train_labels)
test_datasets = TensorDataset(core_channel_data_test, test_labels)
batch_train_data = DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
batch_test_data = DataLoader(test_datasets, batch_size=batch_size, shuffle=False)

#开始训练
loss_epoch_train_attention = []
loss_epoch_test_attention = []
acc_epoch_train_attention = []
acc_epoch_test_attention = []

loss_epoch_train_no_attention = []
loss_epoch_test_no_attention = []
acc_epoch_train_no_attention = []
acc_epoch_test_no_attention = []

# SE_Weights = []
SC_channel_weights = []
SC_spatial_weights = []

for epoch in range(epochs):
    cnn_model_attention.train()
    cnn_model_no_attention.train()

    correct_num_train_attention = 0
    correct_num_test_attention = 0

    correct_num_train_no_attention = 0
    correct_num_test_no_attention = 0

    loss_sum_train_attention = 0
    loss_sum_test_attention = 0

    loss_sum_train_no_attention = 0
    loss_sum_test_no_attention = 0

    with tqdm(batch_train_data, desc=f"Train Epoch {epoch+1}/{epochs}") as pbar:
        for grad_batch_data, grad_batch_label in pbar:
            grad_batch_data = grad_batch_data.to(device)
            grad_batch_label = grad_batch_label.to(device)

            optimizer_attention.zero_grad()
            optimizer_no_attention.zero_grad()

            output_train_attention = cnn_model_attention(grad_batch_data)
            output_train_no_attention = cnn_model_no_attention(grad_batch_data)

            loss_train_attention = criterion(output_train_attention, grad_batch_label)
            loss_train_no_attention = criterion(output_train_no_attention, grad_batch_label)
            loss_sum_train_attention += loss_train_attention.item() * len(grad_batch_data)
            loss_sum_train_no_attention += loss_train_no_attention.item() * len(grad_batch_data)

            loss_train_attention.backward()
            loss_train_no_attention.backward()

            optimizer_attention.step()
            optimizer_no_attention.step()

            prediction_train_attention = output_train_attention.argmax(dim=1)
            prediction_train_no_attention = output_train_no_attention.argmax(dim=1)
            correct_num_attention = prediction_train_attention.eq(grad_batch_label).sum().item()
            correct_num_no_attention = prediction_train_no_attention.eq(grad_batch_label).sum().item()
            correct_num_train_attention += correct_num_attention
            correct_num_train_no_attention += correct_num_no_attention

            pbar.set_postfix({'Train Loss(attention)': loss_train_attention.item(), "Train Accuracy(attention)": correct_num_attention / len(grad_batch_data), 'Train Loss(no_attention)': loss_train_no_attention.item(), "Train Accuracy(no_attention)": correct_num_no_attention / len(grad_batch_data)})

        loss_epoch_train_attention.append(loss_sum_train_attention/len(train_datasets))
        loss_epoch_train_no_attention.append(loss_sum_train_no_attention/len(train_datasets))
        acc_epoch_train_attention.append(correct_num_train_attention/len(train_datasets))
        acc_epoch_train_no_attention.append(correct_num_train_no_attention/len(train_datasets))

    #开始评估
    cnn_model_attention.eval()
    cnn_model_no_attention.eval()
    with torch.no_grad():
        with tqdm(batch_test_data, desc=f"Test Epoch {epoch+1}/{epochs}") as qbar:
            for no_grad_data_batch, no_grad_data_label in qbar:
                no_grad_data_batch = no_grad_data_batch.to(device)
                no_grad_data_label = no_grad_data_label.to(device)

                output_test_attention = cnn_model_attention(no_grad_data_batch)
                output_test_no_attention = cnn_model_no_attention(no_grad_data_batch)

                loss_test_attention = criterion(output_test_attention, no_grad_data_label)
                loss_test_no_attention = criterion(output_test_no_attention, no_grad_data_label)
                loss_sum_test_attention += loss_test_attention.item() * len(no_grad_data_batch)
                loss_sum_test_no_attention += loss_test_no_attention.item() * len(no_grad_data_batch)

                prediction_test_attention = output_test_attention.argmax(dim=1)
                prediction_test_no_attention = output_test_no_attention.argmax(dim=1)
                correct_num_attention = prediction_test_attention.eq(no_grad_data_label).sum().item()
                correct_num_no_attention = prediction_test_no_attention.eq(no_grad_data_label).sum().item()
                correct_num_test_attention += correct_num_attention
                correct_num_test_no_attention += correct_num_no_attention

                qbar.set_postfix({"Test Loss(attention)": loss_test_attention.item(), "Test Accuracy(attention)": correct_num_attention/len(no_grad_data_batch), "Test Loss(no_attention)": loss_test_no_attention.item(), "Test Accuracy(no_attention)": correct_num_no_attention/len(no_grad_data_batch)})

            loss_epoch_test_attention.append(loss_sum_test_attention/len(test_datasets))
            loss_epoch_test_no_attention.append(loss_sum_test_no_attention/len(test_datasets))
            acc_epoch_test_attention.append(correct_num_test_attention/len(test_datasets))
            acc_epoch_test_no_attention.append(correct_num_test_no_attention/len(test_datasets))

    #保存通道权重
    # for layer in cnn_model_attention.children():
    #     for cnn_module in layer.children():
    #         if isinstance(cnn_module, SEBlock):
    #             weight_np = cnn_module.weights.cpu().detach().numpy()
    #             weight = weight_np.reshape(weight_np.shape[0], 6, -1).mean(axis=2).mean(axis=0)
    #             SE_Weights.append(weight)
    weights = cnn_attention.get_attention_weights()
    SC_channel_weights.append(weights['channel_weight'])
    SC_spatial_weights.append(weights['spatial_weight'])

    #设置早停
    if len(acc_epoch_train_attention) > 1:
        if ((abs(acc_epoch_test_attention[-1] - acc_epoch_test_attention[-2]) < 0.015) and
            (abs(acc_epoch_train_attention[-1] - acc_epoch_train_attention[-2]) < 0.015) and
            (abs(acc_epoch_test_attention[-1] - acc_epoch_train_attention[-1]) < 0.015) and
                (abs(acc_epoch_test_no_attention[-1] - acc_epoch_train_no_attention[-1]) < 0.015) and
                (abs(acc_epoch_test_attention[-1] - acc_epoch_test_no_attention[-1]) < 0.015)):
            break

current_dir = os.path.dirname(__file__)
SaveGraphs_dir = os.path.join(current_dir, 'SavedGraphs')
SaveModels_dir = os.path.join(current_dir, 'SavedModels')
SaveWeights_dir = os.path.join(current_dir, 'SavedWeights')
if not os.path.exists(SaveGraphs_dir):
    os.makedirs(SaveGraphs_dir)
if not os.path.exists(SaveModels_dir):
    os.makedirs(SaveModels_dir)
if not os.path.exists(SaveWeights_dir):
    os.makedirs(SaveWeights_dir)
SavePath_graphs = os.path.join(SaveGraphs_dir, 'TrainTestCurves.png')
SavePath_models_attention = os.path.join(SaveModels_dir, 'Attention_of_SixChannels.pth')
SavePath_models_no_attention = os.path.join(SaveModels_dir, 'NoAttention_of_SixChannels.pth')
SavePath_weights = os.path.join(SaveWeights_dir, 'SixChannelWeights.npz')

plt.rcParams.update({
    'font.sans-serif': ['Times New Roman'],
    'axes.unicode_minus': False,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'lines.linewidth': 1.8,
    'lines.markersize': 4,
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'font.family': 'serif',
})

actual_epochs = len(loss_epoch_test_attention)
epoch_x = range(1, actual_epochs + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), tight_layout=True)

# ========== 【优化2】动态调整标记点间隔 ==========
mark_step = max(1, actual_epochs // 5)  # 确保至少显示5个标记点

# 损失曲线
ax1.plot(epoch_x, loss_epoch_train_attention, label='Train (With Attention)',
         color='#1f77b4', marker='o', linestyle='-', markevery=mark_step)
ax1.plot(epoch_x, loss_epoch_train_no_attention, label='Train (No Attention)',
         color='#ff7f0e', marker='o', linestyle='-', markevery=mark_step)
ax1.plot(epoch_x, loss_epoch_test_attention, label='Test (With Attention)',
         color='#1f77b4', marker='s', linestyle='--', markevery=mark_step)
ax1.plot(epoch_x, loss_epoch_test_no_attention, label='Test (No Attention)',
         color='#ff7f0e', marker='s', linestyle='--', markevery=mark_step)

ax1.set_xlabel('Epoch', fontsize=12, labelpad=8)
ax1.set_ylabel('Loss', fontsize=12, labelpad=8)
ax1.set_title('Loss Curves', fontsize=14, fontweight='bold', pad=12)

# 刻度优化
tick_step = max(1, actual_epochs // 6)
ax1.set_xticks(range(1, actual_epochs + 1, tick_step))
ax1.tick_params(axis='both', labelsize=11)
ax1.grid(True, linestyle='--', color='#e0e0e0', linewidth=0.8)
ax1.legend(loc='upper right', fontsize=10, frameon=True, facecolor='white', edgecolor='gray',
           handlelength=2, borderaxespad=0.5)

# 准确率曲线
ax2.plot(epoch_x, acc_epoch_train_attention, label='Train (With Attention)',
         color='#1f77b4', marker='o', linestyle='-', markevery=mark_step)
ax2.plot(epoch_x, acc_epoch_train_no_attention, label='Train (No Attention)',
         color='#ff7f0e', marker='o', linestyle='-', markevery=mark_step)
ax2.plot(epoch_x, acc_epoch_test_attention, label='Test (With Attention)',
         color='#1f77b4', marker='s', linestyle='--', markevery=mark_step)
ax2.plot(epoch_x, acc_epoch_test_no_attention, label='Test (No Attention)',
         color='#ff7f0e', marker='s', linestyle='--', markevery=mark_step)

ax2.set_xlabel('Epoch', fontsize=12, labelpad=8)
ax2.set_ylabel('Accuracy', fontsize=12, labelpad=8)
ax2.set_title('Accuracy Curves', fontsize=14, fontweight='bold', pad=12)
ax2.set_ylim(0, 1.0)
ax2.set_yticks(np.arange(0, 1.01, 0.2))
ax2.set_xticks(range(1, actual_epochs + 1, tick_step))
ax2.tick_params(axis='both', labelsize=11)
ax2.grid(True, linestyle='--', color='#e0e0e0', linewidth=0.8)
ax2.legend(loc='lower right', fontsize=10, frameon=True, facecolor='white', edgecolor='gray',
           handlelength=2, borderaxespad=0.5)

# 保存图片
plt.savefig(SavePath_graphs, format='png', dpi=600, bbox_inches='tight', pad_inches=0.1)

# 释放内存
plt.close(fig)

# 保存模型
torch.save(cnn_model_attention.state_dict(), SavePath_models_attention)
torch.save(cnn_model_no_attention.state_dict(), SavePath_models_no_attention)


#保存权重
np.savez(SavePath_weights, SC_channel_weights = np.array(SC_channel_weights), SC_spatial_weights = np.array(SC_spatial_weights))


print("\n" + "=" * 50)
print("带注意力的模型结构")
print("=" * 50)
summary(model=cnn_model_attention, input_size=(6, 6, 1024), batch_size=batch_size, device='cpu')

print("\n" + "=" * 50)
print("不带注意力的模型结构")
print("=" * 50)
summary(model=cnn_model_no_attention, input_size=(6, 6, 1024), batch_size=batch_size, device='cpu')