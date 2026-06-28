import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from GearBox.Common.NetWorkFrame import BaseModel,BaseModelFreq,CNNNetWorkNoAttention
from torch.utils.data import TensorDataset, DataLoader

def get_acc(model, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = torch.argmax(out, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total



#加载数据
data_dir = "..\\..\\..\\wavelet_dataset"
files = ['Bearing_20_0', 'Bearing_30_2', 'Gear_20_0', 'Gear_30_2']
cross_cond_keys = ['Bearing_30_2', 'Bearing_20_0', 'Gear_30_2','Gear_20_0']
cross_cond_data_dict = {}#跨工况数据集
cross_cond_label_dict = {}
data_dict = {}
label_dict = {}
for idx, file in enumerate(files):
    path = os.path.join(data_dir, f'{file}.npz')
    test_data = np.load(path)['test_data']
    test_label = np.load(path)['test_labels']
    #跨工况数据字典
    cross_cond_data_dict[cross_cond_keys[idx]] = test_data
    cross_cond_label_dict[cross_cond_keys[idx]] = test_label

    #非跨工况数据字典
    data_dict[file] = test_data
    label_dict[file] = test_label

#测试准确率
def test_acc(model, data, label, parameter_path):
    model.load_state_dict(torch.load(parameter_path, map_location=device))
    model.to(device)
    model.eval()

    data = torch.Tensor(data).float().to(device)
    label = torch.Tensor(label).long().to(device)

    dataset = TensorDataset(data, label)
    test_loader = DataLoader(dataset, batch_size=128, shuffle=False)
    acc = get_acc(model, test_loader)

    return acc


#因果精简后的模型与参数
parameter_dir = "..\\SavedModels_6ch\\Seed_42"
parameter_list = os.listdir(parameter_dir)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNNetWorkNoAttention(class_number=5)

#通道精简后的模型与参数
freq_parameter_dir = "..\\SavedModels_freq\\Seed_42"
freq_parameter_list = os.listdir(freq_parameter_dir)
model_freq = BaseModelFreq(class_number=5)

#频率层级精简后的模型与参数
base_parameter_dir = "..\\BaseModels\\Seed_42"
base_parameter_list = os.listdir(base_parameter_dir)
model_base = BaseModel(class_number=5, layer_number=4)

#固定训练工况准确率
full_cond_acc = []
for param in parameter_list:
    all_path = os.path.join(parameter_dir, param)
    freq_path = os.path.join(freq_parameter_dir, param)
    base_path = os.path.join(base_parameter_dir, param)

    key = param.split('.')[0]

    #分工况的数据准备
    base_data_mid = data_dict[key][:,[1,2,3,4],:,:]
    base_data = base_data_mid[:,:,[1,2,3,5],:]
    data = data_dict[key]
    freq_data = data_dict[key][:,:,[1,2,3,5],:]
    labels = label_dict[key]

    #单工况条件下三实验组测试准确率
    acc_list = []

    acc_base = test_acc(model_base, base_data, labels, base_path)
    acc_list.append(acc_base)
    acc_freq = test_acc(model_freq, freq_data, labels, freq_path)
    acc_list.append(acc_freq)
    acc_all = test_acc(model, data, labels, all_path)
    acc_list.append(acc_all)

    #全工况准确率
    full_cond_acc.append(acc_list)

#跨工况准确率
full_cross_cond_acc = []
for param in parameter_list:
    all_path = os.path.join(parameter_dir, param)
    freq_path = os.path.join(freq_parameter_dir, param)
    base_path = os.path.join(base_parameter_dir, param)

    key = param.split('.')[0]

    # 分工况的数据准备
    base_data_mid = cross_cond_data_dict[key][:, [1, 2, 3, 4], :, :]
    base_data = base_data_mid[:, :, [1, 2, 3, 5], :]
    data = cross_cond_data_dict[key]
    freq_data = cross_cond_data_dict[key][:, :, [1, 2, 3, 5], :]
    labels = cross_cond_label_dict[key]

    # 单工况条件下三实验组测试准确率
    acc_list = []

    acc_base = test_acc(model_base, base_data, labels, base_path)
    acc_list.append(acc_base)
    acc_freq = test_acc(model_freq, freq_data, labels, freq_path)
    acc_list.append(acc_freq)
    acc_all = test_acc(model, data, labels, all_path)
    acc_list.append(acc_all)

    # 全工况准确率
    full_cross_cond_acc.append(acc_list)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学字符匹配新罗马

acc_fixed = np.array(full_cond_acc)
acc_cross = np.array(full_cross_cond_acc)

# 工况名称（横坐标）
conditions = ['Bearing_20_0', 'Bearing_30_2', 'Gear_20_0', 'Gear_30_2']

# 模型名称（图例）
model_names = [
    'Channel & Frequency Pruned',
    'Frequency Pruned',
    'Full Feature'
]

# 配色（清晰可用于论文）
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# ===================== 图1：固定训练工况准确率 =====================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

x = np.arange(len(conditions))
width = 0.25

# 绘制三根柱子
ax1.bar(x - width, acc_fixed[:, 0], width, label=model_names[0], color=colors[0])
ax1.bar(x, acc_fixed[:, 1], width, label=model_names[1], color=colors[1])
ax1.bar(x + width, acc_fixed[:, 2], width, label=model_names[2], color=colors[2])

ax1.set_xlabel('Working Conditions', fontsize=12, fontweight='bold')
ax1.set_ylabel('Diagnostic Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Accuracy under Fixed Training Conditions', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(conditions, rotation=15, fontsize=10)
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3, linestyle='--')
ax1.set_ylim(0, 1.05)

# ===================== 图2：跨工况准确率 =====================
ax2.bar(x - width, acc_cross[:, 0], width, label=model_names[0], color=colors[0])
ax2.bar(x, acc_cross[:, 1], width, label=model_names[1], color=colors[1])
ax2.bar(x + width, acc_cross[:, 2], width, label=model_names[2], color=colors[2])

ax2.set_xlabel('Working Conditions', fontsize=12, fontweight='bold')
ax2.set_ylabel('Diagnostic Accuracy', fontsize=12, fontweight='bold')
ax2.set_title('Cross-condition Generalization Accuracy', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(conditions, rotation=15, fontsize=10)
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3, linestyle='--')
ax2.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig('Cross_condition_acc_comparison.png', dpi=600, bbox_inches='tight')