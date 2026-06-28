import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from GearBox.Common.NetWorkFrame import CNNNetWorkNoAttention, BaseModel4Channel

# ===================== 【你只改这里 4 个位置】 =====================
model_6ch_path = "../SavedModels_6ch\\Seed_42\\Bearing_30_2.pth"  # 你的6通道模型
model_4ch_path = "..\\SavedModels_4ch\\Seed_42\\Bearing_30_2.pth"  # 你的4通道模型

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#加载数据
data_path = "..\\..\\..\\wavelet_dataset\\bearing_20_0.npz"
sample = np.load(data_path)['test_data']
sample_4ch = sample[:,[1,2,3,4],:,:]
sample_6ch = sample
labels = np.load(data_path)['test_labels']

#转张量
sample_4ch_tensor = torch.from_numpy(sample_4ch).float().to(device)
sample_6ch_tensor = torch.from_numpy(sample_6ch).float().to(device)
label_tensor = torch.from_numpy(labels).long().to(device)

dataset_4ch = TensorDataset(sample_4ch_tensor, label_tensor)
dataset_6ch = TensorDataset(sample_6ch_tensor, label_tensor)

test_loader_4ch = DataLoader(dataset_4ch, batch_size=128, shuffle=False)
test_loader_6ch = DataLoader(dataset_6ch, batch_size=128, shuffle=False)


# ---------------------- 1. 加载模型 ----------------------
model6 = CNNNetWorkNoAttention(class_number=5)
model4 = BaseModel4Channel(class_number=5)
model6.load_state_dict(torch.load(model_6ch_path, map_location=device))
model4.load_state_dict(torch.load(model_4ch_path, map_location=device))
model6.eval()
model4.eval()

# ---------------------- 2. 计算参数量 ----------------------
def count_params(model):
    return sum(p.numel() for p in model.parameters())

params6 = count_params(model6) / 1e6  # 转 M
params4 = count_params(model4) / 1e6

# ---------------------- 3. 测试准确率 ----------------------
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

acc6 = get_acc(model6, test_loader_6ch)
acc4 = get_acc(model4, test_loader_4ch)

# ---------------------- 4. 平均推理时间（100次） ----------------------
def infer_time(model, loader, n_samples=100):
    model.eval()
    samples = []
    # 从测试集随机拿 100 条真实数据
    for data, _ in loader:
        samples.append(data)
        if len(samples) >= n_samples:
            break
    # 堆叠成 100 条
    real_data = torch.cat(samples, dim=0)[:n_samples].to(device)

    # 开始计时推理
    start_time = time.time()
    with torch.no_grad():
        for i in range(n_samples):
            model(real_data[i:i + 1])  # 一次喂一条，真实单样本时间
    total_time = time.time() - start_time
    return (total_time / n_samples) * 1000  # 转 ms

time6 = infer_time(model6, loader=test_loader_6ch)
time4 = infer_time(model4, loader=test_loader_4ch)

# ---------------------- 5. 打印结果（你可以直接填表5-1） ----------------------
print("==== 表5-1 真实实验结果 ====")
print(f"6通道 准确率: {acc6:.2%} | 参数量: {params6:.2f}M | 推理时间: {time6:.2f}ms")
print(f"4通道 准确率: {acc4:.2%} | 参数量: {params4:.2f}M | 推理时间: {time4:.2f}ms")

plt.rcParams['font.family'] = 'Times New Roman'  # 新罗马字体
plt.rcParams['mathtext.fontset'] = 'stix'

metrics = ["Accuracy (%)", "Parameters (M)", "Inference Time (ms)"]
model6_vals = [acc6*100, params6, time6]
model4_vals = [acc4*100, params4, time4]

x = np.arange(len(metrics))
width = 0.32  # 柱子宽度微调，更美观

plt.figure(figsize=(11, 6), dpi=600)  # 画布更大、DPI=600

# 画柱状图
bars1 = plt.bar(x - width/2, model6_vals, width, label='6-Channel Full Sensors', color='#2E86AB', alpha=0.9)
bars2 = plt.bar(x + width/2, model4_vals, width, label='4-Channel Causal Optimization', color='#F24236', alpha=0.9)

# 字体全部放大
plt.xlabel('Evaluation Metrics', fontsize=18, fontweight='bold')
plt.ylabel('Value', fontsize=18, fontweight='bold')
plt.title('Performance Comparison between 6-Channel and 4-Channel Models', fontsize=20, fontweight='bold')

plt.xticks(x, metrics, fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16, frameon=True, shadow=False)

# 柱子顶部显示数值（更专业）
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=16, fontweight='bold')

add_labels(bars1)
add_labels(bars2)

plt.grid(alpha=0.2, axis='y')
plt.tight_layout()
plt.savefig("Performance_Comparison(Bearing_30_2).png", dpi=600, bbox_inches='tight')
plt.close()