import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import random

from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from GearBox.Common.NetWorkFrame import CNNNetWorkNoAttention

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

# ===================== 绘图参数 =====================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "axes.unicode_minus": False
})

#定义超参数、设备
batch_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BearingLabel_20_0 = [
        "ball_20_0", "bearing_normal_20_0", "comb_20_0",  "inner_20_0", "outer_20_0"
    ]
BearingLabel_30_2 = [
    "ball_30_2", "bearing_normal_30_2", "comb_30_2", "inner_30_2", "outer_30_2"
    ]
GearLabel_20_0 = [
    "chipped_20_0", "gear_normal_20_0",  "miss_20_0", "root_20_0", "surface_20_0"
    ]
GearLabel_30_2 = [
    "chipped_30_2", "gear_normal_30_2", "miss_30_2",  "root_30_2", "surface_30_2"
    ]

label_classes = [BearingLabel_20_0,BearingLabel_30_2,GearLabel_20_0,GearLabel_30_2]

def f(parameter, dataset, data_label, model):
    test_data = TensorDataset(dataset,data_label)
    batch_data = DataLoader(test_data, batch_size=batch_size)

    prediction_label = np.array([])
    real_labels = np.array([])

    model.load_state_dict(parameter)
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_set, batch_label in batch_data:
            batch_set = batch_set.to(device)
            batch_label = batch_label.to(device)

            prediction = model(batch_set)
            outcome = prediction.argmax(dim=1)
            out_np = outcome.detach().cpu().numpy()
            lab_np = batch_label.detach().cpu().numpy()
            prediction_label = np.hstack((prediction_label, out_np))
            real_labels = np.hstack((real_labels, lab_np))
    return prediction_label, real_labels


def precision_recall_fscore_support(real_labels, prediction_label, confusion_path, f1_path, class_names):

    mask = f1_path.split('\\')[-1].split('.')[0]
    cm = confusion_matrix(real_labels, prediction_label)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)  # 归一化

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)

    # 坐标与标签
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=11, weight='bold', rotation=45, ha='right', rotation_mode='anchor')
    ax.set_yticklabels(class_names, fontsize=11, weight='bold')

    # 数字显示
    thresh = cm_norm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f'{cm_norm[i, j]:.2f}',
                    ha='center', va='center',
                    color='white' if cm_norm[i, j] > thresh else 'black',
                    fontsize=11, weight='bold')

    # 轴标签
    ax.set_xlabel('Predicted Label', fontsize=13, weight='bold')
    ax.set_ylabel('True Label', fontsize=13, weight='bold')
    ax.set_title(f'Confusion Matrix({mask})', fontsize=14, weight='bold')

    # 颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(confusion_path, bbox_inches='tight')
    plt.close()

    report = classification_report(real_labels, prediction_label, target_names=class_names, output_dict=True)
    metrics = ['precision', 'recall', 'f1-score']
    datas = {}
    for m in metrics:
        datas[m] = []
        for cls in class_names:
            # 安全获取，不会报错
            datas[m].append(report.get(cls, {}).get(m, 0.0))

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))

    # 三个指标条形
    ax.bar(x - width, datas['precision'], width, label='Precision',
           color='#1F77B4', edgecolor='black', linewidth=0.8)
    ax.bar(x, datas['recall'], width, label='Recall',
           color='#FF7F0E', edgecolor='black', linewidth=0.8)
    ax.bar(x + width, datas['f1-score'], width, label='F1-score',
           color='#2CA02C', edgecolor='black', linewidth=0.8)

    # 数字标签
    def add_labels(rects):
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points",
                        ha='center', fontsize=10, weight='bold')

    for bar in [ax.containers[0], ax.containers[1], ax.containers[2]]:
        add_labels(bar)

    # 样式
    ax.set_ylabel('Score', fontsize=12, weight='bold')
    ax.set_title(f'Classification Metrics({mask})', fontsize=14, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=11, weight='bold')
    ax.legend(frameon=True, edgecolor='black', fontsize=11)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(f1_path, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    # 加载模型
    param_dir = 'SavedModels_6ch/Seed_49'
    param_file = [m for m in os.listdir(param_dir)]
    param_path = [os.path.join(param_dir, f) for f in param_file]
    # print(param_path)

    target_model = CNNNetWorkNoAttention(class_number=5)

    param = [torch.load(m, map_location='cpu') for m in param_path]

    data_dir = '..\\..\\wavelet_dataset'

    bearing_20_0_data = np.load(os.path.join(data_dir, 'bearing_20_0.npz'))['test_data']
    if isinstance(bearing_20_0_data, np.ndarray):
        bearing_20_0_data = torch.from_numpy(bearing_20_0_data).float().to(device)
    bearing_30_2_data = np.load(os.path.join(data_dir, 'bearing_30_2.npz'))['test_data']
    if isinstance(bearing_30_2_data, np.ndarray):
        bearing_30_2_data = torch.from_numpy(bearing_30_2_data).float().to(device)
    gear_20_0_data = np.load(os.path.join(data_dir, 'gear_20_0.npz'))['test_data']
    if isinstance(gear_20_0_data, np.ndarray):
        gear_20_0_data = torch.from_numpy(gear_20_0_data).float().to(device)
    gear_30_2_data = np.load(os.path.join(data_dir, 'gear_30_2.npz'))['test_data']
    if isinstance(gear_30_2_data, np.ndarray):
        gear_30_2_data = torch.from_numpy(gear_30_2_data).float().to(device)

    data = [bearing_20_0_data,bearing_30_2_data,gear_20_0_data,gear_30_2_data]

    bearing_20_0_label = np.load(os.path.join(data_dir, 'bearing_20_0.npz'))['test_labels']
    if isinstance(bearing_20_0_label, np.ndarray):
        bearing_20_0_label = torch.from_numpy(bearing_20_0_label).long().to(device)
    bearing_30_2_label = np.load(os.path.join(data_dir, 'bearing_30_2.npz'))['test_labels']
    if isinstance(bearing_30_2_label, np.ndarray):
        bearing_30_2_label = torch.from_numpy(bearing_30_2_label).long().to(device)
    gear_20_0_label = np.load(os.path.join(data_dir, 'gear_20_0.npz'))['test_labels']
    if isinstance(gear_20_0_label, np.ndarray):
        gear_20_0_label = torch.from_numpy(gear_20_0_label).long().to(device)
    gear_30_2_label = np.load(os.path.join(data_dir, 'gear_30_2.npz'))['test_labels']
    if isinstance(gear_30_2_label, np.ndarray):
        gear_30_2_label = torch.from_numpy(gear_30_2_label).long().to(device)

    label = [bearing_20_0_label,bearing_30_2_label,gear_20_0_label,gear_30_2_label]

    #定义保存路径
    save_dir = 'ConfusionAndF1/Seed_49'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    confusion_dir = os.path.join(save_dir, 'ConfusionMatrix')
    if not os.path.exists(confusion_dir):
        os.makedirs(confusion_dir)
    f1_dir = os.path.join(save_dir, 'F1')
    if not os.path.exists(f1_dir):
        os.makedirs(f1_dir)

    with torch.no_grad():
        with tqdm(param, total=len(param)) as pbar:
            for param in pbar:
                file_name = param_path[pbar.n].split('\\')[-1].split('.')[0]
                confusion_save_path = os.path.join(confusion_dir, f"{file_name}.png")
                f1_save_path = os.path.join(f1_dir, f"{file_name}.png")

                pre_label, real_label = f(parameter=param, dataset=data[pbar.n],data_label=label[pbar.n],model=target_model)
                precision_recall_fscore_support(real_labels=real_label,
                                                prediction_label=pre_label,
                                                confusion_path=confusion_save_path,
                                                f1_path=f1_save_path,
                                                class_names=label_classes[pbar.n])
