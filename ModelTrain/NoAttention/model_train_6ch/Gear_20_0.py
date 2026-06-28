import numpy as np
import torch
import os
import random

from GearBox.Common.ModelTrainAndVisiable import ModelTrain
from GearBox.Common.NetWorkFrame import CNNNetWorkNoAttention

seed = 70
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
#加载数据
BearingLabel_20_0 = [
        "ball_20_0", "comb_20_0", "bearing_normal_20_0", "inner_20_0", "outer_20_0"
]
BearingLabel_30_2 = [
    "ball_30_2", "comb_30_2", "bearing_normal_30_2", "inner_30_2", "outer_30_2"
]
GearLabel_20_0 = [
    "chipped_20_0", "gear_normal_20_0", "miss_20_0", "root_20_0", "surface_20_0"
]
GearLabel_30_2 = [
    "chipped_30_2", "gear_normal_30_2", "miss_30_2", "root_30_2", "surface_30_2"
]
wavelet_path = "../../../wavelet_dataset/gear_20_0.npz"
wavelet_datasets = np.load(wavelet_path)
train_data = wavelet_datasets['train_data']
train_labels = wavelet_datasets['train_labels']
test_data = wavelet_datasets['test_data']
test_labels = wavelet_datasets['test_labels']
class_num = len(wavelet_datasets['class_names'])
print(wavelet_datasets.keys())
##加载bearing_20_0数据
gear_20_0_labels = [np.where(wavelet_datasets['class_names'] == label)[0][0] for label in GearLabel_20_0]
print(gear_20_0_labels)
train_data_mask = np.isin(train_labels, gear_20_0_labels)
test_data_mask = np.isin(test_labels, gear_20_0_labels)
train_data = train_data[train_data_mask]
train_labels = train_labels[train_data_mask]
test_data = test_data[test_data_mask]
test_labels = test_labels[test_data_mask]
datasets = {
    "train_data": train_data,
    "train_labels": train_labels,
    "test_data": test_data,
    "test_labels": test_labels
}

#定义模型
model = CNNNetWorkNoAttention(class_number=class_num)

model_trainer = ModelTrain(model=model)

#训练模型并保存
model_dir = '../SavedModels_6ch/Seed_70'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
path = os.path.join(model_dir, 'Gear_20_0.pth')
train_acc, test_acc, train_loss, test_loss = model_trainer.train_model(datasets, save_path=path)

#保存并可视化
graph_dir = '../SavedGraphs_6ch/Seed_70'
if not os.path.exists(graph_dir):
    os.makedirs(graph_dir)
graph_path = os.path.join(graph_dir, 'Gear_20_0.png')
model_trainer.visible(train_accuracy=train_acc, test_accuracy=test_acc, train_loss=train_loss, test_loss=test_loss, save_path=graph_path)
