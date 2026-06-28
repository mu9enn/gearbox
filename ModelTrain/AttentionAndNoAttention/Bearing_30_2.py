import numpy as np
import torch
import os
import random

from GearBox.Common.ModelTrainAndVisiable import ParallelTraining
from GearBox.Common.NetWorkFrame import CNNNetWorkNoAttention,CNNNetWorkWithAttention

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
wavelet_path = "../../wavelet_dataset/wavelet_dataset.npz"
wavelet_datasets = np.load(wavelet_path)
train_data = wavelet_datasets['train_data']
train_labels = wavelet_datasets['train_labels']
test_data = wavelet_datasets['test_data']
test_labels = wavelet_datasets['test_labels']
class_num = len(wavelet_datasets['class_names'])
print(wavelet_datasets.keys())
##加载bearing_30_2数据
bearing_30_2_labels = [np.where(wavelet_datasets['class_names'] == label)[0][0] for label in BearingLabel_30_2]
print(bearing_30_2_labels)
train_data_mask = np.isin(train_labels, bearing_30_2_labels)
test_data_mask = np.isin(test_labels, bearing_30_2_labels)
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
model = [
    CNNNetWorkWithAttention(class_number=class_num),
    CNNNetWorkNoAttention(class_number=class_num)
]

#训练模型
attention_path = "SavedModels/Seed_49/Bearing_30_2_Attention.pth"
no_attention_path = "SavedModels/Seed_49/Bearing_30_2_NoAttention.pth"
save_graph_path = "SavedGraphs/Seed_49/Bearing_30_2.png"
model_trainer = ParallelTraining(models=model)
acc_train_att, acc_test_att,loss_train_att, loss_test_att, acc_train_no_att, acc_test_no_att,loss_train_no_att,loss_test_no_att,sc_ch_weights,sc_spa_weights = model_trainer.model_train(datasets=datasets, attention_save_path=attention_path, no_attention_save_path=no_attention_path)

#保存通道和空间
save_path = './SavedWeights/Seed_49/Bearing_30_2.npz'
weights = {
    "channel_weight": sc_ch_weights,
    "spatial_weight": sc_spa_weights}

np.savez(save_path, weights=weights)
model_trainer.visible(loss_epoch_train_attention=loss_train_att,
                       loss_epoch_test_attention=loss_test_att,
                       loss_epoch_train_no_attention=loss_train_no_att,
                       loss_epoch_test_no_attention=loss_test_no_att,
                       acc_epoch_train_attention=acc_train_att,
                       acc_epoch_test_attention=acc_test_att,
                       acc_epoch_train_no_attention=acc_train_no_att,
                       acc_epoch_test_no_attention=acc_test_no_att,
                       SavePath_graphs=save_graph_path)