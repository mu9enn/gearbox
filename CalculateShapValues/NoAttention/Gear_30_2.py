import numpy as np
import torch
import shap
import os
import random

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
wavelet_path = '../../wavelet_dataset/gear_30_2.npz'
wavelet_datasets = np.load(wavelet_path)
# print(wavelet_datasets.keys())
# print(wavelet_datasets['class_names'])
train_data = wavelet_datasets['train_data']
train_labels = wavelet_datasets['train_labels']
test_data = wavelet_datasets['test_data']
test_labels = wavelet_datasets['test_labels']
class_num = len(wavelet_datasets['class_names'])
# print(train_data.shape[0] + test_data.shape[0])
# print(train_data.shape)
# print(test_data.shape)

#模型加载
model_path = '../../ModelTrain/NoAttention/SavedModels_6ch/Seed_70/Gear_30_2.pth'

target_model = CNNNetWorkNoAttention(class_num)

target_model.load_state_dict(torch.load(model_path, map_location='cpu'))

#定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
target_model.to(device)
target_model.eval()

#定义超参数
explainable_number_every_fault = 100
background_sample_number = 100

#定义关键通道筛选函数
def core_channel_selection(datasets, index):
    return datasets[:,index,:,:]
#定义shap计算函数
def calculate_shap_values(model, background_datasets, explain_datasets):
    #数据转为张量
    if isinstance(explain_datasets, np.ndarray):
        explain_datasets = torch.from_numpy(explain_datasets).float().to(device)
    if isinstance(background_datasets, np.ndarray):
        background_datasets = torch.from_numpy(background_datasets).float().to(device)
    explainer = shap.GradientExplainer(model, background_datasets)
    shap_values_batch = explainer.shap_values(X=explain_datasets)
    return shap_values_batch

if __name__ == '__main__':
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
    train_data = core_channel_selection(train_data, index=range(6))
    test_data = core_channel_selection(test_data, index=range(6))

    #构建Gear_30_2解释数据集和背景数据集,每种故障类型取50个样本
    ##获取每种故障标签
    Gear_30_2_fault_labels = [np.where(wavelet_datasets['class_names'] == label)[0][0] for label in GearLabel_30_2]
    # print(fault_labels)
    back_label = np.where(wavelet_datasets['class_names'] == 'gear_normal_30_2')[0][0]
    # print(back_label)
    ##构建背景数据集(依据数据集划分比例，训练集取0.8,测试集取0.2)
    background_sample_mask_train = np.isin(train_labels, back_label)
    background_sample_mask_test = np.isin(test_labels, back_label)
    back_train = train_data[background_sample_mask_train]
    back_test = test_data[background_sample_mask_test]

    train_idx = int(background_sample_number*0.8)
    test_idx = background_sample_number-train_idx

    background_samples_train_mask = np.random.choice(len(back_train), size=train_idx, replace=False)
    background_samples_test_mask = np.random.choice(len(back_test), size=test_idx, replace=False)

    background_samples_test = back_test[background_samples_test_mask]
    background_samples_train = back_train[background_samples_train_mask]
    background_samples = np.vstack([background_samples_train, background_samples_test])

    ##构建解释数据集(依据数据集划分比例，训练集取0.8,测试集取0.2)
    explain_samples = []
    for Gear_30_2_label in Gear_30_2_fault_labels:
        mask_train = np.isin(train_labels, Gear_30_2_label)
        mask_test = np.isin(test_labels, Gear_30_2_label)
        every_fault_sample_train = train_data[mask_train]
        every_fault_sample_test = test_data[mask_test]

        train_idx = int(explainable_number_every_fault*0.8)
        test_idx = explainable_number_every_fault-train_idx

        explain_sample_train_mask = np.random.choice(len(every_fault_sample_train), size=train_idx, replace=False)
        explain_sample_test_mask = np.random.choice(len(every_fault_sample_test), size=test_idx, replace=False)

        fault_sample_train = every_fault_sample_train[explain_sample_train_mask]
        fault_sample_test = every_fault_sample_test[explain_sample_test_mask]

        fault_samples = np.vstack([fault_sample_train, fault_sample_test])
        explain_samples.append(fault_samples)#(5,n,6,6,1024)

    explain_samples = np.concatenate(explain_samples, axis=0)#(5*n,6,6,1024)

    shap_values = calculate_shap_values(target_model, background_samples, explain_samples)

    #保存shap值
    current_dir = os.path.dirname(__file__)
    save_dir = os.path.join(current_dir, "Seed_70")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'Gear_30_2.npz')
    # SHAP值维度含义字典（仅保留核心维度说明）
    shap_dim_meaning = {
        0: "样本数（250条解释样本：5类故障×每类50个样本）",
        1: "通道数（6个设备监测通道，是核心分析维度）",
        2: "小波系数层数（小波变换后的6层系数）",
        3: "分帧长度（原始振动数据分帧后的单帧长度，1024个数据点）",
        4: "类别数（数据集总类别数，含故障/正常样本共5类）"
    }
    np.savez(save_path,
             shap_values=shap_values,
             shap_dim_meaning = shap_dim_meaning)