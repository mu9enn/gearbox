import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

#小波变换数据集地址
wavelet_dir = '../wavelet_dataset'
if not os.path.exists(wavelet_dir):
    os.makedirs(wavelet_dir)

#因果数据集地址
PC_dir = '../PC_Datasets/Seed_49'
if not os.path.exists(PC_dir):
    os.makedirs(PC_dir)

#shap地址
shap_dir = '../CalculateShapValues/NoAttention/Seed_49'

#定义节点函数
##频率节点函数
true_lengths = np.array([32, 32, 64, 128, 256]).reshape(1, 1, 5, 1)


def freq_dot(samples, ch_ratio):
    e_sq = samples ** 2  # (N, 6, 5, 512)
    ratio = ch_ratio.reshape(1, 6, 1, 1)
    true_mean_sq = np.sum(e_sq, axis=3, keepdims=True) / true_lengths#(n,6,5,1)

    weighted_sq = true_mean_sq * ratio

    freq_energy = np.sqrt(np.sum(weighted_sq, axis=1, keepdims=True))#(n,1,5,1)
    return freq_energy.squeeze()


def channel_dot(samples, freq_ratio):
    e_sq = samples ** 2  # (N, 6, 5, 512)
    ratio = freq_ratio.reshape(1, 1, 5, 1)

    # 【修正点1】：计算真实的无偏差能量密度
    true_mean_sq = np.sum(e_sq, axis=3, keepdims=True) / true_lengths#(n,6,5,1)

    # 【修正点2】：对真实的能量密度进行 SHAP 频带加权
    weighted_sq = true_mean_sq * ratio

    # 跨频带求和，并开根号
    channel_energy = np.sqrt(np.sum(weighted_sq, axis=2, keepdims=True))#(n,6,1,1)
    return channel_energy.squeeze()

#计算通道、频率层级贡献度，加载数据
seeds = [42,49,56,63,70]

ch_shap_values = []
freq_shap_values = []

for seed in seeds:
    ch_shap = []
    freq_shap = []
    folder = f"../CalculateShapValues/NoAttention/Seed_{seed}"
    files = os.listdir(folder)
    for file in files:
        path = os.path.join(folder, file)
        data = np.load(path, allow_pickle=True)['shap_values'][:, :, 0:-1, :, :]

        ch_abs_shap = np.mean(abs(data), axis=(0, 2, 3, 4))
        freq_abs_shap = np.mean(abs(data), axis=(0, 1, 3, 4))

        ch_shap.append(ch_abs_shap / ch_abs_shap.sum())
        freq_shap.append(freq_abs_shap / freq_abs_shap.sum())

    ch_shap_values.append(ch_shap)
    freq_shap_values.append(freq_shap)

ch_shap_np = np.array(ch_shap_values)
freq_shap_np = np.array(freq_shap_values)
print(ch_shap_np.shape)
print(freq_shap_np.shape)

#三种子求平均
ch_shap_avg = np.mean(ch_shap_np, axis=0)#(4,6)
freq_shap_avg = np.mean(freq_shap_np, axis=0)#(4,5)
# print(ch_shap_avg.mean(axis=0))
# print(freq_shap_avg.mean(axis=0))

files = os.listdir(shap_dir)
channel_ratio = {
    'Bearing_20_0':ch_shap_avg[0],
    'Bearing_30_2':ch_shap_avg[1],
    'Gear_20_0':ch_shap_avg[2],
    'Gear_30_2':ch_shap_avg[3]
}
frequency_ratio = {
    'Bearing_20_0':freq_shap_avg[0],
    'Bearing_30_2':freq_shap_avg[1],
    'Gear_20_0':freq_shap_avg[2],
    'Gear_30_2':freq_shap_avg[3]
}
data_dict = {}
label_dict = {}
for file in files:
    #加载shap值，小波数据
    wavelet_path = os.path.join(wavelet_dir, file)

    wavelet_data = np.load(wavelet_path)['all_data'][:,:,0:5,:]#(5115,6,5,512)
    labels = np.load(wavelet_path)['all_labels']

    #构建通道、频率层级系数字典
    key = file.split('.')[0]
    data_dict[key] = wavelet_data

    # 编码后标签反向映射字符标签列
    # label_map = np.load(wavelet_path, allow_pickle=True)['label_mapping']
    # index_to_label = {idx: label for label, idx in label_map.item().items()}
    # label_dict[key] = np.array([index_to_label[idx] for idx in labels])
    label_dict[key] = labels

channel_dict = {}
freq_dict = {}

#计算通道、频率节点
for file in files:
    key = file.split('.')[0]

    channel_dict[key] = channel_dot(samples=data_dict[key], freq_ratio=frequency_ratio[key])
    freq_dict[key] = freq_dot(samples=data_dict[key], ch_ratio=channel_ratio[key])

#构建因果数据集并保存
feature_cols = ['ch2', 'ch3', 'ch4', 'ch6', 'ch7', 'ch8', 'cA5', 'cD5', 'cD4', 'cD3', 'cD2']

scaler = StandardScaler()
for file in files:
    #构建DataFrame
    key = file.split('.')[0]
    data_ch = scaler.fit_transform(channel_dict[key])
    data_freq = scaler.fit_transform(freq_dict[key])
    features = np.hstack((data_ch, data_freq))
    df = pd.DataFrame(features, columns=feature_cols)
    df['label'] = label_dict[key]

    #保存
    df.to_csv(os.path.join(PC_dir,f'{key}.csv'), index=False)
