import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import os

from scipy import stats
from GearBox.Common.ModelTrainAndVisiable import channel_weights_visible

bearing_label_map = {
    "ball_20_0.csv": "ball_20_0",
    "ball_30_2.csv": "ball_30_2",
    "comb_20_0.csv": "comb_20_0",
    "comb_30_2.csv": "comb_30_2",
    "health_20_0.csv": "bearing_normal_20_0",
    "health_30_2.csv": "bearing_normal_30_2",
    "inner_20_0.csv": "inner_20_0",
    "inner_30_2.csv": "inner_30_2",
    "outer_20_0.csv": "outer_20_0",
    "outer_30_2.csv": "outer_30_2"
}
gear_label_map = {
    "Chipped_20_0.csv": "chipped_20_0",
    "Chipped_30_2.csv": "chipped_30_2",
    "Health_20_0.csv": "gear_normal_20_0",
    "Health_30_2.csv": "gear_normal_30_2",
    "Miss_20_0.csv": "miss_20_0",
    "Miss_30_2.csv": "miss_30_2",
    "Root_20_0.csv": "root_20_0",
    "Root_30_2.csv": "root_30_2",
    "Surface_20_0.csv": "surface_20_0",
    "Surface_30_2.csv": "surface_30_2"
}


def loss_function(fault_data, weights=None):
    weights = torch.softmax(weights, dim=1)  # [0,1] 和为1

    #加权融合
    weights = weights.view(1, 6, 1, 1)
    fused_data = (fault_data * weights).sum(dim=1)  # (20, n, 11)

    #类内标准差
    inner_var = torch.var(fused_data, dim=1, unbiased=False) #(20,11)
    inner_loss,_ = torch.max(inner_var, dim=1)#(20,)

    #类间标准差
    mean = torch.mean(fused_data, dim=1)#(20,11)
    outer_loss = torch.var(mean, dim=0, unbiased=False)#(11,)

    # 损失
    loss = torch.mean(inner_loss)/(torch.min(outer_loss) )
    return loss

def time_frequency_feature(datasets, window_length=512, step=512):
    def window_data(single_file):
        segment_data = []
        for start in range(0, single_file.shape[0] - window_length + 1, step):
            end = start + window_length
            segment_data.append(single_file[start:end])
        return np.array(segment_data)

    all_file_segment_data = {}
    for ids in datasets.keys():
        single_file_segment_data = window_data(datasets[ids])
        all_file_segment_data[ids] = single_file_segment_data

    def time_feature():
        single_label_feature = {}
        for keys, single_data in all_file_segment_data.items():
            single_window_feature = []
            for seg_data in single_data:
                mean = np.mean(seg_data, axis=0)
                std = np.std(seg_data, axis=0)
                rms = np.sqrt(np.mean(np.square(seg_data), axis=0)) + 1e-8
                peak = np.max(np.abs(seg_data), axis=0)
                crest_factor = peak / rms
                kurtosis = stats.kurtosis(seg_data, axis=0, bias=True, fisher=False)
                skewness = stats.skew(seg_data, axis=0, bias=True)
                single_window_feature.append([mean, std, rms, peak, crest_factor, kurtosis, skewness])#(n,7,6)
            single_label_feature[keys] = np.array(single_window_feature).transpose((2, 0, 1))#(20,6,n,7)
        return single_label_feature

    def fft(seg_data, fs=5120):
        fft_vals = np.fft.fft(seg_data, axis=0)
        freqs = np.fft.fftfreq(window_length, 1 / fs)
        positive_mask = freqs > 0
        amplitudes = np.abs(fft_vals) * 2 / window_length
        return freqs[positive_mask], amplitudes[positive_mask]

    def meshing_frequency_energy(fm, freq, amplitude):
        idx1 = np.argmin(np.abs(freq - fm[0]))
        idx2 = np.argmin(np.abs(freq - fm[1]))
        energy1 = amplitude[idx1, :] ** 2
        energy2 = amplitude[idx2, :] ** 2
        total_energy = np.sum(amplitude)
        return (energy1 + energy2) / total_energy

    def edge_band_energy(fm, fr, freq, amplitude):
        band_freqs = []
        for f in fm:
            band_freqs.extend([f - k * fr for k in range(6)])
            band_freqs.extend([f + k * fr for k in range(6)])
        total_energy = np.zeros(amplitude.shape[1])
        for f in band_freqs:
            idx = np.argmin(np.abs(freq - f))
            total_energy += amplitude[idx, :] ** 2
        return total_energy

    def center_frequency(freq, amplitude):
        ei = amplitude ** 2
        f1 = freq.reshape(-1, 1)
        fg = np.sum(ei * f1, axis=0) / (np.sum(ei, axis=0) + 1e-8)
        return fg

    def thd(fr, freq, amplitude):
        fr_idx = np.argmin(np.abs(freq - fr))
        fund_amp = amplitude[fr_idx, :] + 1e-8
        harm_energy = np.zeros(amplitude.shape[1])
        for j in range(2, 11):
            idx = np.argmin(np.abs(freq - fr * j))
            harm_energy += amplitude[idx] ** 2
        thd_val = np.sqrt(harm_energy) / fund_amp ** 2
        return thd_val

    def frequency_feature():
        single_label_feature = {}
        for index, single_data in all_file_segment_data.items():
            single_window_feature = []
            if '30_2' in index:
                fm_list = [313.2, 870]
                basic_fr = 30
            elif '20_0' in index:
                fm_list = [208.8, 580]
                basic_fr = 20
            else:
                raise ValueError('data error')
            for seg_data in single_data:
                freq, amp = fft(seg_data)
                mfe = meshing_frequency_energy(fm_list, freq, amp)
                ebe = edge_band_energy(fm_list, basic_fr, freq, amp)
                cf = center_frequency(freq, amp)
                thd_val = thd(basic_fr, freq, amp)
                single_window_feature.append([mfe, ebe, cf, thd_val])
            single_label_feature[index] = np.array(single_window_feature).transpose((2, 0, 1))#(20,6,n,4)
        return single_label_feature

    all_features = {}
    time_features = time_feature()
    frequency_features = frequency_feature()
    for ids in time_features:
        all_features[ids] = np.concatenate([time_features[ids], frequency_features[ids]], axis=2)
    return all_features#(20,6,n,11)


if __name__ == '__main__':
    data_config = {
        'bearing_20_0': ['ball_20_0', 'comb_20_0', 'bearing_normal_20_0', 'inner_20_0', 'outer_20_0'],
        'bearing_30_2': ['ball_30_2', 'comb_30_2', 'bearing_normal_30_2', 'inner_30_2', 'outer_30_2'],
        'gear_20_0': ['chipped_20_0', 'gear_normal_20_0', 'miss_20_0', 'root_20_0', 'surface_20_0'],
        'gear_30_2': ['chipped_30_2', 'gear_normal_30_2', 'miss_30_2', 'root_30_2', 'surface_30_2']
    }

    pure_data_dir = "../../DAG/PureData"
    bearing_label = bearing_label_map.values()
    gear_label = gear_label_map.values()
    bearing_path = [os.path.join(pure_data_dir, f"{path}.npz") for path in bearing_label]
    gear_path = [os.path.join(pure_data_dir, f"{path}.npz") for path in gear_label]

    dataset_dict = {}
    for single_path in bearing_path:
        sample = np.load(single_path)
        label = single_path.split("\\")[-1].split(".")[0]
        dataset_dict[label] = sample['pure_data']
    for single_path in gear_path:
        sample = np.load(single_path)
        label = single_path.split("\\")[-1].split(".")[0]
        dataset_dict[label] = sample['pure_data']

    feature_dict = time_frequency_feature(datasets=dataset_dict)

    #初始化
    w = nn.Parameter(torch.ones(1, 6))
    optimizer = optim.AdamW([w], lr=0.03)

    data_dict = {}
    for key, value in data_config.items():
        data_dict[key] = np.array([feature_dict[f] for f in value])

    # 标准化
    # for key, tensor_list in data_dict.items():
    #     # tensor_list 形状: (num_classes, num_samples, 6, 11)
    #     print(tensor_list.shape)
    #     all_samples = tensor_list.view(-1, 6, 11)  # 合并所有类别和样本
    #     mean = all_samples.mean(dim=0, keepdim=True).unsqueeze(2)  # (1, 6, 1, 11)
    #     std = all_samples.std(dim=0, keepdim=True).unsqueeze(2) + 1e-8
    #     tensor_list = (tensor_list - mean) / std
    #     print(tensor_list.mean(axis=2), tensor_list.std(axis=2))
    #     data_dict[key] = tensor_list

    #归一化
    # for key, value in data_dict.items():#value(5,6,2047,11)
    #     mean = np.mean(value,axis=2,keepdims=True)
    #     std = np.std(value,axis=2,keepdims=True) + 1e-8
    #     value = (value - mean)*2 / std - 1
    #     data_dict[key] = value

    loss_list = []
    # # 1. 拼接 4 个 (5,6,n,11) → (20,6,n,11)
    result = np.concatenate(list(data_dict.values()), axis=0)  # (20,6,n,11)
    #
    # # 2. 调整维度顺序 → (20,n,6,11)
    # result = result.transpose((0, 2, 1, 3))
    #
    # # 3. 展平成 (20*n, 6, 11) 方便归一化
    # result = result.reshape((-1, 6, 11))
    #
    # # 4. 全局归一化（对 axis=0 归一化，保持维度）
    # mins = result.min(axis=0, keepdims=True)  # (1,6,11)
    # maxs = result.max(axis=0, keepdims=True)
    # result = (result - mins) / (maxs - mins + 1e-8) * 2 - 1  # [-1,1]
    #
    # # 5. 恢复形状 (20, n, 6, 11)
    # result = result.reshape(20, -1, 6, 11)
    #
    # # 6. 还原回原始维度顺序 (20,6,n,11)
    # result = result.transpose((0, 2, 1, 3))

    # 7. 转 tensor
    result = torch.from_numpy(result).float()  # 最终 shape (20,6,n,11)
    for epoch in range(150):

        optimizer.zero_grad()
        loss_value = loss_function(fault_data=result, weights=w)
        loss_list.append(loss_value.detach().cpu().numpy())
        loss_value.backward()
        optimizer.step()


    loss_np = np.array(loss_list)

    save_dir = 'ChannelWeights'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'ChannelWeights.npz')

    # ========== 保存：只保存6维权重，softmax 归一化 ==========
    channel_weights = w
    channel_weights_softmax = torch.softmax(channel_weights, dim=1)
    w_np = channel_weights_softmax.detach().cpu().numpy()

    np.savez(save_path, weights=w_np)
    print("Saved weights shape:", w_np.shape)  # (4,6)
    channel_weights_visible(data=loss_np, path='SavedGraphs/ChannelWeights.png')