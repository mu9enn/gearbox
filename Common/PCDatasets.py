import pandas as pd
import os
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler,MinMaxScaler

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


class CreateDataset:
    def __init__(self):
        pass

    # @staticmethod
    # def load_data(file_dir=None, skip_rows=16, use_cols=8, index=[1,2,3,5,6,7]):
    #     dataset = pd.read_csv(file_dir,
    #                        sep=r'\s+|,',
    #                        skiprows=skip_rows,
    #                        usecols=range(use_cols),
    #                        engine='python').values
    #     return dataset[:,index]

    @staticmethod
    def time_frequency_feature(datasets, ratios, window_length=1024, step=1024):
        def window_data(single_file):
            segment_data = []
            for start in range(0, single_file.shape[0] - window_length + 1, step):
                end = start + window_length
                segment_data.append(single_file[start:end])
            return np.array(segment_data)

        # 滑动窗口分割数据
        all_file_segment_data = {}
        for ids in datasets.keys():
            single_file_segment_data = window_data(datasets[ids])
            all_file_segment_data[ids] = single_file_segment_data#(20,n,512,6)


        def time_feature():
            single_label_feature = {}
            for keys, single_data in all_file_segment_data.items():
                single_window_feature = []
                if keys in bearing_label_map.values():
                    if '20_0' in keys:
                        ratio = ratios['bearing_20_0']
                    elif '30_2' in keys:
                        ratio = ratios['bearing_30_2']
                    else:
                        print("bearing权重处理错误")
                        break
                elif keys in gear_label_map.values():
                    if '20_0' in keys:
                        ratio = ratios['gear_20_0']
                    elif '30_2' in keys:
                        ratio = ratios['gear_30_2']
                    else:
                        print("gear权重处理错误")
                        break
                else:
                    print("权重提取错误")
                    break
                for seg_data in single_data:
                    mean = np.mean(seg_data, axis=0)
                    std = np.std(seg_data, axis=0)
                    rms = np.sqrt(np.mean(np.square(seg_data), axis=0)) + 1e-8
                    peak = np.max(np.abs(seg_data), axis=0)
                    crest_factor = peak / rms
                    kurtosis = stats.kurtosis(seg_data, axis=0, bias=True, fisher=False)
                    skewness = stats.skew(seg_data, axis=0, bias=True)

                    # 加权求和8个通道特征
                    mean = np.sum(mean * ratio)
                    std = np.sum(std * ratio)
                    rms = np.sum(rms * ratio)
                    peak = np.sum(peak * ratio)
                    crest_factor = np.sum(crest_factor * ratio)
                    kurtosis = np.sum(kurtosis * ratio)
                    skewness = np.sum(skewness * ratio)

                    single_window_feature.append([mean, std, rms, peak, crest_factor, kurtosis, skewness])
                single_label_feature[keys] = np.array(single_window_feature)
            return single_label_feature

        # ===================== FFT计算（核心修复） =====================
        def fft(seg_data,fs=5120):
            # 对每个通道做FFT (1024, 8) -> (1024, 8)
            fft_vals = np.fft.fft(seg_data, axis=0)
            # 频率轴
            freqs = np.fft.fftfreq(window_length, 1 / fs)
            # 正频率掩码
            positive_mask = freqs > 0
            # 幅值（归一化）
            amplitudes = np.abs(fft_vals) * 2 / window_length

            # 返回正频率和对应幅值
            return freqs[positive_mask], amplitudes[positive_mask]

        def meshing_frequency_energy(fm, freq, amplitude, ratio):
            # 找到啮合频率对应的索引
            idx1 = np.argmin(np.abs(freq - fm[0]))
            idx2 = np.argmin(np.abs(freq - fm[1]))
            # 计算能量（6通道加权）
            energy1 = np.sum(amplitude[idx1] ** 2 * ratio)
            energy2 = np.sum(amplitude[idx2] ** 2 * ratio)
            total_energy = np.sum(amplitude)
            return (energy1 + energy2)/total_energy

        def edge_band_energy(fm, fr, freq, amplitude, ratio):
            band_freqs = []
            for f in fm:
                band_freqs.extend([f - k * fr for k in range(6)])
                band_freqs.extend([f + k * fr for k in range(6)])

            total_energy = 0
            for f in band_freqs:
                idx = np.argmin(np.abs(freq - f))
                total_energy += np.sum(amplitude[idx] ** 2 * ratio)
            return total_energy

        def center_frequency(freq, amplitude, ratio):
            ei = amplitude ** 2
            f1 = freq.reshape(-1,1)
            fg = np.sum(ei * f1, axis=0) / (np.sum(ei, axis=0) + 1e-8)
            return np.sum(fg * ratio)

        def thd(fr, freq, amplitude, ratio):
            # 基波幅值
            fr_idx = np.argmin(np.abs(freq - fr))
            fund_amp = amplitude[fr_idx] + 1e-8
            # 2-10次谐波
            harm_energy = 0
            for i in range(2, 11):
                idx = np.argmin(np.abs(freq - fr * i))
                harm_energy += np.sum(amplitude[idx] ** 2 * ratio)
            # 总谐波失真
            thd_val = np.sqrt(harm_energy) / np.sum(fund_amp ** 2 * ratio)
            return thd_val

        # ===================== 频域特征主函数 =====================
        def frequency_feature():
            single_label_feature = {}
            for index,single_data in all_file_segment_data.items():
                single_window_feature = []
                if index in bearing_label_map.values():
                    if '20_0' in index:
                        ratio = ratios['bearing_20_0']
                    elif '30_2' in index:
                        ratio = ratios['bearing_30_2']
                    else:
                        print("bearing权重处理错误")
                        break
                elif index in gear_label_map.values():
                    if '20_0' in index:
                        ratio = ratios['gear_20_0']
                    elif '30_2' in index:
                        ratio = ratios['gear_30_2']
                    else:
                        print("gear权重处理错误")
                        break
                else:
                    print("权重提取错误")
                    break
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
                    mfe = meshing_frequency_energy(fm_list, freq, amp, ratio=ratio)
                    ebe = edge_band_energy(fm_list, basic_fr, freq, amp, ratio=ratio)
                    cf = center_frequency(freq, amp, ratio=ratio)
                    thd_val = thd(basic_fr, freq, amp, ratio=ratio)
                    single_window_feature.append([mfe, ebe, cf, thd_val])
                single_label_feature[index] = np.array(single_window_feature)
            return single_label_feature

        # 合并特征
        all_features = {}
        time_features = time_feature()
        frequency_features = frequency_feature()
        for key in time_features:
            all_features[key] = np.concatenate([time_features[key], frequency_features[key]], axis=-1)
        return all_features


if __name__ == "__main__":

    # Bearing_20_0_shap_path = '../CalculateShapValues/Bearing_20_0_ShapValues/Bearing_20_0_ShapValues.npz'
    # Bearing_30_2_shap_path = '../CalculateShapValues/Bearing_30_2_ShapValues/Bearing_30_2_ShapValues.npz'
    # Gear_20_0_shap_path = '../CalculateShapValues/Gear_20_0_ShapValues/Gear_20_0_ShapValues.npz'
    # Gear_30_2_shap_path = '../CalculateShapValues/Gear_30_2_ShapValues/Gear_30_2_ShapValues.npz'
    #
    # bear_20_0_shap = np.load(Bearing_20_0_shap_path)['attention']
    # bear_30_2_shap = np.load(Bearing_30_2_shap_path)['attention']
    # gear_20_0_shap = np.load(Gear_20_0_shap_path)['attention']
    # gear_30_2_shap = np.load(Gear_30_2_shap_path)['attention']
    #
    # bearing_20_0 = np.mean(np.abs(bear_20_0_shap),axis=(0,2,3,4), keepdims=False)
    # bearing_30_2 = np.mean(np.abs(bear_30_2_shap),axis=(0,2,3,4), keepdims=False)
    # gear_20_0 = np.mean(np.abs(gear_20_0_shap),axis=(0,2,3,4), keepdims=False)
    # gear_30_2 = np.mean(np.abs(gear_30_2_shap),axis=(0,2,3,4), keepdims=False)


    # weights = {'bearing_20_0':bearing_20_0/(bearing_20_0.sum()),
    #           'bearing_30_2':bearing_30_2/(bearing_30_2.sum()),
    #           'gear_20_0':gear_20_0/(gear_20_0.sum()),
    #           'gear_30_2':gear_30_2/(gear_30_2.sum())}


    # shap_dir = '../CalculateShapValues/NoAttention/Seed_49'
    # files = os.listdir(shap_dir)
    # data_list = []
    # for file in files:
    #     data = np.load(os.path.join(shap_dir, file))['shap_values']
    #     data = np.mean(abs(data), axis=(0,2,3,4))
    #     data_list.append(data)
    # weights = {
    #             'bearing_20_0': data_list[0]/data_list[0].sum(),
    #             'bearing_30_2':data_list[1]/data_list[1].sum(),
    #             'gear_20_0':data_list[2]/data_list[2].sum(),
    #             'gear_30_2':data_list[3]/data_list[3].sum()}

    weight = np.load('../ModelTrain/AttentionAndNoAttention/ChannelWeights/ChannelWeights.npz')['weights']
    weights = {
        'bearing_20_0': weight,
        'bearing_30_2': weight,
        'gear_20_0': weight,
        'gear_30_2': weight
    }
    print("通道权重：",weights)
    pure_data_dir = "../DAG/PureData/"
    bearing_label = bearing_label_map.values()
    gear_label = gear_label_map.values()
    bearing_path = [os.path.join(pure_data_dir, f"{path}.npz") for path in bearing_label]
    gear_path = [os.path.join(pure_data_dir,f"{path}.npz") for path in gear_label]

    dataset_dict = {}
    create_dataset = CreateDataset()

    for single_path in bearing_path:
        sample = np.load(single_path)#(n,6)
        label = single_path.split("/")[-1].split(".")[0]
        dataset_dict[label] = sample['pure_data']

    for single_path in gear_path:
        sample = np.load(single_path)
        label = single_path.split("/")[-1].split(".")[0]
        dataset_dict[label] = sample['pure_data']

    tensor_list = []
    key_order = []  # 保存键顺序，保证还原后顺序一致
    shapes = []  # 保存每个value的形状，用于切分

    for key, tensor in dataset_dict.items():
        key_order.append(key)
        tensor_list.append(tensor)
        shapes.append(tensor.shape[0])  # 只保存行数 n，用于后面切分

    # 拼接成 (20*n, 6)
    concatenated = np.concatenate(tensor_list, axis=0)

    # ===================== 步骤2：对 axis=0 做全局归一化 =====================
    # 常用：Min-Max 归一化到 [0,1]
    max_vals = np.max(concatenated, axis=0)
    min_vals = np.min(concatenated, axis=0)
    normalized_concat = (concatenated - min_vals) / (max_vals - min_vals + 1e-8) * 2 - 1

    # 如果你需要 Z-Score 标准化（均值0，方差1），用下面这段：
    # mean_vals = torch.mean(concatenated, dim=0)
    # std_vals = torch.std(concatenated, dim=0)
    # normalized_concat = (concatenated - mean_vals) / (std_vals + 1e-8)

    # ===================== 步骤3：按原形状切分并还原字典 =====================
    normalized_dict = {}
    start_idx = 0

    for key, length in zip(key_order, shapes):
        end_idx = start_idx + length
        normalized_tensor = normalized_concat[start_idx:end_idx]  # 取出对应片段
        normalized_dict[key] = normalized_tensor
        start_idx = end_idx

    features = create_dataset.time_frequency_feature(datasets=normalized_dict,ratios=weights, window_length=512,step=512)

    label_cols = []
    cond_cols = []
    fault_cols = []
    print(features.keys())
    for label , value in features.items():
        label_cols.extend([label] * value.shape[0])
        condition = label.split("_")[0]
        if '30_2' in label:
            if label in bearing_label_map.values():
                cond_cols.extend(['bearing_30_2'] * value.shape[0])
            elif label in gear_label_map.values():
                cond_cols.extend(['gear_30_2'] * value.shape[0])
            else:
                print('标签错误')
                break
        else:
            if label in bearing_label_map.values():
                cond_cols.extend(['bearing_20_0'] * value.shape[0])
            elif label in gear_label_map.values():
                cond_cols.extend(['gear_20_0'] * value.shape[0])
            else:
                print('标签错误')
                break
        fault_type = label.replace('_30_2', '').replace('_20_0', '')
        fault_cols.extend([fault_type] * value.shape[0])

    features_type = ['mean', 'std', 'rms', 'peak', 'crest_factor', 'kurtosis', 'skewness','mfe', 'ebe', 'cf', 'thd']
    feature_val = np.concatenate(list(features.values()),axis=0)

    #标准化
    scaler1 = StandardScaler()
    standard_val = scaler1.fit_transform(feature_val)

    #归一化
    scaler2 = MinMaxScaler()
    normal_val = scaler2.fit_transform(feature_val)

    #构建数据集
    df = pd.DataFrame(feature_val,columns=features_type)
    df['label'] = label_cols
    df['condition'] = cond_cols
    df['fault'] = fault_cols

    df_std = pd.DataFrame(standard_val,columns=features_type)
    df_std['label'] = label_cols
    df_std['condition'] = cond_cols
    df_std['fault'] = fault_cols

    df_normal = pd.DataFrame(normal_val,columns=features_type)
    df_normal['label'] = label_cols
    df_normal['condition'] = cond_cols
    df_normal['fault'] = fault_cols

    #保存
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)
    save_dir = os.path.join(parent_dir, 'PC_Datasets')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path1 = os.path.join(save_dir,'PC_Datasets(std).csv')
    save_path2 = os.path.join(save_dir,'PC_Datasets(normal).csv')
    save_path3 = os.path.join(save_dir,'PC_Datasets.csv')

    df_std.to_csv(save_path1,index=False)
    df_normal.to_csv(save_path2,index=False)
    df.to_csv(save_path3,index=False)