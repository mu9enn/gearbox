import numpy as np
import pandas as pd
import os
import pywt

from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from scipy.interpolate import interp1d
from scipy.signal import hilbert, butter, filtfilt



class DataSets:

    def __init__(self):
        pass

    @staticmethod
    def load_data(skip_rows=16, use_cols=8, file_path = None):
        data = pd.read_csv(file_path,
                           skiprows=skip_rows,
                           sep=r'\s+|,',
                           engine='python',
                           usecols=range(use_cols))
        return data.values

    @staticmethod
    def window_data(signal, segment_length=1024, step=1024):
        segmented_data = []
        for start in range(0, len(signal) - segment_length + 1, step):
            end = start + segment_length
            window_data = signal[start:end]
            segmented_data.append(window_data)
        return np.array(segmented_data)#(n, 1024, 6)

    @staticmethod
    def envelope_signal(signal, fs=256, lowcut=40, highcut=100, order=5):

        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, signal)

        analytic_signal = hilbert(filtered_signal)

        envelope = np.abs(analytic_signal)
        envelope = envelope - np.mean(envelope)  # 去除直流分量
        return envelope

    @staticmethod
    def wavelet_function(data, wavelet_type='db4', layers=5):
        wave_coef = []
        for single_data in data:
            single_ch = []
            for i in range(6):
                signal = single_data[:, i]
                wt_coef = pywt.swt(data=signal, wavelet=wavelet_type, level=layers)
                coef = []
                for idx, (u, v) in enumerate(wt_coef):
                    if idx == 0:
                        coef.append(u)
                        coef.append(v)
                    else:
                        coef.append(v)
                single_ch.append(coef)#(6,6,1024)
            wave_coef.append(single_ch)
        return np.array(wave_coef)#(n, 6, 6, 1024)

    @staticmethod
    def angle_resample(vib_signal, fs=5120, rpm=None, angle_fs=256):
        """
        将时域振动信号重采样为角域信号
        Args:
            vib_signal: 振动信号（一维数组）
            fs: 振动信号采样频率 (Hz)
            rpm: 瞬时转速序列 (RPM)，与vib_signal等长
            angle_fs: 角域采样频率 (samples/rev)，即每转采样点数
        Returns:
            angle: 角度轴 (degree)
            vib_angle: 角域重采样后的振动信号
        """
        t = np.arange(len(vib_signal)) / fs

        # 转速从RPM转为Hz（转/秒）
        freq_hz = rpm / 60

        # 积分得到累积角度（单位：度）
        angle = np.cumsum(freq_hz) / fs * 360

        # 生成等角度间隔的目标角度轴
        angle_min, angle_max = angle[0], angle[-1]
        angle_new = np.linspace(angle_min, angle_max, int(len(angle) * angle_fs / fs * freq_hz[0]))

        # 插值得到角域信号
        f_interp = interp1d(angle, vib_signal, kind='cubic', fill_value='extrapolate')
        vib_angle = f_interp(angle_new)

        return angle_new, vib_angle

    #小波变换
    def wavelet_transform(self, set_dict, wavelet_path=None, rpm=None, encoder=None):
        single_cond_wt_train = []
        single_cond_wt_test = []
        single_cond_wt_val = []
        single_cond_cross_finetune = []
        single_cond_cross_test = []

        train_label = []
        test_label = []
        val_label = []
        cross_finetune_label = []
        cross_test_label = []

        encoder.fit(list(set_dict.keys()))
        for key in set_dict.keys():
            single_fault_signal = set_dict[key][:,[1,2,3,5,6,7]]
            # print(single_fault_signal.shape)
            length = len(single_fault_signal)

            full_rpm_template = np.full(shape=length, fill_value=rpm, dtype=np.int32)

            # 基础划分不变：train 0~80%，val 80~90%，test 90~100%
            th1 = int(length * 0.8)
            th2 = th1 + int(length * 0.1)
            train_set = single_fault_signal[:th1]
            valid_set = single_fault_signal[th1:th2]
            test_set = single_fault_signal[th2:]
            train_angle_set = []
            valid_angle_set = []
            test_angle_set = []

            rpm_train = full_rpm_template[:th1]
            rpm_valid = full_rpm_template[th1:th2]
            rpm_test = full_rpm_template[th2:]

            for i in range(6):
                _, train_angle = self.angle_resample(train_set[:, i], rpm=rpm_train)
                _, valid_angle = self.angle_resample(valid_set[:, i], rpm=rpm_valid)
                _, test_angle = self.angle_resample(test_set[:, i], rpm=rpm_test)

                # train_angle = self.envelope_signal(train_angle)
                # valid_angle = self.envelope_signal(valid_angle)
                # test_angle = self.envelope_signal(test_angle)

                train_angle_set.append(train_angle)
                valid_angle_set.append(valid_angle)
                test_angle_set.append(test_angle)
            train_set = np.array(train_angle_set).T
            valid_set = np.array(valid_angle_set).T
            test_set = np.array(test_angle_set).T

            # 改动：全局随机截取总长10%的连续时序作为微调集，允许和train重叠
            clip_len = int(length * 0.1)
            max_start = length - clip_len
            rand_start = np.random.randint(0, max_start)
            cross_finetune_set = single_fault_signal[rand_start: rand_start + clip_len]
            rpm_fine = full_rpm_template[rand_start: rand_start + clip_len]
            # 微调片段以外全部作为cross_test
            cross_test_part1 = single_fault_signal[:rand_start]
            cross_test_part2 = single_fault_signal[rand_start + clip_len:]
            cross_test_set = np.concatenate([cross_test_part1, cross_test_part2], axis=0)
            rpm_cross_test = np.concatenate([full_rpm_template[:rand_start], full_rpm_template[rand_start + clip_len:]])

            fine_angle_set = []
            cross_angle_set = []
            for i in range(6):
                _, fine_angle = self.angle_resample(cross_finetune_set[:, i], rpm=rpm_fine)
                _, cross_angle = self.angle_resample(cross_test_set[:, i], rpm=rpm_cross_test)

#                 fine_angle = self.envelope_signal(fine_angle)
#                 cross_angle = self.envelope_signal(cross_angle)

                fine_angle_set.append(fine_angle)
                cross_angle_set.append(cross_angle)
            cross_finetune_set = np.array(fine_angle_set).T
            cross_test_set = np.array(cross_angle_set).T

            ##滑动窗口采样
            seg_train_set = self.window_data(signal=train_set, step=768)  # (n1, 1024, 6)
            seg_valid_set = self.window_data(signal=valid_set, step=1024)  # (50, 1024, 6)
            seg_test_set = self.window_data(signal=test_set, step=768)  # (n2, 1024, 6)
            seg_cross_finetune_set = self.window_data(signal=cross_finetune_set, step=1024)  # (n3, 1024, 6)
            seg_cross_test_set = self.window_data(signal=cross_test_set, step=768)

            ##小波变换
            train_coef = self.wavelet_function(data=seg_train_set)  # (n1,6,6,1024)
            valid_coef = self.wavelet_function(data=seg_valid_set)  # (50,6,6,1024)
            test_coef = self.wavelet_function(data=seg_test_set)  # (n2,6,6,1024)
            cross_finetune_coef = self.wavelet_function(data=seg_cross_finetune_set)
            cross_test_coef = self.wavelet_function(data=seg_cross_test_set)

            single_cond_wt_train.append(train_coef)
            single_cond_wt_test.append(test_coef)
            single_cond_wt_val.append(valid_coef)
            single_cond_cross_finetune.append(cross_finetune_coef)
            single_cond_cross_test.append(cross_test_coef)

            train_label.extend([key] * train_coef.shape[0])
            test_label.extend([key] * test_coef.shape[0])
            val_label.extend([key] * valid_coef.shape[0])
            cross_finetune_label.extend([key] * cross_finetune_coef.shape[0])
            cross_test_label.extend([key] * cross_test_coef.shape[0])

        train_label = encoder.transform(train_label)
        val_label = encoder.transform(val_label)
        test_label = encoder.transform(test_label)
        cross_finetune_label = encoder.transform(cross_finetune_label)
        cross_test_label = encoder.transform(cross_test_label)

        np.savez(wavelet_path,
                 train_set=np.reshape(single_cond_wt_train, (-1, 6, 6, 1024)),
                 valid_set=np.reshape(single_cond_wt_val, (-1, 6, 6, 1024)),
                 test_set=np.reshape(single_cond_wt_test, (-1, 6, 6, 1024)),
                 cross_finetune_set=np.reshape(single_cond_cross_finetune, (-1, 6, 6, 1024)),
                 cross_test_set=np.reshape(single_cond_cross_test, (-1, 6, 6, 1024)),
                 train_label=np.array(train_label),
                 valid_label=np.array(val_label),
                 test_label=np.array(test_label),
                 cross_finetune_label=np.array(cross_finetune_label),
                 cross_test_label=np.array(cross_test_label),
                 label_mapping=list(zip(set_dict.keys(), range(5))))

if __name__ == "__main__":
    BearingLabel_20_0 = [
        "ball_20_0",
        "comb_20_0",
        "health_20_0",
        "inner_20_0",
        "outer_20_0"
    ]
    BearingLabel_30_2 = [
        "ball_30_2",
        "comb_30_2",
        "health_30_2",
        "inner_30_2",
        "outer_30_2"
    ]
    GearLabel_20_0 = [
        "chipped_20_0",
        "Health_20_0",
        "miss_20_0",
        "root_20_0",
        "surface_20_0"
    ]
    GearLabel_30_2 = [
        "chipped_30_2",
        "Health_30_2",
        "miss_30_2",
        "root_30_2",
        "surface_30_2"
    ]

    encoder = LabelEncoder()

    data_set = DataSets()
    folder_dir = '../DataSets/SEU'
    bearing_20_0_file = [f'{p}.csv' for p in BearingLabel_20_0]
    bearing_30_2_file = [f'{p}.csv' for p in BearingLabel_30_2]
    gear_20_0_file = [f'{p}.csv' for p in GearLabel_20_0]
    gear_30_2_file = [f'{p}.csv' for p in GearLabel_30_2]

    bearing_20_0_dict = {}
    bearing_30_2_dict = {}
    gear_20_0_dict = {}
    gear_30_2_dict = {}

    with tqdm(bearing_20_0_file,total=len(bearing_20_0_file),desc='开始读数据') as qbar:
        for name in qbar:
            path = os.path.join(f'{folder_dir}/bearing_set', name)
            dict_key = name.split('.')[0]
            bearing_20_0_dict[dict_key] = data_set.load_data(file_path=path)
            qbar.set_postfix({"状态": f'文件{name}读取成功，形状为{bearing_20_0_dict[dict_key].shape}'})


    with tqdm(bearing_30_2_file,total=len(bearing_30_2_file),desc='开始读数据') as qbar:
        for name in qbar:
            path = os.path.join(f'{folder_dir}/bearing_set', name)
            dict_key = name.split('.')[0]
            bearing_30_2_dict[dict_key] = data_set.load_data(file_path=path)
            # bearing_30_2_dict[dict_key] = np.load(path)['pure_data']
            qbar.set_postfix({"状态": f'文件{name}读取成功，形状为{bearing_30_2_dict[dict_key].shape}'})

    with tqdm(gear_20_0_file,total=len(gear_20_0_file),desc='开始读数据') as qbar:
        for name in qbar:
            path = os.path.join(f'{folder_dir}/gear_set', name)
            dict_key = name.split('.')[0]
            gear_20_0_dict[dict_key] = data_set.load_data(file_path=path)
            # gear_20_0_dict[dict_key] = np.load(path)['pure_data']
            qbar.set_postfix({"状态": f'文件{name}读取成功，形状为{gear_20_0_dict[dict_key].shape}'})

    with tqdm(gear_30_2_file, total=len(gear_30_2_file), desc='开始读数据') as qbar:
        for name in qbar:
            path = os.path.join(f'{folder_dir}/gear_set', name)
            dict_key = name.split('.')[0]
            gear_30_2_dict[dict_key] = data_set.load_data(file_path=path)
            # gear_30_2_dict[dict_key] = np.load(path)['pure_data']
            qbar.set_postfix({"状态": f'文件{name}读取成功，形状为{gear_30_2_dict[dict_key].shape}'})


    bearing_20_0_path = '../wavelet_dataset/Bearing_20_0.npz'
    bearing_30_2_path = '../wavelet_dataset/Bearing_30_2.npz'
    gear_20_0_path = '../wavelet_dataset/Gear_20_0.npz'
    gear_30_2_path = '../wavelet_dataset/Gear_30_2.npz'

    data_set.wavelet_transform(bearing_20_0_dict,wavelet_path=bearing_20_0_path, rpm=1200, encoder=encoder)
    data_set.wavelet_transform(bearing_30_2_dict,wavelet_path=bearing_30_2_path, rpm=1800, encoder=encoder)
    data_set.wavelet_transform(gear_20_0_dict, wavelet_path=gear_20_0_path, rpm=1200, encoder=encoder)
    data_set.wavelet_transform(gear_30_2_dict, wavelet_path=gear_30_2_path, rpm=1800, encoder=encoder)




