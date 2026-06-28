import numpy as np
import scipy
import pywt
import os

from tqdm import tqdm

def window_segment(signal, window_size=1024, step=1024):
    start = 0
    end = window_size
    seg_data = []
    while end <= len(signal):
        window_data = signal[start:end]
        seg_data.append(window_data)
        start += step
        end += step
    return np.array(seg_data)#(n,1024)

def wavelet_transform(signal):
    target_length = [32, 32, 64, 128, 256, 512]
    seg_data = window_segment(signal)#(n,1024)
    single_col_data = []
    for data in seg_data:
        coefficients = pywt.wavedec(data, wavelet='db4', level=5)
        wave_coef = []
        for idx, coef in enumerate(coefficients):
            if len(coef) == target_length[idx]:
                coe = np.repeat(coef, int(512/len(coef)))
            elif len(coef) > target_length[idx]:
                diff = len(coef) - target_length[idx]
                left_remove = diff//2
                right_remove = diff-left_remove
                coe_i = coef[left_remove:-right_remove]
                coe = np.repeat(coe_i, int(512/len(coe_i)))
            else:
                diff = target_length[idx] - len(coef)
                left_pad = diff//2
                right_pad = diff-left_pad
                coe_pad = np.pad(coef, (left_pad, right_pad), 'constant')
                coe = np.repeat(coe_pad, int(512/len(coe_pad)))
            # print(len(coef), coe.shape)
            wave_coef.append(coe)#(6,512)
        single_col_data.append(wave_coef)
    return np.array(single_col_data)#(n,6,512)

if __name__ == '__main__':
    data_dir = '..\\DataSets\\CWRU\\12k Drive End Bearing Fault Data'
    save_dir = "..\\DataSets\\CWRU\\WaveletDataset"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cond_file_list = [['105', '118', '130', '144', '156'],
                     ['106', '119', '131', '145', '158'],
                     ['107', '120', '132', '146', '159'],
                     ['108', '121', '133', '147', '160']]

    fault_types = ['IR', 'Ball', 'OR centred', 'OR orthogonal', 'OR opposite']

    all_cond_wave = []
    for f, cond_file in enumerate(cond_file_list):
        all_fault_dict = {}
        with tqdm(cond_file) as p:
            for m, file in enumerate(p):
                path = os.path.join(data_dir, f'{file}.mat')
                sample = scipy.io.loadmat(path)
                DE_single_fault_sample = sample[f'X{file}_DE_time']
                FE_single_fault_sample = sample[f'X{file}_FE_time']
                BA_single_fault_sample = sample[f'X{file}_BA_time']

                # print(DE_single_fault_sample.shape)

                DE_ch_wave = wavelet_transform(DE_single_fault_sample.reshape(-1))
                FE_ch_wave = wavelet_transform(FE_single_fault_sample.reshape(-1))
                BA_ch_wave = wavelet_transform(BA_single_fault_sample.reshape(-1))

                all_fault_dict[fault_types[m]] = np.transpose([DE_ch_wave, FE_ch_wave, BA_ch_wave], (1, 0, 2, 3))

        all_cond_wave.append(np.array(all_fault_dict))

    save_path = os.path.join(save_dir, 'all_cond_wave.npz')
    np.savez(save_path,
             cond_0_1797 = all_cond_wave[0],
             cond_1_1772 = all_cond_wave[1],
             cond_2_1750 = all_cond_wave[2],
             cond_3_1730 = all_cond_wave[3]
    )
