import os
import pandas as pd
import numpy as np
from GearBox.Common.PCCausal import WaveletDAG




#因果数据路径
pc_data_dir = '../PC_Datasets/Seed_49'

#定义保存路径
orig_save_dir = './PC_DAG/Seed_49/original/Significance_0.05'
filtered_save_dir = './PC_DAG/Seed_49/filtered/Significance_0.05'
if not os.path.exists(orig_save_dir):
    os.makedirs(orig_save_dir)
if not os.path.exists(filtered_save_dir):
    os.makedirs(filtered_save_dir)

#shap地址
seeds = [42,49,56,63,70]


#加载数据
data_dict = {}
ch_shap_dict = {}
freq_shap_dict = {}
files = os.listdir(pc_data_dir)
for file in files:
    ch_shap_val = []
    freq_shap_val = []
    key = file.split('.')[0]
    data_dict[key] = pd.read_csv(os.path.join(pc_data_dir,file))
    for seed in seeds:
        shap_data_dir = f'../CalculateShapValues/NoAttention/Seed_{seed}'
        shap_values = np.load(os.path.join(shap_data_dir,f'{key}.npz'))['shap_values'][:,:,0:-1,:,:]#(250,6,5,512,5)
        ch_vals = np.mean(abs(shap_values), axis=(0,2,3,4))#(6,)
        freq_vals = np.mean(abs(shap_values), axis=(0,1,3,4))#(5,
        ch_shap_val.append(ch_vals)
        freq_shap_val.append(freq_vals)

    ch_shap_np = np.array(ch_shap_val)
    freq_shap_np = np.array(freq_shap_val)

    ch_vals = ch_shap_np.mean(axis=0)
    freq_vals = freq_shap_np.mean(axis=0)
    ch_shap_dict[key] = {
        'ch2':ch_vals[0],
        'ch3':ch_vals[1],
        'ch4':ch_vals[2],
        'ch6':ch_vals[3],
        'ch7':ch_vals[4],
        'ch8':ch_vals[5]
    }
    freq_shap_dict[key] = {
        'cA5':freq_vals[0],
        'cD5':freq_vals[1],
        'cD4':freq_vals[2],
        'cD3':freq_vals[3],
        'cD2':freq_vals[4]
    }

#因果推断并保存
for key in data_dict.keys():
    wave_dag = WaveletDAG(dataset=data_dict[key])
    orig_ch_path = os.path.join(orig_save_dir,f'{key}_ch.png')
    orig_freq_path = os.path.join(orig_save_dir,f'{key}_freq.png')

    filtered_ch_path = os.path.join(filtered_save_dir, f'{key}_ch.png')
    filtered_freq_path = os.path.join(filtered_save_dir, f'{key}_freq.png')

    #因果推断
    ch_dag, freq_dag = wave_dag.pc_dag()
    wave_dag.visualize_both_dags(ch_dag, freq_dag, orig_ch_path, orig_freq_path,filtered_ch_path, filtered_freq_path,ch_shap_dict[key], freq_shap_dict[key])
