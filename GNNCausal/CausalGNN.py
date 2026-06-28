import random

import numpy as np
import os

import optuna
import torch

from itertools import combinations
from GearBox.Common.GNNTrain import GNNTrainSEU
from sklearn.preprocessing import StandardScaler
from GearBox.GNNCausal.cross_cond_test import CrossTest
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
optuna.samplers.RandomSampler(seed=42)

def load_data(file_name):
    path = os.path.join(data_dir, file_name)
    data = np.load(path,allow_pickle=True)
    train_set = data['train_set'][:,:,1:,:]#(n1,6,5,512)
    test_set = data['test_set'][:,:,1:,:]#(n2,6,5,512)
    valid_set = data['valid_set'][:,:,1:,:]#(50,6,5,512)
    finetune_set = data['cross_finetune_set'][:,:,1:,:]
    cross_test_set = data['cross_test_set'][:,:,1:,:]
    # #标准化
    # mean_orig = np.mean(train_set, axis=(0,3), keepdims=True)
    # std_orig = np.std(train_set, axis=(0,3), keepdims=True)
    # train_set = (train_set - mean_orig)/(1e-8 + std_orig)
    # test_set = (test_set - mean_orig)/(1e-8 + std_orig)
    # valid_set = (valid_set - mean_orig)/(1e-8 + std_orig)
    #
    # mean_cross = np.mean(finetune_set, axis=(0,3), keepdims=True)
    # std_cross = np.std(finetune_set, axis=(0,3), keepdims=True)
    # finetune_set = (finetune_set - mean_cross)/(std_cross + 1e-8)
    # cross_test_set = (cross_test_set - mean_cross)/(std_cross + 1e-8)

    train_label = data['train_label']
    valid_label = data['valid_label']
    test_label = data['test_label']
    finetune_label = data['cross_finetune_label']
    cross_test_label = data['cross_test_label']

    return (train_set.reshape(-1, 30, 1024), valid_set.reshape(-1, 30, 1024), test_set.reshape(-1, 30, 1024),
            finetune_set.reshape(-1, 30, 1024), cross_test_set.reshape(-1, 30, 1024),
            train_label, valid_label, test_label, finetune_label, cross_test_label)

def data_transfer(train_data, valid_data, test_data, finetune_data, cross_test_data):
    """
    输入data: (N, 30, 1024) 6通道×3低频小波层时序
    输出: (N, 120) 每个节点4维统计特征拼接，用于PC-KCI因果发现
    """
    def feature(data):
        avrig = np.mean(data, axis=-1, keepdims=True)  # (n,30,1)
        abs_avrig = np.mean(abs(data), axis=-1)  # (n,30)
        var = np.var(data, axis=-1)
        x_pp = np.max(data, axis=-1) - np.min(data, axis=-1)
        kurt = np.mean((data - avrig) ** 4, axis=-1) / var ** 2  # (n,30)
        impulse_factor = np.max(abs(data), axis=-1) / abs_avrig

        feat_4dim = np.concatenate([var, x_pp, kurt, impulse_factor], axis=-1)

        causal_sample = feat_4dim.reshape(data.shape[0], -1)
        return causal_sample
    train_sample = feature(train_data)
    valid_sample = feature(valid_data)
    test_sample = feature(test_data)
    finetune_sample = feature(finetune_data)
    cross_test_sample = feature(cross_test_data)

    # 全局标准化，消除量纲差异
    scaler = StandardScaler()
    train_sample = scaler.fit_transform(train_sample)
    valid_sample = scaler.transform(valid_sample)
    test_sample = scaler.transform(test_sample)
    finetune_sample = scaler.fit_transform(finetune_sample)
    cross_test_sample = scaler.transform(cross_test_sample)
    return train_sample.reshape(train_sample.shape[0], 30, -1), valid_sample.reshape(valid_sample.shape[0], 30, -1), test_sample.reshape(test_sample.shape[0], 30, -1),\
            finetune_sample.reshape(finetune_sample.shape[0], 30, -1), cross_test_sample.reshape(cross_test_sample.shape[0], 30, -1)

def load_and_split_pairs(causal_edge: torch.Tensor = None):
    causal_set = set()
    # 1. 120维特征边映射回18个物理节点，统一存储无向边(u, v) u<v去重
    if causal_edge is not None and causal_edge.numel() > 0:
        feat_edges = list(zip(causal_edge[0].tolist(), causal_edge[1].tolist()))
        for feat_i, feat_j in feat_edges:
            # 72维特征索引 → 18物理节点
            phys_u = feat_i // 4
            phys_v = feat_j // 4
            if phys_u == phys_v:
                continue
            # 统一为无向小值在前，避免(1,3)和(3,1)重复
            if phys_u > phys_v:
                phys_u, phys_v = phys_v, phys_u
            causal_set.add((phys_u, phys_v))

    # 2. 生成全部30节点无向两两组合
    all_phys_pairs = list(combinations(range(30), 2))
    non_causal_pairs = []

    def get_channel(node_id: int) -> int:
        """物理节点 → 所属传感器通道 0~5"""
        return node_id // 5

    for u, v in all_phys_pairs:
        ch_u = get_channel(u)
        ch_v = get_channel(v)
        # 分支1：跨通道，强制归为非因果（物理先验硬约束）
        if ch_u != ch_v:
            non_causal_pairs.append((u, v))
        # 分支2：同通道，但不在PC因果集合内，也属于非因果
        elif (u, v) not in causal_set:
            non_causal_pairs.append((u, v))
    causal_pairs = [f for f in all_phys_pairs if f not in non_causal_pairs]

    return non_causal_pairs, causal_pairs

def get_causal_link(causal_edge):
    causal_link = []
    for (u, v) in list(zip(causal_edge[0].tolist(), causal_edge[1].tolist())):
        i = u // 4
        j = v // 4
        if i == j:
            continue

        ch_i = i //5
        ch_j = j // 5

        if ch_i != ch_j:
            continue

        if (i, j) not in causal_link:
            causal_link.append((i, j))

    return causal_link

def train():
    cond_data_files = os.listdir(data_dir)
    cond_data_files = [f for f in cond_data_files if '0' in f]

    matrix_file = ['Bearing_20_0_reliable_edge.pt', 'Bearing_20_0_reliable_edge.pt', 'Bearing_20_0_reliable_edge.pt', 'Bearing_20_0_reliable_edge.pt']

    valid_data, valid_label = [], []
    test_data, test_label = [], []
    finetune_data, cross_test_data = [], []
    finetune_label, cross_test_label = [], []
    valid_acc = []

    for idx in range(0, len(cond_data_files)-1, 2):

        cond1_file = cond_data_files[idx]
        cond2_file = cond_data_files[idx+1]

        train_data1, valid_data1, test_data1, finetune_data1, cross_test_data1, train_label1, valid_label1, test_label1, finetune_label1, cross_test_label1 = load_data(cond1_file)
        train_data2, valid_data2, test_data2, finetune_data2, cross_test_data2, train_label2, valid_label2, test_label2, finetune_label2, cross_test_label2 = load_data(cond2_file)

        train_data1, valid_data1, test_data1, finetune_data1, cross_test_data1 = data_transfer(train_data1, valid_data1, test_data1, finetune_data1, cross_test_data1)
        train_data2, valid_data2, test_data2, finetune_data2, cross_test_data2 = data_transfer(train_data2, valid_data2, test_data2, finetune_data2, cross_test_data2)

        valid_data.append(valid_data1)
        valid_label.append(valid_label1)
        valid_data.append(valid_data2)
        valid_label.append(valid_label2)

        test_data.append(test_data1)
        test_label.append(test_label1)
        test_data.append(test_data2)
        test_label.append(test_label2)

        finetune_data.append(finetune_data1)
        finetune_label.append(finetune_label1)
        finetune_data.append(finetune_data2)
        finetune_label.append(finetune_label2)

        cross_test_data.append(cross_test_data1)
        cross_test_label.append(cross_test_label1)
        cross_test_data.append(cross_test_data2)
        cross_test_label.append(cross_test_label2)

        causal_matrix1 = torch.load(matrix_file[idx])
        causal_matrix2 = torch.load(matrix_file[idx+1])
        causal_link1 = get_causal_link(causal_matrix1)
        causal_link2 = get_causal_link(causal_matrix2)
        non_causal_pairs1, causal_pairs1 = load_and_split_pairs(causal_matrix1)
        non_causal_pairs2, causal_pairs2 = load_and_split_pairs(causal_matrix2)

        GNN1 = GNNTrainSEU(causal_pairs=causal_pairs1, non_causal_pairs=non_causal_pairs1, causal_link=causal_link1)
        GNN2 = GNNTrainSEU(causal_pairs=causal_pairs2, non_causal_pairs=non_causal_pairs2, causal_link=causal_link2)

        model_path1 = os.path.join(model_dir, cond1_file.replace('.npz', '.pth'))
        model_path2 = os.path.join(model_dir, cond2_file.replace('.npz', '.pth'))
        graph_path1 = os.path.join(graph_dir, cond1_file.replace('.npz', '.png'))
        graph_path2 = os.path.join(graph_dir, cond2_file.replace('.npz', '.png'))

        train_acc1, train_loss1, test_acc1, test_loss1, valid_acc1, _ = GNN1.forward(train_data1, train_label1, valid_data1, valid_label1, test_data1, test_label1, model_path1)
        train_acc2, train_loss2, test_acc2, test_loss2, valid_acc2, _ = GNN2.forward(train_data2, train_label2, valid_data2, valid_label2, test_data2, test_label2, model_path2)

        valid_acc.append(valid_acc1[-1])
        valid_acc.append(valid_acc2[-1])

        GNN1.visible(train_acc1, train_loss1, test_acc1, test_loss1, graph_path1)
        GNN2.visible(train_acc2, train_loss2, test_acc2, test_loss2, graph_path2)
    return finetune_data, finetune_label, cross_test_data, cross_test_label, valid_acc

def objective(trail):

    pos_reg1 = trail.suggest_float('pos_reg1', 1e-3, 2, log=True)
    neg_reg1 = trail.suggest_float('neg_reg1',1e-3, 2, log=True)
    pos_reg2 = trail.suggest_float('pos_reg2', 1e-3, 2, log=True)
    neg_reg2 = trail.suggest_float('neg_reg2',1e-3, 2, log=True)
    pos_reg3 = trail.suggest_float('pos_reg3', 1e-3, 2, log=True)
    neg_reg3 = trail.suggest_float('neg_reg3',1e-3, 2, log=True)
    pos_reg4 = trail.suggest_float('pos_reg4', 1e-3, 2, log=True)
    neg_reg4 = trail.suggest_float('neg_reg4',1e-3, 2, log=True)

    reg_ratio = [[pos_reg1, neg_reg1], [pos_reg2, neg_reg2], [pos_reg3, neg_reg3], [pos_reg4, neg_reg4]]
    _, _, _, _, valid_acc = train()
    return np.mean(valid_acc, dtype=np.float32) - np.std(valid_acc, dtype=np.float32)



if __name__ == '__main__':

    data_dir = '..\\wavelet_dataset'
    model_dir = '.\\trained_models\\GNN'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    graph_dir = '.\\train_curves\\GNN'
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
    dag_dir = '.\\dag_figs'
    if not os.path.exists(dag_dir):
        os.makedirs(dag_dir)

    files = os.listdir(data_dir)
    file_name = [f for f in files if '0' in f]

    # study = optuna.create_study(direction='maximize')
    # study.optimize(objective, n_trials=5)
    #
    # best_pos_reg1 = study.best_params['pos_reg1']
    # best_neg_reg1 = study.best_params['neg_reg1']
    # best_pos_reg2 = study.best_params['pos_reg2']
    # best_neg_reg2 = study.best_params['neg_reg2']
    # best_pos_reg3 = study.best_params['pos_reg3']
    # best_neg_reg3 = study.best_params['neg_reg3']
    # best_pos_reg4 = study.best_params['pos_reg4']
    # best_neg_reg4 = study.best_params['neg_reg4']
    #
    # best_reg = [[best_pos_reg1, best_neg_reg1], [best_pos_reg2, best_neg_reg2], [best_pos_reg3, best_neg_reg3], [best_pos_reg4, best_neg_reg4]]

    finetune_samples, finetune_labels, sample, label, _ = train()
    tester = CrossTest()
    acc = tester(sample, label, finetune_samples, finetune_labels)
    # print('pos_reg1:', best_pos_reg1, 'neg_reg1:', best_neg_reg1, '\npos_reg2:', best_pos_reg2, 'neg_reg2:', best_neg_reg2, '\npos_reg3:', best_pos_reg3, 'neg_reg3:', best_neg_reg3, '\npos_reg4:', best_pos_reg4, 'neg_reg4:', best_neg_reg4 )
    print('\n',acc)

    # data1 = torch.load('.\\Bearing_20_0_reliable_edge.pt')
    # causal1 = get_causal_link(data1)
    # print(causal1, len(causal1))
    #
    # data2 = torch.load('.\\Bearing_30_2_reliable_edge.pt')
    # causal2 = get_causal_link(data2)
    # print(causal2, len(causal2))

