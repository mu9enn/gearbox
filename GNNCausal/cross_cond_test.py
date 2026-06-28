import os

import numpy as np
import torch
from torch import nn

from GearBox.Common.NetWorkFrame import GNNCausalSEU
from itertools import combinations
from torch.utils.data import TensorDataset, DataLoader

param_dir = '.\\trained_models\\GNN'
data_dir = '..\\wavelet_dataset'

class CrossTest(nn.Module):
    def __init__(self):
        super(CrossTest, self).__init__()
        # self.reg_ratio = reg_ratio
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.CrossEntropyLoss()
    @staticmethod
    def cross_param(model, loc_param_dict, cross_param_dict, transfer_list):
        model.load_state_dict(loc_param_dict)

        for name in transfer_list:
            target_param = model.state_dict()[name]
            source_param = cross_param_dict[name]
            target_param.data.copy_(source_param.data)
    @staticmethod
    def test(model, loader):
        correct_sum = 0
        model.eval()
        with torch.no_grad():
            for sample, label in loader:
                output = model(sample)
                prediction = torch.argmax(output, dim=1)
                correct_num = prediction.eq(label).sum().item()
                correct_sum += correct_num
        acc = correct_sum/len(loader.dataset)
        return acc

    def fine_tune(self, model, x_add, label_add, adjust_epoch=80):
        # model.unique_ratio.requires_grad = False
        # model.common_ratio.requires_grad = False

        for param in model.classifier.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-4)

        adj_dataset = TensorDataset(torch.from_numpy(x_add).float().to(self.device),
                                    torch.from_numpy(label_add).long().to(self.device))
        loader = DataLoader(adj_dataset, batch_size=150, shuffle=True)
        for epoch in range(adjust_epoch):
            model.train()
            for data, label in loader:
                output = model(data)
                optimizer.zero_grad()
                loss = self.criterion(output, label)
                loss.backward()
                optimizer.step()

    @staticmethod
    def load_and_split_pairs(causal_edge: torch.Tensor = None):
        causal_set = set()
        # 1. 72维特征边映射回18个物理节点，统一存储无向边(u, v) u<v去重
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

        # 2. 生成全部18节点无向两两组合
        all_phys_pairs = list(combinations(range(18), 2))
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

    @staticmethod
    def get_causal_link(causal_edge):
        causal_link = []
        for (u, v) in list(zip(causal_edge[0].tolist(), causal_edge[1].tolist())):
            i = u // 4
            j = v // 4
            if i == j:
                continue

            ch_i = i // 5
            ch_j = j // 5

            if ch_i != ch_j:
                continue

            if (i, j) not in causal_link:
                causal_link.append((i, j))

        return causal_link

    def forward(self, test_sample, test_label, finetune_x, finetune_y):

        example = os.listdir(data_dir)
        data_files = [f for f in example if '0' in f]
        model_files = [ f.replace('.npz', '.pth') for f in data_files ]
        acc = []
        for f in range(0, len(data_files)-1, 2):
            matrix_file = ['Bearing_20_0_reliable_edge.pt', 'Bearing_20_0_reliable_edge.pt',
                           'Bearing_20_0_reliable_edge.pt', 'Bearing_20_0_reliable_edge.pt']
            sample1, label1 = test_sample[f], test_label[f]
            sample2, label2 = test_sample[f+1], test_label[f+1]

            if isinstance(sample2, np.ndarray):
                sample2 = torch.from_numpy(sample2).float().to(self.device)
            if isinstance(sample1, np.ndarray):
                sample1 = torch.from_numpy(sample1).float().to(self.device)
            if isinstance(label2, np.ndarray):
                label2 = torch.from_numpy(label2).long().to(self.device)
            if isinstance(label1, np.ndarray):
                label1 = torch.from_numpy(label1).long().to(self.device)

            dataset1 = TensorDataset(sample1, label1)
            dataset2 = TensorDataset(sample2, label2)

            loader1 = DataLoader(dataset1, batch_size=128, shuffle=False)
            loader2 = DataLoader(dataset2, batch_size=128, shuffle=False)

            model_path1 = os.path.join(param_dir, model_files[f])
            model_path2 = os.path.join(param_dir, model_files[f+1])

            param1 = torch.load(model_path1, map_location='cpu')
            param2 = torch.load(model_path2, map_location='cpu')

            # transfer_param_list = ['classifier.0.weight',
            #                                    'classifier.0.bias', 'classifier.1.weight', 'classifier.1.bias',
            #                                    'classifier.1.running_mean', 'classifier.1.running_var', 'classifier.1.num_batches_tracked',
            #                                    'classifier.5.weight', 'classifier.5.bias','classifier.8.weight', 'classifier.8.bias']
            # transfer_param_list = ['classifier.1.weight', 'classifier.1.bias', 'classifier.4.weight', 'classifier.4.bias', 'classifier.7.weight', 'classifier.7.bias', 'classifier.10.weight', 'classifier.10.bias']

            causal_matrix1 = torch.load(matrix_file[f])
            causal_matrix2 = torch.load(matrix_file[f + 1])
            non_causal_pairs1, causal_pairs1 = self.load_and_split_pairs(causal_matrix1)
            non_causal_pairs2, causal_pairs2 = self.load_and_split_pairs(causal_matrix2)
            causal_link1 = self.get_causal_link(causal_matrix1)
            causal_link2 = self.get_causal_link(causal_matrix2)

            model1 = GNNCausalSEU(causal_pairs=causal_pairs1, non_causal_pairs=non_causal_pairs1, causal_link=causal_link1).to(self.device)
            model2 = GNNCausalSEU(causal_pairs=causal_pairs2, non_causal_pairs=non_causal_pairs2, causal_link=causal_link2).to(self.device)

            model1.eval()
            model2.eval()

            # self.cross_param(model1, param1, param2, transfer_param_list)
            # self.cross_param(model2, param2, param1, transfer_param_list)
            model1.load_state_dict(param1)
            model2.load_state_dict(param2)

            # model1.common_ratio.data = torch.randn(common_ratio1.shape).to(self.device)
            # model2.common_ratio.data = torch.randn(common_ratio2.shape).to(self.device)
            # model1.common_ratio.data.zero_()
            # model2.common_ratio.data.zero_()
            # model1.common_norm.weight.data.zero_()
            # model1.common_norm.bias.data.zero_()

            # self.fine_tune(model1, finetune_x[f+1], finetune_y[f+1])
            # self.fine_tune(model2, finetune_x[f], finetune_y[f])

            acc1 = self.test(model1, loader2)
            acc2 = self.test(model2, loader1)
            acc.append(acc1)
            acc.append(acc2)
        return acc

# if __name__ == '__main__':
#     tester = CrossTest(reg_ratio=np.random.randn(4,2))
#     acc = tester.forward()
#     print(acc)
