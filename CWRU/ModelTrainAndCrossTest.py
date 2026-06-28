import numpy as np
import torch
import torch.nn as nn
import os
import itertools
import optuna
import random

from sklearn.preprocessing import LabelEncoder
from GearBox.Common.GNNTrain import GNNTrainCWRU
from GearBox.GNNCausal.CausalGNN import causal, pc_mi, split_matrix, select_sample
from torch.utils.data import TensorDataset, DataLoader
from GearBox.Common.NetWorkFrame import GNNCausalCWRU
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
optuna.samplers.RandomSampler(seed=42)
class CrossTest(nn.Module):
    def __init__(self, reg_ratio):
        super(CrossTest, self).__init__()
        self.reg_ratio = reg_ratio
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
        model.unique_ratio.requires_grad = False
        model.common_ratio.requires_grad = False

        for param in model.classifier.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-4)

        if isinstance(x_add, np.ndarray):
            x_add = torch.from_numpy(x_add).float().to(self.device)
        if isinstance(label_add, np.ndarray):
            label_add = torch.from_numpy(label_add).long().to(self.device)

        adj_dataset = TensorDataset(x_add, label_add)
        loader = DataLoader(adj_dataset, batch_size=50, shuffle=True)
        for epoch in range(adjust_epoch):
            model.train()
            for data, label in loader:
                output = model(data)
                optimizer.zero_grad()
                loss = self.criterion(output, label)
                loss.backward()
                optimizer.step()

    def forward(self, test_sample, test_label, add_x, add_y, mean, std, param_path):

        x1, y1 = test_sample[0], test_label[0]
        x2, y2 = test_sample[1], test_label[1]

        sample1, label1 = (x2 - mean[1]) / (std[1] + 1e-8), y2
        sample2, label2 = (x1 - mean[0]) / (std[0] + 1e-8), y1

        finetune_x1, finetune_y1 = add_x[0], add_y[0]
        finetune_x2, finetune_y2 = add_x[1], add_y[1]

        finetune_sample1, finetune_label1 = (finetune_x2 - mean[1]) / (std[1] + 1e-8), finetune_y2
        finetune_sample2, finetune_label2 = (finetune_x1 - mean[0]) / (std[0] + 1e-8), finetune_y1

        if isinstance(sample1, np.ndarray):
            sample1 = torch.from_numpy(sample1).float().to(self.device)
        if isinstance(sample2, np.ndarray):
            sample2 = torch.from_numpy(sample2).float().to(self.device)
        if isinstance(label1, np.ndarray):
            label1 = torch.from_numpy(label1).long().to(self.device)
        if isinstance(label2, np.ndarray):
            label2 = torch.from_numpy(label2).long().to(self.device)
        dataset1 = TensorDataset(sample1, label1)
        dataset2 = TensorDataset(sample2, label2)
        loader1 = DataLoader(dataset1, batch_size=25, shuffle=False)
        loader2 = DataLoader(dataset2, batch_size=25, shuffle=False)

        model_path1 = param_path[0]
        model_path2 = param_path[1]

        param1 = torch.load(model_path1, map_location='cpu')
        param2 = torch.load(model_path2, map_location='cpu')
        common_ratio1 = param1['common_ratio']
        common_ratio2 = param2['common_ratio']
        unique_index1 = param1['unique_index']
        unique_index2 = param2['unique_index']
        common_index = param1['common_index']

        # transfer_param_list = ['common_ratio', 'common_norm.weight', 'common_norm.bias','classifier.0.weight',
        #                                    'classifier.0.bias', 'classifier.1.weight', 'classifier.1.bias',
        #                                    'classifier.1.running_mean', 'classifier.1.running_var', 'classifier.1.num_batches_tracked',
        #                                    'classifier.5.weight', 'classifier.5.bias','classifier.8.weight', 'classifier.8.bias']
        # transfer_param_list = ['common_ratio', 'common_norm.weight', 'common_norm.bias']

        model1 = GNNCausalCWRU(unique_index=unique_index1, common_index=common_index, common_ratio=common_ratio1, reg_ratio=self.reg_ratio[0])
        model2 = GNNCausalCWRU(unique_index=unique_index2, common_index=common_index, common_ratio=common_ratio2, reg_ratio=self.reg_ratio[1])

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

        # self.fine_tune(model1, finetune_sample1, finetune_label1)
        # self.fine_tune(model2, finetune_sample2, finetune_label2)

        acc1 = self.test(model1, loader1)
        acc2 = self.test(model2, loader2)
        return acc1, acc2

def load_data():
    all_cond_data = np.load('..\\DataSets\\CWRU\\WaveletDataset\\all_cond_wave.npz',allow_pickle=True)
    cond_data_list = []
    for condition in conditions:
        single_cond_data = all_cond_data[condition][()]
        cond_data_list.append(single_cond_data)
    return cond_data_list

def struct_data(data_dict):
    fault_types = ['IR', 'Ball', 'OR centred', 'OR orthogonal', 'OR opposite']
    label = []
    sample = []
    for fault_type in fault_types:
        data = data_dict[fault_type]
        sample.append(data)
        label.extend([fault_type] * data.shape[0])
    mid_sample = np.concatenate(sample, axis=0)#(n,3,6,512)
    sample = mid_sample.reshape(-1, 18, 512)#(n,18,512)
    encoder = LabelEncoder()
    label = encoder.fit_transform(label)#(n,)
    poly_sample = np.mean(sample ** 2, axis=-1)#(n,18)
    return sample, label, poly_sample

def train_test(reg_coef, alpha):
    #四工况取两工况组合
    results = list(itertools.combinations(range(4), 2))

    #全工况数据
    all_cond_sample = load_data()

    #六种组合跨域测试准确率
    acc = []
    for comb in results:
        cond1_sample = all_cond_sample[comb[0]]
        cond2_sample = all_cond_sample[comb[1]]

        sample1, label1, poly_sample1 = struct_data(cond1_sample)
        sample2, label2, poly_sample2 = struct_data(cond2_sample)

        #划分微调数据
        selected_sample1, selected_label1, remaining_sample1, remaining_label1 = select_sample(sample1, label1, num_per_class=15)
        selected_sample2, selected_label2, remaining_sample2, remaining_label2 = select_sample(sample2, label2, num_per_class=15)

        #因果发现
        _, matrix1 = causal(poly_sample1, alpha[comb[0]])
        _, matrix2 = causal(poly_sample2, alpha[comb[1]])

        #提取工况专属邻接矩阵、跨工况不变邻接矩阵
        unique_index1, unique_index2, common_index = split_matrix(matrix1, matrix2)

        #计算互信息初始化权重
        common_weight1 = pc_mi(poly_sample1, common_index)
        common_weight2 = pc_mi(poly_sample2, common_index)

        #训练
        ##定义模型保存路径
        model_path1 = os.path.join(model_dir, f'{conditions[comb[0]]}.pth')
        model_path2 = os.path.join(model_dir, f'{conditions[comb[1]]}.pth')

        trainer1 = GNNTrainCWRU(class_number=5, unique_index=unique_index1, common_index=common_index, common_ratio=common_weight1, reg_ratio=reg_coef[comb[0]])
        trainer2 = GNNTrainCWRU(class_number=5, unique_index=unique_index2, common_index=common_index, common_ratio=common_weight2, reg_ratio=reg_coef[comb[1]])

        train_acc1, train_loss1, test_acc1, test_loss1, x_test1, y_test1, mean1, std1 = trainer1(remaining_sample1, remaining_label1, model_path1)
        train_acc2, train_loss2, test_acc2, test_loss2, x_test2, y_test2, mean2, std2 = trainer2(remaining_sample2, remaining_label2, model_path2)

        ##训练曲线保存路径
        graph_path1 = os.path.join(graph_dir, f'{conditions[comb[0]]}.png')
        graph_path2 = os.path.join(graph_dir, f'{conditions[comb[1]]}.png')

        trainer1.visible(train_accuracy=train_acc1, train_loss=train_loss1, test_accuracy=test_acc1, test_loss=test_loss1, save_path=graph_path1)
        trainer2.visible(train_accuracy=train_acc2, train_loss=train_loss2, test_accuracy=test_acc2, test_loss=test_loss2, save_path=graph_path2)

        #跨域测试
        x_test = [x_test1, x_test2]
        y_test = [y_test1, y_test2]
        fine_tune_x = [selected_sample1, selected_sample2]
        fine_tune_y = [selected_label1, selected_label2]
        mean = [mean1, mean2]
        std = [std1, std2]
        tester = CrossTest(reg_ratio=[reg_coef[comb[0]], reg_coef[comb[1]]])
        ac1, ac2 = tester(x_test, y_test, fine_tune_x, fine_tune_y, mean, std, [model_path1, model_path2])
        acc.append(ac1)
        acc.append(ac2)

    return acc

def objective(trail):
    unique_reg1 = trail.suggest_float('unique_reg1', 1e-1, 1, log=True)
    isolate_reg1 = trail.suggest_float('isolate_reg1', 0.05, 1, log=True)
    unique_reg2 = trail.suggest_float('unique_reg2', 1e-1, 1, log=True)
    isolate_reg2 = trail.suggest_float('isolate_reg2', 0.05, 1, log=True)
    unique_reg3 = trail.suggest_float('unique_reg3', 1e-1, 1, log=True)
    isolate_reg3 = trail.suggest_float('isolate_reg3', 0.05, 1, log=True)
    unique_reg4 = trail.suggest_float('unique_reg4', 1e-1, 1, log=True)
    isolate_reg4 = trail.suggest_float('isolate_reg4', 0.05, 1, log=True)

    alpha1 = trail.suggest_float('alpha1', 1e-7, 1e-3, log=True)
    alpha2 = trail.suggest_float('alpha2', 1e-7, 1e-3, log=True)
    alpha3 = trail.suggest_float('alpha3', 1e-7, 1e-3, log=True)
    alpha4 = trail.suggest_float('alpha4', 1e-7, 1e-3, log=True)

    reg_ratio = [[unique_reg1, isolate_reg1], [unique_reg2, isolate_reg2], [unique_reg3, isolate_reg3], [unique_reg4, isolate_reg4]]
    alpha = [alpha1, alpha2, alpha3, alpha4]

    accuracy = train_test(reg_ratio, alpha)

    return np.mean(accuracy, dtype=np.float32) - np.std(accuracy, dtype=np.float32)

if __name__ == '__main__':

    model_dir = '.\\trained_models\\GNN'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    graph_dir = '.\\train_curves\\GNN'
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
    dag_dir = '.\\dag_figs'
    if not os.path.exists(dag_dir):
        os.makedirs(dag_dir)

    conditions = ['cond_0_1797', 'cond_1_1772', 'cond_2_1750', 'cond_3_1730']

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    best_unique_reg1 = study.best_params['unique_reg1']
    best_isolate_reg1 = study.best_params['isolate_reg1']
    best_unique_reg2 = study.best_params['unique_reg2']
    best_isolate_reg2 = study.best_params['isolate_reg2']
    best_unique_reg3 = study.best_params['unique_reg3']
    best_isolate_reg3 = study.best_params['isolate_reg3']
    best_unique_reg4 = study.best_params['unique_reg4']
    best_isolate_reg4 = study.best_params['isolate_reg4']

    best_alpha1 = study.best_params['alpha1']
    best_alpha2 = study.best_params['alpha2']
    best_alpha3 = study.best_params['alpha3']
    best_alpha4 = study.best_params['alpha4']

    best_reg = [[best_unique_reg1, best_isolate_reg1], [best_unique_reg2, best_isolate_reg2],
                [best_unique_reg3, best_isolate_reg3], [best_unique_reg4, best_isolate_reg4]]
    best_alpha = [best_alpha1, best_alpha2, best_alpha3, best_alpha4]
    all_cond_comb_acc = train_test(best_reg, best_alpha)
    print('unique_reg1:', best_unique_reg1, 'isolate_reg1:', best_isolate_reg1, '\nunique_reg2:', best_unique_reg2,
          'isolate_reg2:', best_isolate_reg2, '\nunique_reg3:', best_unique_reg3, 'isolate_reg3:', best_isolate_reg3,
          '\nunique_reg4:', best_unique_reg4, 'isolate_reg4:', best_isolate_reg4)
    print('\nalpha1', best_alpha1, '\nalpha2', best_alpha2, '\nalpha3', best_alpha3, '\nalpha4', best_alpha4)
    print('\n', all_cond_comb_acc)
