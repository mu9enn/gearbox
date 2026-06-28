import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from sklearn.model_selection import train_test_split
from GearBox.Common.NetWorkFrame import GNNCausalCWRU, GNNCausalSEU
from tqdm import tqdm


class GNNTrainCWRU(nn.Module):
    def __init__(self,class_number=5, unique_index=None, common_index=None, common_ratio=None, reg_ratio=None):
        super(GNNTrainCWRU, self).__init__()
        self.class_number = class_number
        self.unique_index = torch.from_numpy(unique_index).long().cpu()
        self.common_index = torch.from_numpy(common_index).long().cpu()
        self.common_ratio = common_ratio
        self.reg_ratio = reg_ratio

    def forward(self, x,label,model_path):
        #超参数
        lr = 0.0001
        batch_size = 128
        epoch_sum = 300

        #优化器，设别，模型
        model = GNNCausalCWRU(self.class_number, self.unique_index, self.common_index, self.common_ratio, self.reg_ratio)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        criterion = nn.CrossEntropyLoss()

        #加载器
        x_train_orig, x_test_orig, y_train, y_test = train_test_split(x, label, test_size=0.2, random_state=42)
        mean = x_train_orig.mean(axis=(0, 2), keepdims=True)
        std = x_train_orig.std(axis=(0, 2), keepdims=True)
        x_train = (x_train_orig - mean) / std
        x_test = (x_test_orig - mean) / std
        x_train_tensor = torch.from_numpy(x_train).float().to(device)
        x_test_tensor = torch.from_numpy(x_test).float().to(device)
        y_train_tensor = torch.from_numpy(y_train).long().to(device)
        y_test_tensor = torch.from_numpy(y_test).long().to(device)
        x_train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        x_test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

        train_loader = DataLoader(x_train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(x_test_dataset, batch_size=batch_size, shuffle=False)
        #训练
        ep_loss_train = []
        ep_acc_train = []
        ep_loss_test = []
        ep_acc_test = []
        for epoch in range(epoch_sum):
            model.train()
            loss_sum_train = 0
            loss_sum_test = 0
            correct_sum_train = 0
            correct_sum_test = 0
            with tqdm(train_loader, desc=f'{epoch+1}/{epoch_sum}Training:') as t:
                for data, label in t:
                    data, label = data.to(device), label.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    prediction = output.argmax(dim=1)
                    loss = criterion(output, label) + model.regularization_loss
                    loss.backward()
                    optimizer.step()
                    loss_sum_train += loss.item()*data.size(0)
                    correct_num = prediction.eq(label).sum().item()
                    correct_sum_train += correct_num
                    t.set_postfix({'loss': loss.item(), 'acc': correct_num/data.size(0)})
            ep_loss_train.append(loss_sum_train/x_train_tensor.shape[0])
            ep_acc_train.append(correct_sum_train/x_train_tensor.shape[0])

            model.eval()
            with torch.no_grad():
                with tqdm(test_loader, desc=f'{epoch+1}/{epoch_sum}Testing:') as t:
                    for data, label in t:
                        data, label = data.to(device), label.to(device)
                        output = model(data)
                        prediction = output.argmax(dim=1)
                        loss = criterion(output, label)
                        loss_sum_test += loss.item()*data.size(0)
                        correct_num = prediction.eq(label).sum().item()
                        correct_sum_test += correct_num
                        t.set_postfix({'loss': loss.item(), 'acc': correct_num/data.size(0)})
                ep_loss_test.append(loss_sum_test/x_test_tensor.shape[0])
                ep_acc_test.append(correct_sum_test/x_test_tensor.shape[0])

            #早停逻辑
            if len(ep_loss_train) > 60:
                if abs(ep_acc_train[-1] - ep_acc_test[-1]) < 0.01 and abs(ep_acc_train[-1] - ep_acc_train[-2]) < 0.005 and abs(ep_acc_test[-1] - ep_acc_test[-3]) < 0.01:
                    break

        torch.save(model.state_dict(), model_path)
        train_acc = np.array(ep_acc_train)
        train_loss = np.array(ep_loss_train)
        test_acc = np.array(ep_acc_test)
        test_loss = np.array(ep_loss_test)
        return train_acc, train_loss, test_acc, test_loss, x_test_orig, y_test_tensor, mean, std

    @staticmethod
    def visible(train_accuracy, train_loss, test_accuracy, test_loss, save_path):
        plt.rcParams.update({
            'font.sans-serif': ['Times New Roman'],
            'axes.unicode_minus': False,
            'figure.dpi': 300,
            'savefig.dpi': 600,
            'lines.linewidth': 1.8,
            'lines.markersize': 4,
            'axes.linewidth': 1.2,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'xtick.major.width': 1.2,
            'ytick.major.width': 1.2,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'font.family': 'serif',
        })
        actual_epochs = len(train_accuracy)
        epoch_x = range(1, actual_epochs + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), tight_layout=True)

        mark_step = max(1, actual_epochs // 5)  # 确保至少显示5个标记点

        # 损失曲线
        ax1.plot(epoch_x, train_loss, label='Train Loss',
                 color='#1f77b4', marker='o', linestyle='-', markevery=mark_step)
        ax1.plot(epoch_x, test_loss, label='Test Loss',
                 color='#ff7f0e', marker='o', linestyle='-', markevery=mark_step)

        ax1.set_xlabel('Epoch', fontsize=12, labelpad=8)
        ax1.set_ylabel('Loss', fontsize=12, labelpad=8)
        ax1.set_title('Loss Curves', fontsize=14, fontweight='bold', pad=12)

        # 刻度优化
        tick_step = max(1, actual_epochs // 6)
        ax1.set_xticks(range(1, actual_epochs + 1, tick_step))
        ax1.tick_params(axis='both', labelsize=11)
        ax1.grid(True, linestyle='--', color='#e0e0e0', linewidth=0.8)
        ax1.legend(loc='upper right', fontsize=10, frameon=True, facecolor='white', edgecolor='gray',
                   handlelength=2, borderaxespad=0.5)

        # 准确率曲线
        ax2.plot(epoch_x, train_accuracy, label='Train Accuracy',
                 color='#1f77b4', marker='o', linestyle='-', markevery=mark_step)
        ax2.plot(epoch_x, test_accuracy, label='Test Accuracy',
                 color='#ff7f0e', marker='o', linestyle='-', markevery=mark_step)

        ax2.set_xlabel('Epoch', fontsize=12, labelpad=8)
        ax2.set_ylabel('Accuracy', fontsize=12, labelpad=8)
        ax2.set_title('Accuracy Curves', fontsize=14, fontweight='bold', pad=12)
        ax2.set_ylim(0, 1.0)
        ax2.set_yticks(np.arange(0, 1.01, 0.2))
        ax2.set_xticks(range(1, actual_epochs + 1, tick_step))
        ax2.tick_params(axis='both', labelsize=11)
        ax2.grid(True, linestyle='--', color='#e0e0e0', linewidth=0.8)
        ax2.legend(loc='lower right', fontsize=10, frameon=True, facecolor='white', edgecolor='gray',
                   handlelength=2, borderaxespad=0.5)

        # 保存图片
        plt.savefig(save_path, format='png', dpi=600, bbox_inches='tight', pad_inches=0.1)

        # 释放内存
        plt.close(fig)

class GNNTrainSEU(nn.Module):
    def __init__(self,class_number=5, causal_pairs=None, non_causal_pairs=None, causal_link=None):
        super(GNNTrainSEU, self).__init__()
        self.class_number = class_number
        self.causal_pairs = causal_pairs
        self.non_causal_pairs = non_causal_pairs
        self.causal_link = causal_link
        self.stage2_epoch = 80   # 第二阶段：冻结T_net+node_encoder，仅训练分类分支

    def forward(self, x_train, y_train, x_valid, y_valid, x_test, y_test, model_path):
        lr = 0.0001
        batch_size = 256

        model = GNNCausalSEU(self.class_number,  self.causal_pairs, self.non_causal_pairs, self.causal_link)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        criterion = nn.CrossEntropyLoss()

        def get_dataloader(x_np, y_np, shuffle):
            x_tensor = torch.from_numpy(x_np).float().to(device)
            y_tensor = torch.from_numpy(y_np).long().to(device)
            dataset = TensorDataset(x_tensor, y_tensor)
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        train_loader = get_dataloader(x_train, y_train, shuffle=True)
        valid_loader = get_dataloader(x_valid, y_valid, shuffle=False)
        test_loader = get_dataloader(x_test, y_test, shuffle=False)

        ep_loss_train, ep_acc_train = [], []
        ep_loss_valid, ep_acc_valid = [], []
        ep_loss_test, ep_acc_test = [], []
        best_valid_acc = 0.0

        #拆分参数
        mine_params = list(model.T_net.parameters())
        main_params = [p for n, p in model.named_parameters() if 'T_net' not in n]

        main_optim = optim.Adam(main_params, lr=lr)
        mine_optim = optim.Adam(mine_params, lr=10*lr)

        for epoch in range(self.stage2_epoch):
            model.train()
            loss_sum_train = 0.0
            correct_sum_train = 0
            tbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.stage2_epoch} Train")
            for data, label in tbar:

                # for p in main_params:
                #     p.requires_grad = False
                # mine_optim.zero_grad()
                # model(data)
                # mi_loss = model.regularization_loss
                # loss_mine = -mi_loss
                # loss_mine.backward()
                # mine_optim.step()
                # tbar.set_postfix(mi_loss=mi_loss.item())
                #
                # for p in main_params:
                #     p.requires_grad = True

                main_optim.zero_grad()
                output = model(data)
                loss = criterion(output, label)
                loss.backward()
                main_optim.step()

                bs = data.size(0)
                loss_sum_train += loss.item() * bs
                pred = output.argmax(dim=-1)
                correct_sum_train += pred.eq(label).sum().item()
                batch_acc = pred.eq(label).sum().item() / bs
                tbar.set_postfix(loss=loss.item(), acc=batch_acc)

            # 训练集指标
            train_loss = loss_sum_train / len(x_train)
            train_acc = correct_sum_train / len(x_train)
            ep_loss_train.append(train_loss)
            ep_acc_train.append(train_acc)

            # 验证集
            model.eval()
            loss_sum_valid, correct_sum_valid = 0.0, 0
            with torch.no_grad():
                for data, label in valid_loader:
                    out = model(data)
                    ls = criterion(out, label)
                    loss_sum_valid += ls.item() * data.size(0)
                    pred = out.argmax(-1)
                    correct_sum_valid += pred.eq(label).sum().item()
            valid_loss = loss_sum_valid / len(x_valid)
            valid_acc = correct_sum_valid / len(x_valid)
            ep_loss_valid.append(valid_loss)
            ep_acc_valid.append(valid_acc)

            # 测试集
            loss_sum_test, correct_sum_test = 0.0, 0
            with torch.no_grad():
                for data, label in test_loader:
                    out = model(data)
                    ls = criterion(out, label)
                    loss_sum_test += ls.item() * data.size(0)
                    pred = out.argmax(-1)
                    correct_sum_test += pred.eq(label).sum().item()
            test_loss = loss_sum_test / len(x_test)
            test_acc = correct_sum_test / len(x_test)
            ep_loss_test.append(test_loss)
            ep_acc_test.append(test_acc)

            # 仅第二阶段保存最优权重
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                torch.save(model.state_dict(), model_path)
            # 仅第二阶段执行早停
            if len(ep_acc_valid) > 40 and ep_acc_valid[-1] <= max(ep_acc_valid[-10:]):
                print(f"第二阶段早停，最优val_acc={best_valid_acc:.4f}")
                break

        return (
            np.array(ep_acc_train), np.array(ep_loss_train),
            np.array(ep_acc_test), np.array(ep_loss_test),
            np.array(ep_acc_valid), np.array(ep_loss_valid)
        )

    @staticmethod
    def visible(train_accuracy, train_loss, test_accuracy, test_loss, save_path):
        plt.rcParams.update({
            'font.sans-serif': ['Times New Roman'],
            'axes.unicode_minus': False,
            'figure.dpi': 300,
            'savefig.dpi': 600,
            'lines.linewidth': 1.8,
            'lines.markersize': 4,
            'axes.linewidth': 1.2,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'xtick.major.width': 1.2,
            'ytick.major.width': 1.2,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'font.family': 'serif',
        })
        actual_epochs = len(train_accuracy)
        epoch_x = range(1, actual_epochs + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), tight_layout=True)

        mark_step = max(1, actual_epochs // 5)  # 确保至少显示5个标记点

        # 损失曲线
        ax1.plot(epoch_x, train_loss, label='Train Loss',
                 color='#1f77b4', marker='o', linestyle='-', markevery=mark_step)
        ax1.plot(epoch_x, test_loss, label='Test Loss',
                 color='#ff7f0e', marker='o', linestyle='-', markevery=mark_step)

        ax1.set_xlabel('Epoch', fontsize=12, labelpad=8)
        ax1.set_ylabel('Loss', fontsize=12, labelpad=8)
        ax1.set_title('Loss Curves', fontsize=14, fontweight='bold', pad=12)

        # 刻度优化
        tick_step = max(1, actual_epochs // 6)
        ax1.set_xticks(range(1, actual_epochs + 1, tick_step))
        ax1.tick_params(axis='both', labelsize=11)
        ax1.grid(True, linestyle='--', color='#e0e0e0', linewidth=0.8)
        ax1.legend(loc='upper right', fontsize=10, frameon=True, facecolor='white', edgecolor='gray',
                   handlelength=2, borderaxespad=0.5)

        # 准确率曲线
        ax2.plot(epoch_x, train_accuracy, label='Train Accuracy',
                 color='#1f77b4', marker='o', linestyle='-', markevery=mark_step)
        ax2.plot(epoch_x, test_accuracy, label='Test Accuracy',
                 color='#ff7f0e', marker='o', linestyle='-', markevery=mark_step)

        ax2.set_xlabel('Epoch', fontsize=12, labelpad=8)
        ax2.set_ylabel('Accuracy', fontsize=12, labelpad=8)
        ax2.set_title('Accuracy Curves', fontsize=14, fontweight='bold', pad=12)
        ax2.set_ylim(0, 1.0)
        ax2.set_yticks(np.arange(0, 1.01, 0.2))
        ax2.set_xticks(range(1, actual_epochs + 1, tick_step))
        ax2.tick_params(axis='both', labelsize=11)
        ax2.grid(True, linestyle='--', color='#e0e0e0', linewidth=0.8)
        ax2.legend(loc='lower right', fontsize=10, frameon=True, facecolor='white', edgecolor='gray',
                   handlelength=2, borderaxespad=0.5)

        # 保存图片
        plt.savefig(save_path, format='png', dpi=600, bbox_inches='tight', pad_inches=0.1)

        # 释放内存
        plt.close(fig)
