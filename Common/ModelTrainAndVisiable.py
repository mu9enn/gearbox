import numpy as np
import torch
import matplotlib.pyplot as plt
import os

from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

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

class ModelTrain(nn.Module):
    def __init__(self, model):
        super(ModelTrain, self).__init__()
        self.model = model

    @staticmethod
    def core_channel_selection(data, index):
        """筛选指定通道的数据"""
        return data[:, index, :, :]

    @staticmethod
    def param_init(model):
        for name, module in model.named_modules():
            # 1. 卷积层初始化（包括SCAttention2D内的Conv2d）
            if isinstance(module, nn.Conv2d):
                # Kaiming初始化（fan_in模式，适配ReLU激活）
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                # 卷积层偏置初始化为0（如果有）
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

            # 2. 批归一化层初始化
            elif isinstance(module, nn.BatchNorm2d):
                # BN层权重(gamma)初始化为1，偏置(beta)初始化为0
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

            # 3. 全连接层初始化（仅分类头的Linear）
            elif isinstance(module, nn.Linear):
                # Kaiming初始化（适配ReLU激活）
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                # Linear层偏置初始化为0（如果有）
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def train_model(self, data, save_path):
        """单模型训练主函数"""
        x_train, x_test, y_train, y_test = data['train_data'], data['test_data'], data['train_labels'], data['test_labels']

        # ndarray转为张量并筛选关键通道数据
        if isinstance(x_train, np.ndarray):
            x_train = torch.from_numpy(x_train).float()
            x_test = torch.from_numpy(x_test).float()
            train_label = torch.from_numpy(y_train).long()
            test_label = torch.from_numpy(y_test).long()
        else:
            x_train = x_train.float()
            x_test = x_test.float()
            train_label = y_train.long()
            test_label = y_test.long()

        # 超参数定义
        learning_rate = 0.00015
        batch_size = 256
        epochs = 50
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 模型初始化
        self.model.to(device)
        self.param_init(self.model)

        # 优化器和损失函数
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # 数据加载器（修复：变量名不覆盖）
        train_dataset = TensorDataset(x_train, train_label)
        test_dataset = TensorDataset(x_test, test_label)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 训练记录
        train_acc_list = []
        test_acc_list = []
        train_loss_list = []
        test_loss_list = []

        for epoch in range(epochs):
            train_correct_num = 0
            test_correct_num = 0
            train_loss_num = 0
            test_loss_num = 0

            # 训练阶段
            self.model.train()
            with tqdm(train_loader, total=len(train_loader)) as pbar:
                pbar.set_description(f'Train Epoch {epoch + 1}/{epochs}')
                for batch_data, batch_label in pbar:
                    batch_data = batch_data.to(device)
                    batch_label = batch_label.to(device)

                    optimizer.zero_grad()
                    output = self.model(batch_data)
                    loss_value = criterion(output, batch_label)
                    train_loss_num += loss_value.item() * batch_data.size(0)

                    loss_value.backward()
                    optimizer.step()

                    # 计算准确率
                    prediction = output.argmax(dim=1)
                    correct_num = prediction.eq(batch_label).sum().item()
                    train_correct_num += correct_num

                    pbar.set_postfix({'Accuracy': correct_num / batch_data.size(0), 'Loss': loss_value.item()})

            # 记录训练指标
            train_acc = train_correct_num / len(train_dataset)
            train_loss = train_loss_num / len(train_dataset)
            train_acc_list.append(train_acc)
            train_loss_list.append(train_loss)

            # 测试阶段
            self.model.eval()
            with torch.no_grad():
                with tqdm(test_loader, total=len(test_loader)) as pbar:
                    pbar.set_description(f'Test Epoch {epoch + 1}/{epochs}')
                    for batch_data, batch_label in pbar:
                        batch_data = batch_data.to(device)
                        batch_label = batch_label.to(device)

                        output = self.model(batch_data)
                        loss_value = criterion(output, batch_label)
                        test_loss_num += loss_value.item() * batch_data.size(0)

                        prediction = output.argmax(dim=1)
                        correct_num = prediction.eq(batch_label).sum().item()
                        test_correct_num += correct_num

                        pbar.set_postfix({'Accuracy': correct_num / batch_data.size(0), 'Loss': loss_value.item()})

            # 记录测试指标
            test_acc = test_correct_num / len(test_dataset)
            test_loss = test_loss_num / len(test_dataset)
            test_acc_list.append(test_acc)
            test_loss_list.append(test_loss)

            # 早停逻辑
            if len(train_acc_list) > 1:
                if (abs(train_acc_list[-1] - train_acc_list[-2]) < 0.005) and (abs(train_acc_list[-1] - test_acc_list[-1]) < 0.015):
                    # 保存模型
                    torch.save(self.model.state_dict(), save_path)

                    # 转为numpy数组返回
                    train_acc_np = np.array(train_acc_list)
                    test_acc_np = np.array(test_acc_list)
                    train_loss_np = np.array(train_loss_list)
                    test_loss_np = np.array(test_loss_list)
                    return train_acc_np, test_acc_np, train_loss_np, test_loss_np

        #保存模型
        torch.save(self.model.state_dict(), save_path)

        # 转为numpy数组返回
        train_acc_np = np.array(train_acc_list)
        test_acc_np = np.array(test_acc_list)
        train_loss_np = np.array(train_loss_list)
        test_loss_np = np.array(test_loss_list)

        return train_acc_np, test_acc_np, train_loss_np, test_loss_np

    @staticmethod
    def visible(train_accuracy, test_accuracy, train_loss, test_loss, save_path):
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



class ParallelTraining(nn.Module):
    def __init__(self, models):
        super(ParallelTraining, self).__init__()
        self.model_attention = models[0]
        self.model_no_attention = models[1]

    @staticmethod
    def core_channel_selection(data, index):
        return data[:, index, :, :]

    @staticmethod
    def param_init(model):
        for name, module in model.named_modules():
            # 1. 卷积层初始化（包括SCAttention2D内的Conv2d）
            if isinstance(module, nn.Conv2d):
                # Kaiming初始化（fan_in模式，适配ReLU激活）
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                # 卷积层偏置初始化为0（如果有）
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

            # 2. 批归一化层初始化
            elif isinstance(module, nn.BatchNorm2d):
                # BN层权重(gamma)初始化为1，偏置(beta)初始化为0
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

            # 3. 全连接层初始化（仅分类头的Linear）
            elif isinstance(module, nn.Linear):
                # Kaiming初始化（适配ReLU激活）
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                # Linear层偏置初始化为0（如果有）
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def model_train(self, datasets, attention_save_path, no_attention_save_path):
        """双模型并行训练"""
        try:
            x_train, x_test, y_train, y_test = datasets['train_data'], datasets['test_data'], datasets['train_labels'], datasets['test_labels']
            x_train = self.core_channel_selection(x_train, range(6))
            x_test = self.core_channel_selection(x_test, range(6))

            # 数据类型转换
            if isinstance(x_train, np.ndarray):
                x_train = torch.from_numpy(x_train).float()
                x_test = torch.from_numpy(x_test).float()
                train_label = torch.from_numpy(y_train).long()
                test_label = torch.from_numpy(y_test).long()
            else:
                x_train = x_train.float()
                x_test = x_test.float()
                train_label = y_train.long()
                test_label = y_test.long()

            # 超参数
            learning_rate = 0.00015
            batch_size = 256
            epochs = 50
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # 模型初始化
            self.model_attention.to(device)
            self.model_no_attention.to(device)
            self.param_init(self.model_attention)
            self.param_init(self.model_no_attention)

            # 修复：分别为两个模型定义优化器
            optimizer_attention = torch.optim.Adam(self.model_attention.parameters(), lr=learning_rate)
            optimizer_no_attention = torch.optim.Adam(self.model_no_attention.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()

            # 数据加载器
            train_dataset = TensorDataset(x_train, train_label)
            test_dataset = TensorDataset(x_test, test_label)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # 训练记录
            loss_train_att = []
            loss_test_att = []
            loss_train_no_att = []
            loss_test_no_att = []
            acc_train_att = []
            acc_test_att = []
            acc_train_no_att = []
            acc_test_no_att = []

            SC_channel_weights = []
            SC_spatial_weights = []

            for epoch in range(epochs):
                # 训练阶段
                self.model_attention.train()
                self.model_no_attention.train()
                loss_sum_train_att = 0
                loss_sum_train_no_att = 0
                correct_train_att = 0
                correct_train_no_att = 0

                with tqdm(train_loader, total=len(train_loader),desc=f"Train Epoch:{epoch+1}/{epochs}") as pbar:
                    for batch_data, batch_label in pbar:
                        batch_data = batch_data.to(device)
                        batch_label = batch_label.to(device)

                        # 带注意力模型前向传播
                        output_att = self.model_attention(batch_data)
                        loss_att = criterion(output_att, batch_label)
                        loss_sum_train_att += loss_att.item() * batch_data.size(0)

                        # 不带注意力模型前向传播
                        output_no_att = self.model_no_attention(batch_data)
                        loss_no_att = criterion(output_no_att, batch_label)
                        loss_sum_train_no_att += loss_no_att.item() * batch_data.size(0)

                        # 反向传播
                        optimizer_attention.zero_grad()
                        optimizer_no_attention.zero_grad()
                        loss_att.backward()
                        loss_no_att.backward()
                        optimizer_attention.step()
                        optimizer_no_attention.step()

                        # 计算准确率
                        pred_att = output_att.argmax(dim=1)
                        pred_no_att = output_no_att.argmax(dim=1)
                        correct_sum_attention = pred_att.eq(batch_label).sum().item()
                        correct_sum_no_att = pred_no_att.eq(batch_label).sum().item()
                        correct_train_att += correct_sum_attention
                        correct_train_no_att += correct_sum_no_att

                        # 更新进度条
                        pbar.set_postfix({
                            'AttAcc': correct_sum_attention / batch_data.size(0),
                            'AttLoss': loss_att.item(),
                            'NoAttAcc': correct_sum_no_att / batch_data.size(0),
                            'NoAttLoss': loss_no_att.item()
                        })

                # 记录训练指标
                train_acc_att = correct_train_att / len(train_dataset)
                train_acc_no_att = correct_train_no_att / len(train_dataset)
                train_loss_att = loss_sum_train_att / len(train_dataset)
                train_loss_no_att = loss_sum_train_no_att / len(train_dataset)
                acc_train_att.append(train_acc_att)
                acc_train_no_att.append(train_acc_no_att)
                loss_train_att.append(train_loss_att)
                loss_train_no_att.append(train_loss_no_att)

                # 测试阶段
                self.model_attention.eval()
                self.model_no_attention.eval()
                loss_sum_test_att = 0
                loss_sum_test_no_att = 0
                correct_test_att = 0
                correct_test_no_att = 0

                with torch.no_grad():
                    with tqdm(test_loader, total=len(test_loader),desc=f"Test Epoch:{epoch+1}/{epochs}") as pbar:
                        for batch_data, batch_label in pbar:
                            batch_data = batch_data.to(device)
                            batch_label = batch_label.to(device)

                            output_att = self.model_attention(batch_data)
                            loss_att = criterion(output_att, batch_label)
                            loss_sum_test_att += loss_att.item() * batch_data.size(0)

                            output_no_att = self.model_no_attention(batch_data)
                            loss_no_att = criterion(output_no_att, batch_label)
                            loss_sum_test_no_att += loss_no_att.item() * batch_data.size(0)

                            pred_att = output_att.argmax(dim=1)
                            pred_no_att = output_no_att.argmax(dim=1)
                            correct_sum_attention = pred_att.eq(batch_label).sum().item()
                            correct_sum_no_att = pred_no_att.eq(batch_label).sum().item()
                            correct_test_att += correct_sum_attention
                            correct_test_no_att += correct_sum_no_att

                            pbar.set_postfix({
                                'AttAcc': correct_sum_attention / batch_data.size(0),
                                'AttLoss': loss_att.item(),
                                'NoAttAcc': correct_sum_no_att / batch_data.size(0),
                                'NoAttLoss': loss_no_att.item()
                            })

                # 记录测试指标
                test_acc_att = correct_test_att / len(test_dataset)
                test_acc_no_att = correct_test_no_att / len(test_dataset)
                test_loss_att = loss_sum_test_att / len(test_dataset)
                test_loss_no_att = loss_sum_test_no_att / len(test_dataset)
                acc_test_att.append(test_acc_att)
                acc_test_no_att.append(test_acc_no_att)
                loss_test_att.append(test_loss_att)
                loss_test_no_att.append(test_loss_no_att)



                #保存权重
                weights = self.model_attention.get_attention_weights()
                SC_channel_weights.append(weights['channel_weight'])
                SC_spatial_weights.append(weights['spatial_weight'])

                # 早停逻辑
                if (abs(acc_test_att[-1] - acc_train_att[-1]) <= 0.01) and (
                        abs(acc_test_no_att[-1] - acc_train_no_att[-1]) <= 0.01) and (
                        abs(acc_test_no_att[-1] - acc_train_att[-1]) <= 0.01):
                    if (len(acc_train_att) > 1) and (abs(acc_test_no_att[-1] - acc_test_no_att[-2]) <= 0.01) and (
                            abs(acc_train_no_att[-1] - acc_train_no_att[-2]) <= 0.01) and (
                            abs(acc_test_att[-1] - acc_test_att[-2]) <= 0.01) and (
                            abs(acc_train_att[-1] - acc_train_att[-2]) <= 0.01):
                        # 保存模型
                        torch.save(self.model_attention.state_dict(), attention_save_path)
                        torch.save(self.model_no_attention.state_dict(), no_attention_save_path)

                        # 转为numpy数组返回
                        return (
                            np.array(acc_train_att), np.array(acc_test_att),
                            np.array(loss_train_att), np.array(loss_test_att),
                            np.array(acc_train_no_att), np.array(acc_test_no_att),
                            np.array(loss_train_no_att), np.array(loss_test_no_att),
                            np.array(SC_channel_weights), np.array(SC_spatial_weights)
                        )

            #保存模型
            torch.save(self.model_attention.state_dict(), attention_save_path)
            torch.save(self.model_no_attention.state_dict(), no_attention_save_path)


            # 转为numpy数组返回
            return (
                np.array(acc_train_att), np.array(acc_test_att),
                np.array(loss_train_att), np.array(loss_test_att),
                np.array(acc_train_no_att), np.array(acc_test_no_att),
                np.array(loss_train_no_att), np.array(loss_test_no_att),
                np.array(SC_channel_weights), np.array(SC_spatial_weights)
            )

        except Exception as e:
            print(f"并行训练出错: {e}")
            raise

    @staticmethod
    def visible(loss_epoch_train_attention,
                loss_epoch_test_attention,
                loss_epoch_train_no_attention,
                loss_epoch_test_no_attention,
                acc_epoch_train_attention,
                acc_epoch_train_no_attention,
                acc_epoch_test_attention,
                acc_epoch_test_no_attention,
                SavePath_graphs):
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

        # 提取工况 Bearing_20_0
        file_name = os.path.basename(SavePath_graphs)
        working_condition = os.path.splitext(file_name)[0]

        actual_epochs = len(loss_epoch_test_attention)
        epoch_x = range(1, actual_epochs + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), tight_layout=True)

        # 整张图大标题（你要的效果）
        fig.suptitle(f'Train Curves ({working_condition})', fontsize=16, fontweight='bold', y=1.02)

        mark_step = max(1, actual_epochs // 5)

        # 损失曲线
        ax1.plot(epoch_x, loss_epoch_train_attention, label='Train (With Attention)',
                 color='#1f77b4', marker='o', linestyle='-', markevery=mark_step)
        ax1.plot(epoch_x, loss_epoch_train_no_attention, label='Train (No Attention)',
                 color='#ff7f0e', marker='o', linestyle='-', markevery=mark_step)
        ax1.plot(epoch_x, loss_epoch_test_attention, label='Test (With Attention)',
                 color='#1f77b4', marker='s', linestyle='--', markevery=mark_step)
        ax1.plot(epoch_x, loss_epoch_test_no_attention, label='Test (No Attention)',
                 color='#ff7f0e', marker='s', linestyle='--', markevery=mark_step)

        ax1.set_xlabel('Epoch', fontsize=12, labelpad=8)
        ax1.set_ylabel('Loss', fontsize=12, labelpad=8)
        ax1.set_title('Loss Curves', fontsize=14, fontweight='bold', pad=12)

        tick_step = max(1, actual_epochs // 6)
        ax1.set_xticks(range(1, actual_epochs + 1, tick_step))
        ax1.tick_params(axis='both', labelsize=11)
        ax1.grid(True, linestyle='--', color='#e0e0e0', linewidth=0.8)
        ax1.legend(loc='upper right', fontsize=10, frameon=True, facecolor='white', edgecolor='gray',
                   handlelength=2, borderaxespad=0.5)

        # 准确率曲线
        ax2.plot(epoch_x, acc_epoch_train_attention, label='Train (With Attention)',
                 color='#1f77b4', marker='o', linestyle='-', markevery=mark_step)
        ax2.plot(epoch_x, acc_epoch_train_no_attention, label='Train (No Attention)',
                 color='#ff7f0e', marker='o', linestyle='-', markevery=mark_step)
        ax2.plot(epoch_x, acc_epoch_test_attention, label='Test (With Attention)',
                 color='#1f77b4', marker='s', linestyle='--', markevery=mark_step)
        ax2.plot(epoch_x, acc_epoch_test_no_attention, label='Test (No Attention)',
                 color='#ff7f0e', marker='s', linestyle='--', markevery=mark_step)

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

        plt.savefig(SavePath_graphs, format='png', dpi=600, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)


def channel_weights_visible(data, path):
    x = np.arange(len(data))  # x轴：样本点/迭代次数

    # ======================
    # 4条曲线配色（顶刊标准）
    # ======================
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    labels = ['Bearing_20_0', 'Bearing_30_2', 'Gear_20_0', 'Gear_30_2']

    # ======================
    # 绘图
    # ======================
    plt.figure(figsize=(6, 4))  # 顶刊常用尺寸

    # for i in range(4):
    #     plt.plot(x, data[:, i], color=colors[i], label=labels[i])
    plt.plot(x,data, color=colors[0])
    # ======================
    # 标签与图例
    # ======================
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    # plt.legend(frameon=True, edgecolor='black', fancybox=False)
    plt.grid(alpha=0.2, linestyle='--')  # 轻网格，期刊允许

    # ======================
    # 紧凑布局 + 保存高清图
    # ======================
    plt.tight_layout()
    plt.savefig(path, dpi=600, bbox_inches='tight')
    plt.close()



