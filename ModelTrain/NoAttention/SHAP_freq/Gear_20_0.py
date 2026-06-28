import numpy as np
import torch
import shap
import os
import random

from GearBox.Common.NetWorkFrame import BaseModel4Channel

seed = 42
random.seed(seed)
# 设置NumPy的随机种子
np.random.seed(seed)
# 设置操作系统层面的随机数生成器（部分库会用到）
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
# 设置所有GPU的随机种子
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# 确保CUDA卷积操作的确定性
torch.backends.cudnn.deterministic = True
# 禁用cuDNN的自动优化（可能会降低速度，但保证确定性）
torch.backends.cudnn.benchmark = False

#SHAP计算函数
def calculate_shap_values(model, back_sample, ex_sample):
    explainer = shap.GradientExplainer(model, back_sample)
    shap_values = explainer.shap_values(ex_sample)
    return shap_values

#加载数据
data_path = "..\\..\\..\\wavelet_dataset\\Gear_20_0.npz"
sample = np.load(data_path, allow_pickle=True)
print(list(sample.keys()))
print(sample['label_mapping'])
data = sample['test_data'][:,[1,2,3,4],:,:]
labels = sample['test_labels']
print(data.shape)
print(labels.shape)

#定义背景和解释样本
health_label = 1
back_mask = labels == health_label
print(back_mask.shape)
all_back_mask = data[back_mask]
back_sample = all_back_mask[:50]
ex_sample = data

#加载模型，定义设备
##设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

##参数地址
parameter_path = "..\\SavedModels_4ch\\Seed_42\\Gear_20_0.pth"

##模型
model = BaseModel4Channel(class_number=5)
model.load_state_dict(torch.load(parameter_path, map_location=torch.device('cpu')))

if __name__ == '__main__':
    model = model.to(device)
    model.eval()

    if not isinstance(back_sample, torch.Tensor):
        back_sample = torch.from_numpy(back_sample).float().to(device)
    if not isinstance(ex_sample, torch.Tensor):
        ex_sample = torch.from_numpy(ex_sample).float().to(device)
    back_sample = back_sample.to(device)
    ex_sample = ex_sample.to(device)

    shap_results = calculate_shap_values(model, back_sample, ex_sample)

    #保存
    save_dir = ".\\ShapResults"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, 'Gear_20_0.npz')
    np.savez(save_path, results=shap_results)




