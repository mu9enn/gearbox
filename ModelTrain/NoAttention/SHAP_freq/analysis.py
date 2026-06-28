import numpy as np
import os

shap_dir = '.\\ShapResults'
cond_files = os.listdir(shap_dir)

# 你自定义的 每个工况 非法频率索引（直接用你的）
INVALID_IDX = {
    'Bearing_20_0': [0, 3, 4],
    'Bearing_30_2': [0, 5],
    'Gear_20_0': [0, 4],
    'Gear_30_2': [0, 4]
}

for file in cond_files:
    # 从文件名提取工况 key（自动匹配上面的字典）
    key = file.replace(".npz", "").strip()

    path = os.path.join(shap_dir, file)
    shap_values = np.load(path)['results']

    # 计算每个样本的 6 个频率贡献度 (样本数, 6)
    shap_freq = np.mean(np.abs(shap_values), axis=(1, 3, 4))

    total = len(shap_freq)
    block_count = 0
    pass_count = 0

    for sf in shap_freq:
        # ====================== 这里改成 TOP2 ======================
        # 取出贡献度最大的 2 个频率索引
        top2_idx = np.argsort(sf)[::-1][:2]  # 从大到小排序，取前2个

        # 防火墙规则：前2个里 任意一个 在非法列表 → 闭锁
        if any(idx in INVALID_IDX[key] for idx in top2_idx):
            block_count += 1
        else:
            pass_count += 1

    # 输出表格数据（直接填表5-2）
    print(f"===== {key} =====")
    print(f"总样本数：{total}")
    print(f"放行样本：{pass_count}")
    print(f"闭锁样本：{block_count}")
    print(f"放行率：{pass_count / total * 100:.2f}%")
    print(f"闭锁率：{block_count / total * 100:.2f}%")
    print()