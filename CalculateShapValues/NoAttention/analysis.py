import numpy as np
import os
import matplotlib.pyplot as plt

# ====================== 数据加载 ======================
path = './Seed_42'
save_dir = './Seed_42_AnalysisGraphs'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
files = os.listdir(path)

ch_shap = []
freq_shap = []
for file in files:
    data = np.load(os.path.join(path, file), allow_pickle=True)['shap_values'][:, :, 1:6, :, :]

    ch_abs_shap = np.mean(abs(data), axis=(0, 2, 3, 4))
    freq_abs_shap = np.mean(abs(data), axis=(0, 1, 3, 4))

    ch_shap.append(ch_abs_shap / ch_abs_shap.sum())
    freq_shap.append(freq_abs_shap / freq_abs_shap.sum())

ch_shap_np = np.array(ch_shap)
freq_shap_np = np.array(freq_shap)
print(ch_shap_np.shape)
print(freq_shap_np.shape)

# ====================== 期刊样式 ======================
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'axes.linewidth': 1.5,
    'axes.unicode_minus': False
})

# 工况对应名称
conditions = [
    'Bearing_20_0',
    'Bearing_30_2',
    'Gear_20_0',
    'Gear_30_2'
]

ch_names = ['ch2', 'ch3', 'ch4', 'ch6', 'ch7', 'ch8']
freq_names = ['cD2', 'cD3', 'cD4', 'cD5', 'cA5']
width = 0.65

# ====================== 绘图：每张图标注数值 ======================
for i in range(4):
    ch = ch_shap_np[i]
    freq = freq_shap_np[i]
    cond_name = conditions[i]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ========== 左图：通道 SHAP + 柱状图标值 ==========
    bars1 = ax1.bar(np.arange(len(ch)), ch, width=width,
                    color='#2E86AB', alpha=0.85, edgecolor='black')
    ax1.set_xticks(np.arange(len(ch)))
    ax1.set_xticklabels(ch_names)
    ax1.set_ylabel('Normalized SHAP Weight', fontsize=13)
    ax1.set_title(f'Channel Layer({cond_name})', fontsize=14, pad=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.set_ylim(0, max(ch) * 1.15)

    # 在柱子上方标注数值（保留3位小数）
    for bar, val in zip(bars1, ch):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # ========== 右图：频率 SHAP + 柱状图标值 ==========
    bars2 = ax2.bar(np.arange(len(freq)), freq, width=width,
                    color='#A23B72', alpha=0.85, edgecolor='black')
    ax2.set_xticks(np.arange(len(freq)))
    ax2.set_xticklabels(freq_names)
    ax2.set_ylabel('Normalized SHAP Weight', fontsize=13)
    ax2.set_title(f'Frequency Layer({cond_name})', fontsize=14, pad=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    ax2.set_ylim(0, max(freq) * 1.15)

    # 在柱子上方标注数值
    for bar, val in zip(bars2, freq):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{save_dir}\\SHAP_{cond_name}.png', dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()

print("✅ 4张带数值标注的SHAP图已保存完成！")