import numpy as np
import os
import matplotlib.pyplot as plt

seeds = [42,49,56,63,70]
save_dir = './TotalAnalysisGraphs'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


ch_shap_values = []
freq_shap_values = []

for seed in seeds:
    ch_shap = []
    freq_shap = []
    folder = f"./Seed_{seed}"
    files = os.listdir(folder)
    for file in files:
        path = os.path.join(folder, file)
        data = np.load(path, allow_pickle=True)['shap_values'][:, :, :, :, :]

        ch_abs_shap = np.mean(abs(data), axis=(0, 2, 3, 4))
        freq_abs_shap = np.mean(abs(data), axis=(0, 1, 3, 4))

        ch_shap.append(ch_abs_shap / ch_abs_shap.sum())
        freq_shap.append(freq_abs_shap / freq_abs_shap.sum())

    ch_shap_values.append(ch_shap)
    freq_shap_values.append(freq_shap)

ch_shap_np = np.array(ch_shap_values)
freq_shap_np = np.array(freq_shap_values)
print(ch_shap_np.shape)
print(freq_shap_np.shape)

#三种子求平均
ch_shap_avg = np.mean(ch_shap_np, axis=0)#(4,6)
freq_shap_avg = np.mean(freq_shap_np, axis=0)#(4,5)
# print(ch_shap_avg.mean(axis=0))
# print(freq_shap_avg.mean(axis=0))
all_cond_avg_ch_shap = ch_shap_avg.mean(axis=0)
all_cond_avg_freq_shap = freq_shap_avg.mean(axis=0)

# ====================== 期刊样式 ======================
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 20,
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
freq_names = ['cA5', 'cD5', 'cD4', 'cD3', 'cD2', 'cD1']
width = 0.65

# ====================== 绘图：每张图标注数值 ======================
for i in range(4):
    ch = ch_shap_avg[i]
    freq = freq_shap_avg[i]
    cond_name = conditions[i]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ========== 左图：通道 SHAP + 柱状图标值 ==========
    bars1 = ax1.bar(np.arange(len(ch)), ch, width=width,
                    color='#2E86AB', alpha=0.85, edgecolor='black')
    ax1.set_xticks(np.arange(len(ch)))
    ax1.set_xticklabels(ch_names)
    ax1.set_ylabel('Normalized SHAP Weight', fontsize=20)
    ax1.set_title(f'Channel Layer({cond_name})', fontsize=20, pad=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.set_ylim(0, max(ch) * 1.15)

    # 在柱子上方标注数值（保留3位小数）
    for bar, val in zip(bars1, ch):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=18, fontweight='bold')

    # ========== 右图：频率 SHAP + 柱状图标值 ==========
    bars2 = ax2.bar(np.arange(len(freq)), freq, width=width,
                    color='#A23B72', alpha=0.85, edgecolor='black')
    ax2.set_xticks(np.arange(len(freq)))
    ax2.set_xticklabels(freq_names)
    ax2.set_ylabel('Normalized SHAP Weight', fontsize=20)
    ax2.set_title(f'Frequency Layer({cond_name})', fontsize=20, pad=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    ax2.set_ylim(0, max(freq) * 1.15)

    # 在柱子上方标注数值
    for bar, val in zip(bars2, freq):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=18, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{save_dir}\\SHAP_{cond_name}.png', dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
print("✅ 4张带数值标注的SHAP图已保存完成！")

fig1, ax1 = plt.subplots(figsize=(10, 6))

bars1 = ax1.bar(
    np.arange(len(all_cond_avg_ch_shap)),
    all_cond_avg_ch_shap,
    width=0.65,
    color='#2E86AB',
    alpha=0.85,
    edgecolor='black'
)

ax1.set_xticks(np.arange(len(all_cond_avg_ch_shap)))
ax1.set_xticklabels(ch_names)
ax1.set_ylabel('Normalized SHAP Weight', fontsize=20)
ax1.set_title('Channel Layer (All Conditions Average)', fontsize=20, pad=12)
ax1.grid(axis='y', linestyle='--', alpha=0.3)
ax1.set_ylim(0, max(all_cond_avg_ch_shap) * 1.15)

# 柱子上标注数值
for bar, val in zip(bars1, all_cond_avg_ch_shap):
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width()/2.,
        height + 0.005,
        f'{val:.3f}',
        ha='center',
        va='bottom',
        fontsize=18,
        fontweight='bold'
    )

plt.tight_layout()
plt.savefig(f'{save_dir}/SHAP_All_Conditions_Channel.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.close(fig1)

# ====================== 2. 频率全局平均柱状图 ======================
fig2, ax2 = plt.subplots(figsize=(10, 6))

bars2 = ax2.bar(
    np.arange(len(all_cond_avg_freq_shap)),
    all_cond_avg_freq_shap,
    width=0.65,
    color='#A23B72',
    alpha=0.85,
    edgecolor='black'
)

ax2.set_xticks(np.arange(len(all_cond_avg_freq_shap)))
ax2.set_xticklabels(freq_names)
ax2.set_ylabel('Normalized SHAP Weight', fontsize=20)
ax2.set_title('Frequency Layer (All Conditions Average)', fontsize=20, pad=12)
ax2.grid(axis='y', linestyle='--', alpha=0.3)
ax2.set_ylim(0, max(all_cond_avg_freq_shap) * 1.15)

# 柱子上标注数值
for bar, val in zip(bars2, all_cond_avg_freq_shap):
    height = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width()/2.,
        height + 0.005,
        f'{val:.3f}',
        ha='center',
        va='bottom',
        fontsize=18,
        fontweight='bold'
    )

plt.tight_layout()
plt.savefig(f'{save_dir}/SHAP_All_Conditions_Frequency.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.close(fig2)

print("✅ 两张全局平均SHAP柱状图已保存完成！")