import pandas as pd
import matplotlib.pyplot as plt

# ===================== 全局统一绘图设置（全部使用 Times New Roman 新罗马字体）=====================
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['font.family'] = 'sans-serif'
# 新增：统一边框线的抗锯齿和渲染设置，解决线条淡/不均匀问题
plt.rcParams['path.simplify'] = False
plt.rcParams['patch.edgecolor'] = 'black'
plt.rcParams['patch.linewidth'] = 0.6  # 全局统一边框线宽

# 加载数据
df = pd.read_csv('PC_Datasets(normal).csv')

# 时频特征列
feature_cols = ['mean', 'std', 'rms', 'peak', 'crest_factor', 'kurtosis',
                'skewness', 'mfe', 'ebe', 'cf', 'thd']
feature_data = df[feature_cols]

# ===================== 图1：特征分布直方图（修复边框线条）=====================
plt.figure(figsize=(16, 10))
for i, col in enumerate(feature_cols):
    plt.subplot(3, 4, i + 1)
    # 关键修复：强制指定 edgecolor、linewidth，并关闭alpha对边框的影响
    plt.hist(
        feature_data[col],
        bins=50,
        color='#1f77b4',
        edgecolor='black',    # 明确指定黑色边框
        linewidth=0.6,        # 统一边框线宽
        alpha=0.9,            # 取消透明度，避免边框被淡化
        rwidth=0.95           # 柱子间留微小空隙，让边框更清晰
    )
    plt.title(f'{col} distribution', pad=8, fontfamily='Times New Roman')
    plt.xlabel('Normalized Value', labelpad=5, fontfamily='Times New Roman')
    plt.ylabel('Frequency', labelpad=5, fontfamily='Times New Roman')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout(pad=2.0)
plt.savefig('feature_distribution.png', bbox_inches='tight')

# ===================== 图2：标准差柱状图（同步修复边框线条）=====================
std_values = feature_data.std().values

plt.figure(figsize=(14, 6))
bars = plt.bar(
    x=feature_cols,
    height=std_values,
    color='#1f77b4',
    edgecolor='black',
    linewidth=0.8,
    alpha=1.0  # 取消透明度，避免边框被淡化
)

max_std = std_values.max()
for bar, val in zip(bars, std_values):
    y_offset = max_std * 0.015
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + y_offset,
        f'{val:.4f}',
        ha='center', va='bottom',
        fontsize=16,
        fontweight='medium',
        fontfamily='Times New Roman'
    )

plt.xlabel('Time-Frequency Features', fontweight='bold', labelpad=10, fontfamily='Times New Roman')
plt.ylabel('Standard Deviation', fontweight='bold', labelpad=10, fontfamily='Times New Roman')
plt.title('feature_std_bar_normalized', pad=20, fontfamily='Times New Roman')

plt.xticks(rotation=30, ha='right', rotation_mode='anchor', fontfamily='Times New Roman')
plt.yticks(fontfamily='Times New Roman')

plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.ylim(0, max_std * 1.15)
plt.tight_layout()
plt.savefig('feature_normalized_bar.png', bbox_inches='tight')