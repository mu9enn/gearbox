import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# ===================== 1. 全局 SCI 样式设置 =====================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "axes.unicode_minus": False
})

# ===================== 2. 你的数据（替换成真实标签即可） =====================
# 5 种故障类型
class_names = ['Normal', 'Ball', 'Inner', 'Outer', 'Gear']

# 替换成你的 real_label 和 prediction_label
y_true = np.array([0,0,0,1,1,2,2,3,3,4,4,4])
y_pred = np.array([0,0,1,1,1,2,2,3,4,4,4,4])

# ===================== 3. 绘制：归一化混淆矩阵 =====================
cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)  # 归一化

fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)

# 坐标与标签
ax.set_xticks(np.arange(len(class_names)))
ax.set_yticks(np.arange(len(class_names)))
ax.set_xticklabels(class_names, fontsize=11, weight='bold')
ax.set_yticklabels(class_names, fontsize=11, weight='bold')

# 数字显示
thresh = cm_norm.max() / 2.0
for i in range(len(class_names)):
    for j in range(len(class_names)):
        ax.text(j, i, f'{cm_norm[i, j]:.2f}',
                ha='center', va='center',
                color='white' if cm_norm[i, j] > thresh else 'black',
                fontsize=11, weight='bold')

# 轴标签
ax.set_xlabel('Predicted Label', fontsize=13, weight='bold')
ax.set_ylabel('True Label', fontsize=13, weight='bold')
ax.set_title('Confusion Matrix', fontsize=14, weight='bold')

# 颜色条
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.ax.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig("confusion_matrix_SCI.png", bbox_inches='tight')
plt.close()

# ===================== 4. 绘制：分类指标条形图 =====================
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
metrics = ['precision', 'recall', 'f1-score']
data = {m: [report[cls][m] for cls in class_names] for m in metrics}

x = np.arange(len(class_names))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 5))

# 三个指标条形
ax.bar(x - width, data['precision'], width, label='Precision',
       color='#1F77B4', edgecolor='black', linewidth=0.8)
ax.bar(x,         data['recall'],    width, label='Recall',
       color='#FF7F0E', edgecolor='black', linewidth=0.8)
ax.bar(x + width, data['f1-score'],  width, label='F1-score',
       color='#2CA02C', edgecolor='black', linewidth=0.8)

# 数字标签
def add_labels(rects):
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f'{h:.2f}', xy=(rect.get_x()+rect.get_width()/2, h),
                    xytext=(0,2), textcoords="offset points",
                    ha='center', fontsize=10, weight='bold')

for bar in [ax.containers[0], ax.containers[1], ax.containers[2]]:
    add_labels(bar)

# 样式
ax.set_ylabel('Score', fontsize=12, weight='bold')
ax.set_title('Classification Metrics', fontsize=14, weight='bold')
ax.set_xticks(x)
ax.set_xticklabels(class_names, fontsize=11, weight='bold')
ax.legend(frameon=True, edgecolor='black', fontsize=11)
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig("classification_metrics.png", bbox_inches='tight')
plt.close()
