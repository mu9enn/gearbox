import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
from pgmpy.models import DiscreteBayesianNetwork

from GearBox.Common.PCCausal import GearBoxCausalAnalyzer

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'axes.unicode_minus': False,
    'axes.linewidth': 1.2,
    'legend.frameon': True,
    'legend.framealpha': 1.0,
    'legend.edgecolor': 'black',
    'legend.fontsize': 10,
})

# ====================== 读取数据 & 学习原始DAG ======================
dataset = pd.read_csv('../PC_Datasets/PC_Datasets(std).csv')
dag_analyzer = GearBoxCausalAnalyzer(dataset=dataset)
learned_dag = dag_analyzer.pc_dag()

fault_col = 'fault'
operate_col = 'condition'
feature_col = ['mean', 'std', 'rms', 'peak', 'crest_factor', 'kurtosis', 'skewness',
               'mfe', 'ebe', 'cf', 'thd']
outcome_col = 'label'

# ====================== 节点名称映射 & 配色 ======================
node_label_map = {
    fault_col: 'Fault Type',
    operate_col: 'Operating Condition',
    outcome_col: 'Fault Inference',
    'mean': 'Mean', 'std': 'Std', 'rms': 'RMS', 'peak': 'Peak',
    'crest_factor': 'Crest Factor', 'kurtosis': 'Kurtosis', 'skewness': 'Skewness',
    'mfe': 'MFE', 'ebe': 'EBE', 'cf': 'CF', 'thd': 'THD'
}

COLOR_MAP = {'root': '#1F77B4', 'feature': '#2CA02C', 'outcome': '#D62728'}
NODE_SIZE = 3800
NODE_BORDER_WIDTH = 1.5
save_dir = 'PC_DAG'
os.makedirs(save_dir, exist_ok=True)

# ====================== 布局函数：所有节点强制保留！不丢任何一个 ======================
def three_layer_layout_optimized(G):
    pos = {}
    layer_y = {'root': 0.8, 'feature': 0.4, 'outcome': 0.0}

    # 根节点
    pos[fault_col] = (0.25, layer_y['root'])
    pos[operate_col] = (0.75, layer_y['root'])

    # 所有特征全部保留！强制画满11个！
    all_features = feature_col
    num_features = len(all_features)
    x_min, x_max = 0.05, 0.95
    x_step = (x_max - x_min) / (num_features - 1) if num_features > 1 else 0
    for i, node in enumerate(all_features):
        pos[node] = (x_min + i * x_step, layer_y['feature'])

    # 结果节点
    pos[outcome_col] = (0.5, layer_y['outcome'])
    return pos

# ====================== 统一颜色 ======================
def get_unified_colors(G):
    colors = []
    all_nodes = [fault_col, operate_col] + feature_col + [outcome_col]
    for node in all_nodes:
        if node == outcome_col:
            colors.append(COLOR_MAP['outcome'])
        elif node in [fault_col, operate_col]:
            colors.append(COLOR_MAP['root'])
        else:
            colors.append(COLOR_MAP['feature'])
    return colors

# ====================== 绘图函数：强制画所有节点！统一圆圈 ======================
def draw_and_save(G, title, save_path, rad1=0.02, rad2=0.05):
    plt.figure(figsize=(20, 12))
    pos = three_layer_layout_optimized(G)

    # 强制画所有节点！保证每个都有圆圈！
    all_nodes = [fault_col, operate_col] + feature_col + [outcome_col]
    current_colors = get_unified_colors(G)

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=all_nodes,  # 强制画满所有节点
        node_size=NODE_SIZE,
        node_color=current_colors,
        alpha=0.8,
        edgecolors='black',
        linewidths=NODE_BORDER_WIDTH
    )

    # 画边
    feat_edges = []
    other_edges = []
    for u, v in G.edges():
        if u in feature_col and v in feature_col:
            feat_edges.append((u, v))
        else:
            other_edges.append((u, v))

    nx.draw(G, pos, edgelist=other_edges, with_labels=False, arrowstyle="->", arrowsize=32,
            edge_color="#333", width=2.0, alpha=0.9, connectionstyle=f'arc3,rad={rad1}')
    nx.draw(G, pos, edgelist=feat_edges, with_labels=False, arrowstyle="->", arrowsize=32,
            edge_color="#7F8C8D", width=1.8, alpha=0.9, connectionstyle=f'arc3,rad={rad2}')

    # 画标签
    nx.draw_networkx_labels(G, pos, labels=node_label_map, font_size=12, font_family="Times New Roman", font_weight="bold")

    plt.title(title, fontsize=16, fontweight='bold', pad=30)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=600, format='png', facecolor='white')
    plt.close()
    print(f"已保存：{save_path}")

# ====================== 1. 原始DAG ======================
draw_and_save(
    learned_dag,
    'Original Causal DAG',
    f'{save_dir}/GearBox_PC_DAG_original.png'
)

# ====================== 2. 过滤DAG ======================
valid_edges = []
for u, v in learned_dag.edges():
    if (u in [fault_col, operate_col] and v in feature_col) or (u in feature_col and v == outcome_col):
        valid_edges.append((u, v))
    else:
        print(f"过滤无效边：({u},{v})")

filtered_dag = DiscreteBayesianNetwork(valid_edges)
draw_and_save(
    filtered_dag,
    'Filtered Causal DAG',
    f'{save_dir}/GearBox_PC_DAG_filtered.png'
)

# ====================== 3. 物理先验DAG ======================
physical_edges = []
for f in feature_col:
    physical_edges.append((fault_col, f))
    physical_edges.append((operate_col, f))
for f in feature_col:
    physical_edges.append((f, outcome_col))

physical_dag = DiscreteBayesianNetwork(physical_edges)
draw_and_save(
    physical_dag,
    'Physical Prior DAG (Fault+Condition → All Features)',
    f'{save_dir}/GearBox_PC_DAG_physical_prior.png',
    rad1=0.0, rad2=0.0
)



# ====================== 计算因果强度======================
def get_causal_strength_dict(dag, feature_cols, fault_col, condition_col, label_col):
    G = nx.DiGraph(dag)
    causal_strength = {}
    for feat in feature_cols:
        from_fault = 1 if G.has_edge(fault_col, feat) else 0
        from_cond = 1 if G.has_edge(condition_col, feat) else 0
        to_label = 1 if G.has_edge(feat, label_col) else 0
        total = round(from_fault * 0.4 + from_cond * 0.2 + to_label * 0.4, 2)
        causal_strength[feat] = total
    return causal_strength

causal_strength_dict = get_causal_strength_dict(
    dag=learned_dag,
    feature_cols=feature_col,
    fault_col=fault_col,
    condition_col=operate_col,
    label_col=outcome_col
)


# ====================== 绘图：带因果强度显示 ======================
def draw_and_save_with_strength(G, title, save_path, rad1=0.02, rad2=0.05):
    plt.figure(figsize=(20, 12))
    pos = three_layer_layout_optimized(G)
    all_nodes = [fault_col, operate_col] + feature_col + [outcome_col]
    current_colors = get_unified_colors(G)

    # 画节点
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=all_nodes,
        node_size=NODE_SIZE,
        node_color=current_colors,
        alpha=0.8,
        edgecolors='black',
        linewidths=NODE_BORDER_WIDTH
    )

    # 画边
    feat_edges = []
    other_edges = []
    for u, v in G.edges():
        if u in feature_col and v in feature_col:
            feat_edges.append((u, v))
        else:
            other_edges.append((u, v))

    nx.draw(G, pos, edgelist=other_edges, with_labels=False, arrowstyle="->", arrowsize=32,
            edge_color="#333", width=2.0, alpha=0.9, connectionstyle=f'arc3,rad={rad1}')
    nx.draw(G, pos, edgelist=feat_edges, with_labels=False, arrowstyle="->", arrowsize=32,
            edge_color="#7F8C8D", width=1.8, alpha=0.9, connectionstyle=f'arc3,rad={rad2}')

    # 画节点名称
    nx.draw_networkx_labels(G, pos, labels=node_label_map, font_size=12, font_family="Times New Roman", font_weight="bold")

    # ====================== 直接在节点上显示因果强度 ======================
    strength_labels = {feat: f"{causal_strength_dict[feat]:.2f}" for feat in feature_col}
    pos_strength = {k: (x, y - 0.065) for k, (x, y) in pos.items()}  # 放在节点下方
    nx.draw_networkx_labels(
        G, pos_strength,
        labels=strength_labels,
        font_size=11,
        font_family="Times New Roman",
        font_weight="bold",
        font_color="navy"
    )

    plt.title(title, fontsize=16, fontweight='bold', pad=30)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=600, format='png', facecolor='white')
    plt.close()

# ====================== 绘图：过滤后的DAG（带因果强度） ======================
valid_edges = []
for u, v in learned_dag.edges():
    if (u in [fault_col, operate_col] and v in feature_col) or (u in feature_col and v == outcome_col):
        valid_edges.append((u, v))

filtered_dag = DiscreteBayesianNetwork(valid_edges)

draw_and_save_with_strength(
    filtered_dag,
    'Filtered Causal DAG with Feature Causal Strength',
    f'{save_dir}/GearBox_PC_DAG_filtered_with_strength.png'
)