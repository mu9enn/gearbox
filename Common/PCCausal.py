import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from pgmpy.estimators import ExpertKnowledge
from matplotlib.colors import LinearSegmentedColormap
from sklearn.mixture import GaussianMixture
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import CIT
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'axes.unicode_minus': False,
    'axes.linewidth': 1.2,
})

# class GearBoxCausalAnalyzer:
#     def __init__(self, dataset):
#         self.dataset = dataset.copy()
#         self.operate_col = 'condition'
#         self.fault_col = 'fault'
#         self.outcome_col = 'label'
#         self.feature_col = [
#             'mean', 'std', 'rms', 'peak', 'crest_factor', 'kurtosis', 'skewness',
#             'mfe', 'ebe', 'cf', 'thd'
#         ]
#
#     def pc_dag(self):
#         df = self.dataset.copy()
#         # 1. 分类变量编码
#         for col in [self.operate_col, self.fault_col, self.outcome_col]:
#             df[col] = df[col].astype('category').cat.codes
#
#         # 2. 特征预处理 → 强制转为离散分箱格式（提升区分度）
#
#         for feat in self.feature_col:
#             # 清洗异常值
#             df[feat] = pd.to_numeric(df[feat], errors='coerce')
#             df[feat] = df[feat].replace([np.inf, -np.inf], np.nan)
#             df[feat] = df[feat].fillna(df[feat].median())
#
#             df[feat] = pd.qcut(
#                 df[feat],
#                 q=5,
#                 labels=False,
#                 duplicates='drop'
#             )
#
#         all_nodes = self.feature_col + [self.operate_col, self.fault_col, self.outcome_col]
#
#         forbidden = []
#         # 特征 不能 → 故障/工况
#         for f in self.feature_col:
#             forbidden.append((f, self.fault_col))
#             forbidden.append((f, self.operate_col))
#
#         # 最终标签 不能 → 任何变量
#         for n in all_nodes:
#             forbidden.append((self.outcome_col, n))
#
#         # 时序因果顺序：强制因果方向（最核心）
#         temporal_order = [
#             [self.fault_col, self.operate_col],  # 第一层：根因
#             self.feature_col,                    # 第二层：特征
#             [self.outcome_col]                   # 第三层：结果
#         ]
#
#         ek = ExpertKnowledge(
#             forbidden_edges=forbidden,
#             temporal_order=temporal_order
#         )
#
#         pc = PC(data=df)
#         dag = pc.estimate(
#             ci_test='chi_square',
#             # ci_test='pearsonr',
#             expert_knowledge=ek,
#             significance_level=0.15,
#             max_cond_vars=2,
#             variant='stable',
#             show_progress=True
#         )
#
#         return dag

class WaveletDAG:
    def __init__(self, dataset):
        self.dataset = dataset.copy()
        self.ch_feature_cols = ['ch2', 'ch3', 'ch4', 'ch6', 'ch7', 'ch8']
        self.freq_feature_cols = ['cD2', 'cD3', 'cD4', 'cD5', 'cA5']
        self.outcome_col = 'label'

    def _build_temporal_bk(self, feature_cols):
        """
        内部工具：根据「特征列 + 结果列」构建时序约束 BK
        规则：label 不能指向任何特征（特征时间上更早）
        """
        all_cols = feature_cols + [self.outcome_col]
        # 1. 列名 -> GraphNode
        nodes = {col: GraphNode(col) for col in all_cols}
        # 2. 初始化 BK
        bk = BackgroundKnowledge()
        # 3. 时序约束：禁止 label → 特征（未来不能影响过去）
        for feat in feature_cols:
            bk.add_forbidden_by_node(nodes[self.outcome_col], nodes[feat])
        return bk

    def pc_dag(self):
        all_nodes = self.ch_feature_cols + self.freq_feature_cols
        # ---------------------- 数据清洗 ----------------------
        for feat in all_nodes:
            self.dataset[feat] = pd.to_numeric(self.dataset[feat], errors='coerce')
            self.dataset[feat] = self.dataset[feat].replace([np.inf, -np.inf], np.nan)
            self.dataset[feat] = self.dataset[feat].fillna(self.dataset[feat].median())

        # ---------------------- 构建时序BK ----------------------
        ch_bk = self._build_temporal_bk(self.ch_feature_cols)
        freq_bk = self._build_temporal_bk(self.freq_feature_cols)

        # ---------------------- 准备数据 ----------------------
        ch_data = self.dataset[self.ch_feature_cols + [self.outcome_col]]
        freq_data = self.dataset[self.freq_feature_cols + [self.outcome_col]]

        # ---------------------- 条件独立检验 ----------------------
        ch_ci_test = CIT(data=ch_data.values, method='kci')
        freq_ci_test = CIT(data=freq_data.values, method='kci')

        # ---------------------- PC 因果推断 ----------------------
        ch_dag = pc(
            ch_data.values,
            ci_test=ch_ci_test,
            background_knowledge=ch_bk,   # 改用 BK
            significance_level=0.05,
            max_cond_vars=len(ch_data.columns) - 2,
            stable=True,
            show_progress=True
        )

        freq_dag = pc(
            freq_data.values,
            ci_test=freq_ci_test,
            background_knowledge=freq_bk, # 改用 BK
            significance_level=0.05,
            max_cond_vars=len(freq_data.columns) - 2,
            stable=True,
            show_progress=True
        )

        return ch_dag, freq_dag

    def draw_single_dag(self, current_dag, plot_title, save_name, all_nodes, feature_cols, outcome_col, pos,
                        shap_values_dict, min_shap, max_shap, cmap):
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams.update({
            'font.size': 16,
            'axes.linewidth': 1.2,
        })
        plt.figure(figsize=(17, 8), dpi=600)
        plt.ylim(0, 0.58)

        G_plot = nx.DiGraph()
        G_plot.add_nodes_from(all_nodes)

        # ===================== 自动兼容两种图类型 =====================
        # 如果是 causallearn 的 CausalGraph
        if hasattr(current_dag, 'G'):
            name_map = {node.get_name(): all_nodes[i] for i, node in enumerate(current_dag.G.get_nodes())}
            for edge in current_dag.G.get_graph_edges():
                u = name_map[edge.get_node1().get_name()]
                v = name_map[edge.get_node2().get_name()]
                G_plot.add_edge(u, v)
        # 如果是 networkx DiGraph（过滤后的图）
        else:
            for u, v in current_dag.edges():
                G_plot.add_edge(u, v)
        # ============================================================

        # 画节点
        nx.draw_networkx_nodes(G_plot, pos, nodelist=feature_cols, node_color='#2CA02C', node_size=7000,
                               linewidths=2.5)
        nx.draw_networkx_nodes(G_plot, pos, nodelist=[outcome_col], node_color='#D62728', node_size=9000,
                               linewidths=2.5)

        # 逐条画边
        for u, v in G_plot.edges():
            if u in feature_cols and v == outcome_col:
                shap_norm = shap_values_dict[u]
                width = 1.5 + (shap_norm - min_shap) / (max_shap - min_shap) * 5
                color = cmap(width)
                conn = 'arc3,rad=0'

                nx.draw_networkx_edges(
                    G_plot, pos,
                    edgelist=[(u, v)],
                    width=width,
                    edge_color=color,
                    connectionstyle=conn,
                    arrowstyle='->',
                    arrowsize=60,
                    min_source_margin=45,
                    min_target_margin=45
                )

                nx.draw_networkx_edge_labels(
                    G_plot, pos,
                    edge_labels={(u, v): f"{shap_norm:.7f}"},
                    font_size=27,
                    font_color='black',
                    font_weight='bold',
                    label_pos=0.5,
                    font_family='Times New Roman'
                )

            else:
                width = 1.5
                color = "#444444"
                conn = 'arc3,rad=0.15'

                nx.draw_networkx_edges(
                    G_plot, pos,
                    edgelist=[(u, v)],
                    width=width,
                    edge_color=color,
                    connectionstyle=conn,
                    arrowstyle='->',
                    arrowsize=35,
                    min_source_margin=40,
                    min_target_margin=40
                )

        nx.draw_networkx_labels(
            G_plot, pos,
            font_weight='bold',
            font_size=34,
            font_family='Times New Roman'
        )

        plt.title(plot_title, fontweight='bold', pad=30, fontsize=34)
        plt.axis('off')
        plt.savefig(save_name, dpi=600, bbox_inches='tight', facecolor='white')
        plt.close()

    def plot_dag_clean(self, dag, title, feature_cols, raw_save_path, filtered_save_path, shap_values_dict):
        outcome_col = self.outcome_col
        all_nodes = feature_cols + [outcome_col]

        # ---------------------- 你的 pos 布局（保持不变） ----------------------
        pos = {
            'ch2': (0.1, 0.5),
            'ch3': (0.25, 0.5),
            'ch4': (0.4, 0.5),
            'ch6': (0.55, 0.5),
            'ch7': (0.7, 0.5),
            'ch8': (0.85, 0.5),
            'cD2': (0.1, 0.5),
            'cD3': (0.25, 0.5),
            'cD4': (0.4, 0.5),
            'cD5': (0.55, 0.5),
            'cA5': (0.7, 0.5),
            'label': (0.5, 0.1)
        }

        min_shap = min(shap_values_dict.values())
        max_shap = max(shap_values_dict.values())
        cmap = plt.cm.Greens

        # ---------------------- 绘制原始 DAG ----------------------
        self.draw_single_dag(
            dag, title, raw_save_path,
            all_nodes, feature_cols, outcome_col,
            pos, shap_values_dict, min_shap, max_shap, cmap
        )

        # ---------------------- 绘制过滤后的 DAG（只保留特征→label） ----------------------
        filtered_dag = nx.DiGraph()
        filtered_dag.add_nodes_from(all_nodes)

        name_map = {node.get_name(): all_nodes[i] for i, node in enumerate(dag.G.get_nodes())}
        for edge in dag.G.get_graph_edges():
            u = name_map[edge.get_node1().get_name()]
            v = name_map[edge.get_node2().get_name()]
            if u in feature_cols and v == outcome_col:
                filtered_dag.add_edge(u, v)

        self.draw_single_dag(
            filtered_dag, title, filtered_save_path,
            all_nodes, feature_cols, outcome_col,
            pos, shap_values_dict, min_shap, max_shap, cmap
        )

    def visualize_both_dags(self, ch_dag, freq_dag, orig_ch_path, orig_freq_path, filtered_ch_path,
                            filtered_freq_path, ch_shap_dict, freq_shap_dict):
        cond_ch = orig_ch_path.split('\\')[-1].split('.')[0]
        cond_freq = orig_freq_path.split('\\')[-1].split('.')[0]
        self.plot_dag_clean(ch_dag,
                            f"Channel Causal DAG({cond_ch}, alpha = 0.05)",
                            self.ch_feature_cols,
                            orig_ch_path,
                            filtered_ch_path, ch_shap_dict)
        self.plot_dag_clean(freq_dag,
                            f"Frequency Causal DAG({cond_freq}, alpha = 0.05)",
                            self.freq_feature_cols,
                            orig_freq_path,
                            filtered_freq_path,
                            freq_shap_dict)