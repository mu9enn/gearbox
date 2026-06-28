import numpy as np
import torch
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import CIT
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode

# 矩阵转稀疏索引
def np_dense_to_sparse(mat: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.from_numpy(mat).nonzero(as_tuple=False).t(), torch.from_numpy(mat)[mat != 0]


class CausalAnalysis:
    def __init__(self, poly_data: np.ndarray = None):
        self.poly_data = poly_data
        self.n_samples, self.n_vars = poly_data.shape
        self.kci_cit = CIT(data=self.poly_data, method="kci")

    @staticmethod
    def build_mechanical_background_knowledge() -> BackgroundKnowledge:
        bk = BackgroundKnowledge()
        num_phys_node = 30
        stat_num = 4
        # 缓存所有GraphNode，避免重复创建
        node_cache = {}
        # 预生成全部72个节点
        for feat_id in range(num_phys_node * stat_num):
            node_cache[str(feat_id)] = GraphNode(str(feat_id))

        for phys_u in range(num_phys_node):
            ch_u = phys_u // 3
            feat_u_base = phys_u * stat_num
            for phys_v in range(num_phys_node):
                if phys_u == phys_v:
                    continue
                ch_v = phys_v // 3
                feat_v_base = phys_v * stat_num

                if ch_u == ch_v:
                    # 同通道双向强制边
                    for fu in range(stat_num):
                        f_u_idx = feat_u_base + fu
                        node_u = node_cache[str(f_u_idx)]
                        for fv in range(stat_num):
                            f_v_idx = feat_v_base + fv
                            node_v = node_cache[str(f_v_idx)]
                            bk.add_required_by_node(node_u, node_v)
                            bk.add_required_by_node(node_v, node_u)
                else:
                    # 跨通道双向禁止边
                    for fu in range(stat_num):
                        f_u_idx = feat_u_base + fu
                        node_u = node_cache[str(f_u_idx)]
                        for fv in range(stat_num):
                            f_v_idx = feat_v_base + fv
                            node_v = node_cache[str(f_v_idx)]
                            bk.add_forbidden_by_node(node_u, node_v)
                            bk.add_forbidden_by_node(node_v, node_u)
        return bk

    def causal(self, p):
        cg = pc(
            data=self.poly_data,
            ci_test=self.kci_cit,
            alpha=p,
            show_progress=True,
            stable=True,
            background_knowledge=self.build_mechanical_background_knowledge(),
            max_k=6
        )
        matrix = cg.G.graph
        index, _ = np_dense_to_sparse(matrix)
        src_mask = torch.argsort(index[0])
        src_index = index[0][src_mask]
        target_index = index[1][src_mask]
        edge_matrix = torch.stack([src_index, target_index], dim=0)
        return cg, edge_matrix

    def find_all_confounders(self, edge_matrix: torch.Tensor, alpha: float = 0.05) -> dict:
        edge_conf_map = {}
        for e_idx in range(edge_matrix.shape[1]):
            x = int(edge_matrix[0, e_idx])
            y = int(edge_matrix[1, e_idx])
            conf_candidates = []
            for z in range(self.n_vars):
                if z == x or z == y:
                    continue
                p_xz = self.kci_cit(X=x, Y=z, condition_set=None)
                p_yz = self.kci_cit(X=y, Y=z, condition_set=None)
                if p_xz < alpha and p_yz < alpha:
                    conf_candidates.append(z)
            edge_conf_map[(x, y)] = conf_candidates
        return edge_conf_map

    def recheck_edge_with_confounder(self, alpha: float = 0.05) -> tuple[torch.Tensor, dict]:
        _, origin_edge = self.causal(p=alpha)
        edge_conf_map = self.find_all_confounders(origin_edge, alpha)

        _, edge_ctrl = self.causal(p=alpha)
        valid_edges = []
        valid_conf = {}

        edge_set_ctrl = set(zip(edge_ctrl[0].tolist(), edge_ctrl[1].tolist()))
        for (x, y), confs in edge_conf_map.items():
            if (x, y) in edge_set_ctrl:
                valid_edges.append([x, y])
                valid_conf[(x, y)] = confs

        if valid_edges:
            valid_edge_mat = torch.tensor(valid_edges).t()
        else:
            valid_edge_mat = torch.empty((2, 0), dtype=torch.long)
        return valid_edge_mat, valid_conf

    def virtual_do_intervention_quantile(self, x: int, y: int, conf_list: list, quantiles=None) -> tuple[list, list, bool]:
        if quantiles is None:
            quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]

        feat_idx = [x] + conf_list
        X_feat = self.poly_data[:, feat_idx]
        Y_true = self.poly_data[:, y]
        x_col = self.poly_data[:, x]

        # 特征方差校验
        if np.std(x_col) < 1e-8:
            print(f"[警告] 变量{x}方差趋近于0，无法判定因果趋势")
            return [], [], False

        inter_values = np.quantile(x_col, quantiles)
        inter_values = np.atleast_1d(inter_values).tolist()

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_split=8,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=42,
            oob_score=False
        )
        model.fit(X_feat, Y_true)

        y_mean_result = []
        for val in inter_values:
            inter_feat = X_feat.copy()
            inter_feat[:, 0] = val
            y_pred = model.predict(inter_feat)
            y_mean_result.append(np.mean(y_pred))

        diff = np.diff(y_mean_result)
        thresh = 1e-4
        is_pos = np.all(diff > thresh)
        is_neg = np.all(diff < -thresh)
        is_valid = is_pos or is_neg

        return inter_values, y_mean_result, is_valid

    def generate_and_save_reliable_edges(self, alpha: float = 0.05, save_path: str = "../GNNCausal/reliable_edge.pt") -> torch.Tensor:
        print("===== 开始离线生成可靠因果边矩阵 =====")
        valid_edge_mat, edge_conf_map = self.recheck_edge_with_confounder(alpha)
        print(f"PC初筛边数量: {valid_edge_mat.shape[1]}")

        final_edges = []
        for (x, y), confs in edge_conf_map.items():
            inter_x, inter_y, is_valid = self.virtual_do_intervention_quantile(x, y, confs)
            print(f"\n边 {x} -> {y} 干预校验: {'有效因果' if is_valid else '伪关联，剔除'}")
            if inter_x and inter_y:
                for xv, yv in zip(inter_x, inter_y):
                    print(f"do(X = {xv:.2f})  →  Y均值 = {yv:.4f}")
            if is_valid:
                final_edges.append([x, y])

        if final_edges:
            reliable_edge = torch.tensor(final_edges).t()
        else:
            reliable_edge = torch.empty((2, 0), dtype=torch.long)

        # 自动创建目录
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(reliable_edge, save_p)
        print(f"\n✅ 最终可靠边矩阵已保存至: {save_path} | 有效边数: {reliable_edge.shape[1]}")
        return reliable_edge


def load_data(file_name):
    data = np.load(file_name, allow_pickle=True)
    train_set = data['train_set'][:, :, 1:, :]
    test_set = data['test_set'][:, :, 1:, :]
    valid_set = data['valid_set'][:, :, 1:, :]
    finetune_set = data['cross_finetune_set'][:, :, 1:, :]
    cross_test_set = data['cross_test_set'][:, :, 1:, :]

    train_label = data['train_label']
    valid_label = data['valid_label']
    test_label = data['test_label']
    finetune_label = data['cross_finetune_label']
    cross_test_label = data['cross_test_label']

    return (train_set.reshape(-1, 30, 1024), valid_set.reshape(-1, 30, 1024), test_set.reshape(-1, 30, 1024),
            finetune_set.reshape(-1, 30, 1024), cross_test_set.reshape(-1, 30, 1024),
            train_label, valid_label, test_label, finetune_label, cross_test_label)


def causal_data(data, scaler):
    """
    输入data: (N, 30, 1024)
    """
    avrig = np.mean(data, axis=-1, keepdims=True)#(n,30,1)
    abs_avrig = np.mean(abs(data), axis=-1)#(n,30)
    var = np.var(data, axis=-1)
    x_pp = np.max(data, axis=-1) - np.min(data, axis=-1)
    kurt = np.mean((data - avrig) ** 4, axis=-1)/var ** 2#(n,30)
    impulse_factor = np.max(abs(data), axis=-1)/abs_avrig

    feat_4dim = np.concatenate([var, x_pp, kurt, impulse_factor], axis=-1)

    causal_sample = feat_4dim.reshape(data.shape[0], -1)

    return scaler.fit_transform(causal_sample)


if __name__ == '__main__':
    # 全局固定随机种子，保证实验可复现
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    data_dir = '../wavelet_dataset'
    files = os.listdir(data_dir)
    file_name = [f for f in files if '0' in f]

    scaler = StandardScaler()

    for idx, file in enumerate(file_name):
        try:
            print(f"\n========== 正在处理 {idx+1}/{len(file_name)} | 文件: {file} ==========")
            file_path = os.path.join(data_dir, file)
            name = file.split('.')[0]

            x_train, x_valid, x_test, x_finetune, x_cross_test, \
            y_train, y_valid, y_test, y_finetune, y_cross_test = load_data(file_path)

            train_causal = causal_data(x_train, scaler)
            analysis = CausalAnalysis(poly_data=train_causal)
            out_path = f'../GNNCausal/{name}_reliable_edge.pt'
            analysis.generate_and_save_reliable_edges(alpha=0.05, save_path=out_path)
            # _,matrix = analysis.causal(0.05)
            # print(matrix)
            print(f"✅ {file} 处理完成")

        except Exception as e:
            print(f"❌ {file} 处理失败: {str(e)}")
            continue
