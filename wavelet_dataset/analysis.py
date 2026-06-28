import numpy as np
import pandas as pd
import warnings
from causallearn.search.ConstraintBased.PC import pc
from dowhy import CausalModel
from causallearn.utils.cit import CIT

warnings.filterwarnings("ignore", category=RuntimeWarning)

def causal_data(data):
    var = np.var(data, axis=-1)
    avg = np.mean(data, axis=-1, keepdims=True)
    abs_avg = np.mean(abs(data), axis=-1)

    x_pp = np.max(data, axis=-1) - np.min(data, axis=-1)
    kurt = np.mean((data - avg)**4) / var**2
    impulse_factor = np.max(abs(data), axis=-1) / abs_avg

    features = np.concatenate([var, x_pp, kurt, impulse_factor], axis=-1)
    mean = np.mean(features, axis=0, keepdims=True)
    std = np.std(features, axis=0, keepdims=True)
    features = (features - mean) / std
    return features.reshape(data.shape[0], -1)

def dowhy(cg, data, save_name):
    M = cg.G.graph
    n_nodes = cg.G.get_num_nodes()
    feat_names = [f"X{i}" for i in range(data.shape[1])]
    df = pd.DataFrame(data, columns=feat_names)

    direct_edges = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if M[i, j] == -1 and M[j, i] == 1:
                direct_edges.append((i, j))

    if len(direct_edges) == 0:
        print("PC未识别出任何确定有向因果边，终止干预实验")
        return [], []

    print(f"共提取 {len(direct_edges)} 条确定有向因果边，开始批量干预验证\n")
    result_list = []
    skip_edges = []

    for treat_idx, out_idx in direct_edges:
        T = f"X{treat_idx}"
        Y = f"X{out_idx}"
        print("-" * 60)
        print(f"正在验证因果定向：{T} → {Y}")
        try:
            # 最简模型，不做复杂后门筛选，不调用检验/反驳接口
            model = CausalModel(data=df, treatment=T, outcome=Y)
            estimand = model.identify_effect(proceed_when_unidentifiable=True)
            estimate = model.estimate_effect(
                estimand,
                method_name="backdoor.linear_regression"
            )
            ate_val = round(estimate.value, 4)

            record = {
                "T_idx": treat_idx,
                "Y_idx": out_idx,
                "T_name": T,
                "Y_name": Y,
                "ATE": ate_val
            }
            result_list.append(record)
            print(f"平均干预效应 ATE = {ate_val:.4f}")

        except Exception as e:
            print(f"【警告】{T} → {Y} 计算失败，跳过，原因：{str(e)[:180]}")
            skip_edges.append((T, Y))
            continue

    # 汇总打印
    print("\n" + "=" * 70)
    print("全部因果边干预ATE汇总表")
    print("=" * 70)
    for res in result_list:
        print(f"{res['T_name']} → {res['Y_name']} | ATE={res['ATE']}")

    if len(skip_edges) > 0:
        print(f"\n===== 计算失败、跳过的因果边（共{len(skip_edges)}条） =====")
        for t, y in skip_edges:
            print(f"{t} → {y}")

    # 保存csv
    res_df = pd.DataFrame(result_list)
    res_df.to_csv(f"./{save_name}_intervention_result.csv", index=False, encoding="utf-8-sig")
    print(f"\n工况 {save_name} 干预结果已保存至 ./{save_name}_intervention_result.csv")
    return result_list, skip_edges

if __name__ == "__main__":
    path1 = './Bearing_20_0.npz'
    path2 = './Bearing_30_2.npz'
    data1 = np.load(path1)['train_set'][:, :, 1:, :]
    data2 = np.load(path2)['train_set'][:, :, 1:, :]
    label1 = np.load(path1)['train_label']
    label2 = np.load(path2)['train_label']

    cat_mask1 = np.where(label1 == 1)[0]
    cat_mask2 = np.where(label2 == 1)[0]

    cat1_data = data1[cat_mask1]
    cat2_data = data2[cat_mask2]

    cat1_data = causal_data(cat1_data)
    cat2_data = causal_data(cat2_data)

    ci_test1 = CIT(cat1_data, method='kci')
    ci_test2 = CIT(cat2_data, method='kci')

    # PC 兼容传参
    cg1 = pc(
        cat1_data,
        ci_test=ci_test1,
        alpha=0.05,
        show_progress=True,
        stable=True
    )
    cg2 = pc(
        cat2_data,
        ci_test=ci_test2,
        alpha=0.05,
        show_progress=True,
        stable=True
    )

    print("===== 开始处理工况 20_0 =====")
    res_20_0, skip_20_0 = dowhy(cg1, cat1_data, "20_0")
    print("\n===== 开始处理工况 30_2 =====")
    res_30_2, skip_30_2 = dowhy(cg2, cat2_data, "30_2")