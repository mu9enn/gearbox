import numpy as np
from typing import Tuple


def load_npz_file(file_path: str, key: str = None) -> np.ndarray:
    """
    安全加载npz文件，包含异常处理和键值校验

    Args:
        file_path: npz文件路径
        key: 要提取的键值（若为None则返回整个npz对象）

    Returns:
        提取的数组或npz对象
    """
    try:
        data = np.load(file_path)
        if key is not None:
            if key not in data.files:
                raise KeyError(f"npz文件 {file_path} 中不存在键值: {key}")
            data_np = data[key]
            return np.squeeze(data_np)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"文件不存在: {file_path}，请检查路径是否正确")
    except Exception as e:
        raise Exception(f"加载文件 {file_path} 失败: {str(e)}")


def calculate_abs_mean_shap(shap_array: np.ndarray, axes: Tuple = (0, 2, 3, 4)) -> np.ndarray:
    """
    计算SHAP值的绝对值均值（指定轴）

    Args:
        shap_array: SHAP值数组
        axes: 要求均值的轴

    Returns:
        绝对值均值数组
    """
    mean_abs_shap = np.mean(np.abs(shap_array), axis=axes)
    return mean_abs_shap/np.sum(mean_abs_shap)


def calculate_correlation(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """
    计算两个数组的皮尔逊相关系数，包含维度校验

    Args:
        arr1: 第一个数组
        arr2: 第二个数组

    Returns:
        相关系数（-1到1之间）
    """
    # 展平数组并校验长度
    arr1_flat = arr1.flatten()
    arr2_flat = arr2.flatten()

    if len(arr1_flat) != len(arr2_flat):
        raise ValueError(f"数组长度不匹配！arr1长度: {len(arr1_flat)}, arr2长度: {len(arr2_flat)}")

    # 计算相关系数（避免全零数组导致的nan）
    corr_matrix = np.corrcoef(arr1_flat, arr2_flat)
    corr = corr_matrix[0, 1]
    return corr if not np.isnan(corr) else 0.0  # 处理nan情况


# ===================== 主逻辑 =====================
if __name__ == "__main__":
    # 1. 加载权重和SHAP值（安全加载）
    # Bearing_20_0_weights = load_npz_file('../ModelTrain/SavedWeights/Seed_49/Bearing_20_0.npz','SC_channel_weights')
    # Bearing_30_2_weights = load_npz_file('../ModelTrain/SavedWeights/Seed_49/Bearing_30_2.npz','SC_channel_weights')
    # Gear_20_0_weights = load_npz_file('../ModelTrain/SavedWeights/Seed_49/Gear_20_0.npz', 'SC_channel_weights')
    # Gear_30_2_weights = load_npz_file('../ModelTrain/SavedWeights/Seed_49/Gear_30_2.npz', 'SC_channel_weights')

    # Bearing_20_0_weights_last = Bearing_20_0_weights[-1]/np.sum(Bearing_20_0_weights[-1])
    # Bearing_30_2_weights_last = Bearing_30_2_weights[-1]/np.sum(Bearing_30_2_weights[-1])
    # Gear_20_0_weights_last = Gear_20_0_weights[-1]/np.sum(Gear_20_0_weights[-1])
    # Gear_30_2_weights_last = Gear_30_2_weights[-1]/np.sum(Gear_30_2_weights[-1])

    # weights = {
    #     'Bearing_20_0': Bearing_20_0_weights_last,
    #     'Bearing_30_2': Bearing_30_2_weights_last,
    #     'Gear_20_0': Gear_20_0_weights_last,
    #     'Gear_30_2': Gear_30_2_weights_last,
    # }

    # weight = np.load('../ModelTrain/AttentionAndNoAttention/ChannelWeights/ChannelWeights.npz')['weights']
    # weight = np.load("../ModelTrain/AttentionAndNoAttention/ChannelWeights/Seed_56/Bearing_20_0.npz")
    weights = {
        'Bearing_20_0': np.load("../ModelTrain/AttentionAndNoAttention/ChannelWeights/ChannelWeights.npz")['weights'],
        'Bearing_30_2': np.load("../ModelTrain/AttentionAndNoAttention/ChannelWeights/ChannelWeights.npz")['weights'],
        'Gear_20_0': np.load("../ModelTrain/AttentionAndNoAttention/ChannelWeights/ChannelWeights.npz")['weights'],
        'Gear_30_2': np.load("../ModelTrain/AttentionAndNoAttention/ChannelWeights/ChannelWeights.npz")['weights']
    }
    # 定义SHAP文件路径和对应的标签
    shap_file_configs = [
        ('../CalculateShapValues/AttentionAndNoAttention/Seed_56/Bearing_20_0_ShapValues.npz', 'Bearing_20_0'),
        ('../CalculateShapValues/AttentionAndNoAttention/Seed_56/Bearing_30_2_ShapValues.npz', 'Bearing_30_2'),
        ('../CalculateShapValues/AttentionAndNoAttention/Seed_56/Gear_20_0_ShapValues.npz', 'Gear_20_0'),
        ('../CalculateShapValues/AttentionAndNoAttention/Seed_56/Gear_30_2_ShapValues.npz', 'Gear_30_2'),
    ]

    # 存储结果的字典（便于后续分析）
    results = {
        'abs_mean_shap': {},
        'correlation': {}
    }

    # 2. 批量计算绝对值均值和相关系数
    for file_path, label in shap_file_configs:
        shap_data = load_npz_file(file_path)

        # 计算attention和no_attention的绝对值均值
        shap_attention = calculate_abs_mean_shap(shap_data['attention'])
        shap_no_attention = calculate_abs_mean_shap(shap_data['no_attention'])

        # 存储绝对值均值
        results['abs_mean_shap'][f'{label}_attention'] = shap_attention
        results['abs_mean_shap'][f'{label}_no_attention'] = shap_no_attention

        # 计算相关系数
        corr_attention = calculate_correlation(shap_attention, weights[label])
        corr_no_attention = calculate_correlation(shap_no_attention, weights[label])

        # 存储相关系数
        results['correlation'][f'{label}_attention'] = corr_attention
        results['correlation'][f'{label}_no_attention'] = corr_no_attention

    # 3. 结构化输出结果
    print("=" * 60)
    print("通道权重最后一轮:")
    print(weights.items())
    print("=" * 60)

    print("\n【SHAP值绝对均值】")
    for key, value in results['abs_mean_shap'].items():
        print(f"{key}: {value}")

    print("\n【皮尔逊相关系数】")
    for key, value in results['correlation'].items():
        print(f"{key}: {value:.6f}")  # 保留6位小数，便于对比