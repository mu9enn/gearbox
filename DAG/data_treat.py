import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pywt

from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
from tqdm import tqdm
from scipy.signal import savgol_filter

# --------------------------
# 全局绘图风格设置
# --------------------------
plt.rcParams.update({
    'font.sans-serif': ['Times New Roman'],
    'axes.unicode_minus': False,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'lines.linewidth': 1.2,
    'lines.markersize': 4,
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'font.family': 'serif',
    'legend.fontsize': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})

bearing_label_map = {
    "ball_20_0.csv": "ball_20_0",
    "ball_30_2.csv": "ball_30_2",
    "comb_20_0.csv": "comb_20_0",
    "comb_30_2.csv": "comb_30_2",
    "health_20_0.csv": "bearing_normal_20_0",
    "health_30_2.csv": "bearing_normal_30_2",
    "inner_20_0.csv": "inner_20_0",
    "inner_30_2.csv": "inner_30_2",
    "outer_20_0.csv": "outer_20_0",
    "outer_30_2.csv": "outer_30_2"
}

gear_label_map = {
    "Chipped_20_0.csv": "chipped_20_0",
    "Chipped_30_2.csv": "chipped_30_2",
    "Health_20_0.csv": "gear_normal_20_0",
    "Health_30_2.csv": "gear_normal_30_2",
    "Miss_20_0.csv": "miss_20_0",
    "Miss_30_2.csv": "miss_30_2",
    "Root_20_0.csv": "root_20_0",
    "Root_30_2.csv": "root_30_2",
    "Surface_20_0.csv": "surface_20_0",
    "Surface_30_2.csv": "surface_30_2"
}


def denoise_signal(signal, method="savgol", window=51, poly=3, wavelet="db4", level=3):
    """
    对去趋势后的纯净信号进行去噪
    :param signal: 输入信号 (n,)
    :param method: "savgol" 或 "wavelet"
    :return: 去噪后的信号
    """
    if method == "savgol":
        window = min(window, len(signal) // 2 * 2 - 1)  # 保证奇数
        return savgol_filter(signal, window_length=window, polyorder=poly)

    elif method == "wavelet":
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
        denoised = pywt.waverec([pywt.threshold(c, uthresh) for c in coeffs], wavelet)
        return denoised[:len(signal)]


def fast_sliding_fit(x, y, window_size=200, step=100):
    n = len(x)
    half_window = window_size // 2
    model = LinearRegression()

    x_sample = []
    y_sample = []
    for i in range(0, n, step):
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)

        x_win = x[start:end].reshape(-1, 1)
        y_win = y[start:end]

        model.fit(x_win, y_win)
        val = model.predict([[x[i]]])[0]

        x_sample.append(x[i])
        y_sample.append(val)

    f = interp1d(x_sample, y_sample, kind='linear', fill_value="extrapolate")
    prediction = f(x)
    return prediction


save_dir = 'PolyCurves/Noise'
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

Pure_data_dir = 'Noise'
if not os.path.exists(Pure_data_dir):
    os.makedirs(Pure_data_dir, exist_ok=True)

index = [2, 3, 4, 6, 7, 8]

# --------------------------
# 处理轴承数据
# --------------------------
for filename in tqdm(bearing_label_map.keys(), desc="绘制轴承拟合曲线"):
    try:
        file_path = f'../DataSets/bearing_set/{filename}'
        data = pd.read_csv(
            file_path,
            sep=r'\s+|,',
            usecols=range(8),
            skiprows=16,
            engine='python'
        ).values
        dataset = data[:, [1, 2, 3, 5, 6, 7]]

        pure_data = []
        poly_data = []

        for i in range(dataset.shape[1]):
            x = np.arange(len(dataset))
            y = dataset[:, i]

            pred = fast_sliding_fit(x, y, window_size=200, step=5)
            poly_data.append(pred)
            pure_signal = y - pred
            pure_data.append(pure_signal)

        pure_data = np.array(pure_data).T
        poly_data = np.array(poly_data).T
        x_plot = np.arange(len(dataset))

        # 大幅增加画布高度
        fig, axs = plt.subplots(3, 2, figsize=(16, 10))
        axs = axs.flatten()

        for i in range(6):
            # 绘制散点和拟合线
            axs[i].scatter(x_plot, dataset[:, i], color='#1f77b4', s=0.2, alpha=0.4, label='Original')
            axs[i].plot(x_plot, poly_data[:, i], color='#ff4b33', linewidth=0.6, label='Fitted Trend')

            # 设置标题和刻度
            axs[i].set_title(f"Channel {index[i]}", fontsize=18)
            axs[i].tick_params(axis='both', labelsize=14)

            # 【你的核心要求】Y轴范围 = 数据实际的min和max，信号撑满整个子图
            y_data = dataset[:, i]
            y_min = y_data.min()
            y_max = y_data.max()
            axs[i].set_ylim(y_min, y_max)

            # 图例（同时解决标记看不清的问题，兼容所有版本）
            if i == 0:
                axs[i].legend(
                    loc='best',
                    fontsize=12,
                    framealpha=0.8,
                    markerscale=15  # 只放大图例里的点，原图散点不变
                )

        fig.suptitle(bearing_label_map[filename], fontsize=22, fontweight='bold', y=0.98)
        plt.tight_layout(rect=(0.02, 0.02, 0.98, 0.95), h_pad=5.0, w_pad=3.0)

        save_name = bearing_label_map[filename]
        save_path = os.path.join(save_dir, f'{save_name}.png')
        pure_data_path = os.path.join(Pure_data_dir, f'{save_name}.npz')
        np.savez(pure_data_path, pure_data=pure_data)
        plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)

    except Exception as e:
        print(f"文件 {filename} 处理失败: {e}")

print("\n轴承拟合图绘制完成！")

# --------------------------
# 处理齿轮数据
# --------------------------
for filename in tqdm(gear_label_map.keys(), desc="绘制齿轮拟合曲线"):
    try:
        file_path = f'../DataSets/gear_set/{filename}'
        data = pd.read_csv(
            file_path,
            sep=r'\s+|,',
            usecols=range(8),
            skiprows=16,
            engine='python'
        ).values
        dataset = data[:, [1, 2, 3, 5, 6, 7]]

        pure_data = []
        poly_data = []

        for i in range(dataset.shape[1]):
            x = np.arange(len(dataset))
            y = dataset[:, i]
            pred = fast_sliding_fit(x, y, window_size=200, step=5)
            poly_data.append(pred)
            pure_signal = y - pred
            pure_data.append(pure_signal)

        pure_data = np.array(pure_data).T
        poly_data = np.array(poly_data).T
        x_plot = np.arange(len(dataset))

        fig, axs = plt.subplots(3, 2, figsize=(16, 10))
        axs = axs.flatten()

        for i in range(6):
            # 绘制散点和拟合线
            axs[i].scatter(x_plot, dataset[:, i], color='#1f77b4', s=0.2, alpha=0.4, label='Original')
            axs[i].plot(x_plot, poly_data[:, i], color='#ff4b33', linewidth=0.6, label='Fitted Trend')

            # 设置标题和刻度
            axs[i].set_title(f"Channel {index[i]}", fontsize=18)
            axs[i].tick_params(axis='both', labelsize=14)

            # 【你的核心要求】Y轴范围 = 数据实际的min和max，信号撑满整个子图
            y_data = dataset[:, i]
            y_min = y_data.min()
            y_max = y_data.max()
            axs[i].set_ylim(y_min, y_max)

            # 图例（同时解决标记看不清的问题，兼容所有版本）
            if i == 0:
                axs[i].legend(
                    loc='best',
                    fontsize=12,
                    framealpha=0.8,
                    markerscale=15  # 只放大图例里的点，原图散点不变
                )

        fig.suptitle(gear_label_map[filename], fontsize=22, fontweight='bold', y=0.98)
        plt.tight_layout(rect=(0.02, 0.02, 0.98, 0.95), h_pad=5.0, w_pad=3.0)

        save_name = gear_label_map[filename]
        save_path = os.path.join(save_dir, f'{save_name}.png')
        pure_data_path = os.path.join(Pure_data_dir, f'{save_name}.npz')
        np.savez(pure_data_path, pure_data=pure_data)
        plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)

    except Exception as e:
        print(f"文件 {filename} 处理失败: {e}")

print("\n齿轮拟合图绘制完成！")