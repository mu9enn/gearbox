# Project Codebase Knowledge

生成时间：2026-06-28  
工作目录：`/Volumes/mobile-disk/1/GearBox`

## 0. 初始审计记录

| 项目 | 记录 |
| --- | --- |
| 当前目录 | `/Volumes/mobile-disk/1/GearBox` |
| Git 状态 | `fatal: not a git repository`，当前目录及父目录到 `/Volumes` 边界内未发现 `.git` |
| 顶层目录 | `Analysis`, `Common`, `CWRU`, `CalculateShapValues`, `DAG`, `DataSets`, `Frame`, `GNNCausal`, `ModelTrain`, `PC_Datasets`, `wavelet_dataset` |
| 入口/脚本数量 | 约 78 个 Python 脚本，无 shell、notebook、README、requirements、pyproject、Dockerfile |
| 大文件概况 | `CalculateShapValues` 约 40G，`wavelet_dataset` 约 18G，`DataSets` 约 15G，`ModelTrain` 约 4.6G，`DAG` 约 2.2G |

毕业论文《耦合SHAP归因与双视角因果图的旋转机械故障诊断逻辑审计研究》在本轮只作为研究背景和术语参考；本文件的判断依据是当前项目目录中的代码、脚本、数据契约和结果文件。

## 1. 一句话定位

当前项目是一个面向旋转机械故障诊断的实验代码库：以 SEU 齿轮箱/轴承振动数据和 CWRU 轴承数据为对象，构建角域重采样 + 小波多尺度张量，训练 CNN/attention/baseline/GNN 诊断模型，用 SHAP 归因提取通道与频带重要性，并用 PC-KCI、背景知识和虚拟 do 干预形成因果图或可靠因果边，最终服务于“诊断模型是否依赖高归因但弱因果特征/捷径特征”的逻辑审计与 SCI 论文实验链。

与毕业论文 topic 的关系：

- 继承部分：SHAP 归因、通道/频带双视角因果图、旋转机械故障诊断逻辑审计。
- 当前项目扩展：加入 SEU 与 CWRU 两套数据；加入去趋势/纯净信号、角域重采样、小波张量、attention 与 no-attention 对比、4 通道/6 通道/频带裁剪消融、跨工况测试、PC-KCI + 虚拟 do 干预筛边、因果 GNN。
- 最可能支撑 SCI 一区的主线：从“高准确率故障诊断”升级到“可解释归因与因果可靠性一致性的诊断逻辑审计”，并进一步把可靠因果边注入 GNN 验证泛化和抗捷径能力。

## 2. 顶层目录与模块地图

```text
GearBox/
  Analysis/                 # 后处理分析，主要对 SHAP 与 attention 通道权重做相关性
  Common/                   # 核心公共代码：数据加载、模型、训练、PC/因果、GNN 训练
  CWRU/                     # CWRU 数据小波预处理与 GNN 跨工况实验
  CalculateShapValues/      # SHAP 计算脚本与多 seed 结果
  DAG/                      # 去趋势数据、PC-DAG 绘图、通道/频带因果图产物
  DataSets/                 # 原始/中间数据：SEU、CWRU、XJTU-SY
  Frame/                    # 早期/导出模型框架和 ONNX 文件
  GNNCausal/                # SEU 因果边驱动的 GNN 训练与跨工况测试
  ModelTrain/               # CNN/attention/no-attention/baseline/消融训练与评估
  PC_Datasets/              # PC 因果发现用结构化特征表及图
  wavelet_dataset/          # SEU 小波张量数据、PC 干预结果、稳定边结果
```

关键职责：

- 核心源码：`Common/LoadDatasets.py`, `Common/NetWorkFrame.py`, `Common/ModelTrainAndVisiable.py`, `Common/GNNTrain.py`, `Common/CausalAndDoWhy.py`, `Common/PCCausal.py`。
- 实验入口：`ModelTrain/**.py`, `CalculateShapValues/**.py`, `DAG/WaveletDAG.py`, `DAG/data_treat.py`, `GNNCausal/CausalGNN.py`, `CWRU/ModelTrainAndCrossTest.py`, `Common/PCDatasets.py`, `Common/PCDataset(wavelet).py`。
- 数据/缓存/结果：`DataSets/`, `wavelet_dataset/`, `DAG/PureData/`, `DAG/Noise/`, `PC_Datasets/`, `CalculateShapValues/*/Seed_*`, `ModelTrain/**/SavedModels*`, `ModelTrain/**/SavedGraphs*`, `GNNCausal/trained_models`, `CWRU/trained_models`。
- 可能历史遗留/需确认：`Frame/` 与 ONNX 导出物、`DAG/CreateDAG.py` 中被注释掉的 `GearBoxCausalAnalyzer` 依赖、`Analysis/test.py`, `DAG/test.py`, `GNNCausal/analysis.py`, `GNNCausal/trained_models/GNN/analysis.py`。

## 3. 架构与依赖关系

主要 import 关系：

- `ModelTrain/*` 依赖 `GearBox.Common.ModelTrainAndVisiable` 和 `GearBox.Common.NetWorkFrame`。
- `CalculateShapValues/*` 依赖 `GearBox.Common.NetWorkFrame`、训练好的 `.pth` 模型和 `wavelet_dataset`。
- `Common/PCDataset(wavelet).py` 依赖 `CalculateShapValues/NoAttention/Seed_*` 和 `wavelet_dataset`，生成 `PC_Datasets/Seed_49/*.csv`。
- `Common/PCDatasets.py` 依赖 `DAG/PureData/*.npz` 与 `ModelTrain/AttentionAndNoAttention/ChannelWeights/ChannelWeights.npz`，生成 `PC_Datasets/*.csv`。
- `DAG/WaveletDAG.py` 依赖 `PC_Datasets/Seed_49/*.csv` 和多 seed SHAP，调用 `Common.PCCausal.WaveletDAG` 绘制通道/频带 DAG。
- `Common/CausalAndDoWhy.py` 读取 `wavelet_dataset/*.npz`，生成 `GNNCausal/*_reliable_edge.pt`。
- `GNNCausal/CausalGNN.py` 读取 `wavelet_dataset/*.npz` 与 `*_reliable_edge.pt`，训练 `GNNCausal/trained_models/GNN/*.pth`，并调用 `GNNCausal/cross_cond_test.py`。
- `CWRU/ModelTrainAndCrossTest.py` 读取 `DataSets/CWRU/WaveletDataset/all_cond_wave.npz`，复用 `Common.GNNTrain` 和 `Common.NetWorkFrame.GNNCausalCWRU`。

两套主要 pipeline：

1. SEU 小波 CNN/XAI/双视角因果图路线  
   `DataSets/SEU/*.csv` -> `Common/LoadDatasets.py` -> `wavelet_dataset/*.npz` -> `ModelTrain/NoAttention` 或 `ModelTrain/AttentionAndNoAttention` -> `CalculateShapValues/*` -> `Common/PCDataset(wavelet).py` -> `PC_Datasets/Seed_49/*.csv` -> `DAG/WaveletDAG.py` -> `DAG/PC_DAG/Seed_49/**.png`。

2. 可靠因果边 + GNN 路线  
   `wavelet_dataset/*.npz` -> `Common/CausalAndDoWhy.py` 中 30 个物理小波节点 x 4 统计量 -> PC-KCI + 背景知识 + 虚拟 do 干预 -> `GNNCausal/*_reliable_edge.pt` -> `GNNCausal/CausalGNN.py` -> `GNNCausal/trained_models/GNN/*.pth` 与跨工况测试。

还有 CWRU 路线：

`DataSets/CWRU/**/*.mat` -> `CWRU/WT.py` -> `DataSets/CWRU/WaveletDataset/all_cond_wave.npz` -> `CWRU/ModelTrainAndCrossTest.py` -> `CWRU/trained_models/GNN/*.pth`, `CWRU/train_curves/GNN/*.png`。

## 4. 环境与依赖

未发现 `requirements.txt`, `environment.yml`, `pyproject.toml`, `setup.py`, Dockerfile 或 shell 脚本。根据 import 静态恢复的主要依赖：

| 类别 | 依赖 |
| --- | --- |
| 深度学习 | `torch`, `torch.nn`, `torch.optim`, `torch.utils.data`, `torchsummary` |
| 数据处理 | `numpy`, `pandas`, `scipy`, `sklearn` |
| 信号处理 | `pywt`, `scipy.signal`, `scipy.interpolate` |
| XAI | `shap` |
| 因果发现 | `causallearn`, `dowhy`, `pgmpy`, `networkx` |
| 调参/训练辅助 | `optuna`, `tqdm` |
| 绘图 | `matplotlib` |

Python 版本未在项目中声明。当前运行环境使用 Python 3.13，但这不等于项目支持版本；`torch`, `shap`, `causallearn`, `pgmpy` 对 Python 版本较敏感，建议后续锁定 Python 3.10/3.11 并生成依赖文件。

CUDA/GPU：训练脚本普遍使用 `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`，GPU 不是语法必需，但 SHAP 与大规模训练在 CPU 上成本很高。脚本多次设置 `torch.cuda.manual_seed*`，未检查 CUDA 是否可用；通常不会阻塞，但环境兼容性需验证。

硬编码/平台风险：

- 大量相对路径使用 Windows 分隔符，如 `..\\wavelet_dataset`, `.\\trained_models\\GNN`。
- 部分脚本要求从特定目录启动，否则相对路径失效。
- 代码以 `from GearBox...` 绝对包路径导入，但项目根目录当前没有安装包配置；需要从项目父目录运行或配置 `PYTHONPATH`。

## 5. 数据与文件契约

SEU 原始数据：

- 路径：`DataSets/SEU/bearing_set/*.csv`, `DataSets/SEU/gear_set/*.csv`。
- 读取：`Common/LoadDatasets.py:DataSets.load_data(skip_rows=16, use_cols=8)`。
- 通道：代码主要选择原始列 `[1,2,3,5,6,7]`，在图和特征中命名为 `ch2,ch3,ch4,ch6,ch7,ch8`。
- 工况/故障：`Bearing_20_0`, `Bearing_30_2`, `Gear_20_0`, `Gear_30_2`；故障包括 `ball`, `comb`, `bearing_normal`, `inner`, `outer`, `chipped`, `gear_normal`, `miss`, `root`, `surface`。

SEU 小波数据：

- 生成：`Common/LoadDatasets.py:DataSets.wavelet_transform`。
- 文件：`wavelet_dataset/Bearing_20_0.npz`, `Bearing_30_2.npz`, `Gear_20_0.npz`, `Gear_30_2.npz`。
- 键：`train_set`, `valid_set`, `test_set`, `cross_finetune_set`, `cross_test_set`, 对应 label 键和 `label_mapping`。
- 形状示例：`Bearing_20_0.npz` 中 `train_set=(5455,6,6,1024)`, `valid_set=(510,6,6,1024)`, `test_set=(680,6,6,1024)`, `cross_test_set=(6140,6,6,1024)`；`30_2` 工况为 `train_set=(8190,6,6,1024)`。
- 小波层：`pywt.swt(..., wavelet='db4', level=5)`，输出 6 层/系数维；部分后续脚本会取 `[:, :, 1:, :]` 或 `0:-1`。

划分策略：

- `Common/LoadDatasets.py` 对每个故障按原始时序切分：前 80% train，后续 10% valid，最后 10% test。
- `cross_finetune_set` 是全局随机连续 10% 片段，允许与训练时间段重叠；`cross_test_set` 为其余片段。这里存在潜在跨工况/微调泄露讨论点，需要论文中明确。
- 随机片段使用 `np.random.randint`，`LoadDatasets.py` 本身没有在入口处固定 seed。

SHAP 数据：

- `CalculateShapValues/NoAttention/Seed_*/<Condition>.npz`：键 `shap_values`, 形状示例 `(500,6,6,512,5)`。
- `CalculateShapValues/AttentionAndNoAttention/Seed_*/*_ShapValues.npz`：键 `attention`, `no_attention`, 形状示例 `(250,6,6,1024,20)`。
- `ModelTrain/NoAttention/SHAP_freq/ShapResults/*.npz`：键 `results`, 形状示例 `(1025,4,6,1024,5)`。
- 背景样本与解释样本：NoAttention 通常 `background_sample_number=100`, `explainable_number_every_fault=100`；AttentionAndNoAttention 通常每故障 50。

PC 数据：

- `PC_Datasets/Seed_49/*.csv`：列 `ch2,ch3,ch4,ch6,ch7,ch8,cA5,cD5,cD4,cD3,cD2,label`。
- `PC_Datasets/PC_Datasets*.csv`：列 `mean,std,rms,peak,crest_factor,kurtosis,skewness,mfe,ebe,cf,thd,label,condition,fault`。
- `Common/PCDatasets.py` 中频域参数硬编码：`30_2` 使用啮合频率 `[313.2,870]`、转频 `30`；`20_0` 使用 `[208.8,580]`、转频 `20`。

可靠因果边：

- `Common/CausalAndDoWhy.py` 将 `wavelet_dataset/*.npz` 取 `[:, :, 1:, :]`，reshape 为 `(N,30,1024)`。
- 每个节点提取 4 个统计特征：方差、峰峰值、峭度、冲击因子，拼成 `(N,120)`。
- PC-KCI 输出经混杂候选和虚拟 do 干预验证后保存为 `GNNCausal/*_reliable_edge.pt`。

## 6. 主要运行入口

| 入口 | 用途 | 输入 | 输出 | 轻量安全性 | 状态 |
| --- | --- | --- | --- | --- | --- |
| `Common/LoadDatasets.py` | SEU CSV -> 小波 `.npz` | `DataSets/SEU/*/*.csv` | `wavelet_dataset/*.npz` | 高成本，不建议本轮运行 | 核心预处理 |
| `DAG/data_treat.py` | 去趋势/拟合纯净信号 | `DataSets/*` 路径当前疑似少了 `SEU` 层 | `DAG/PureData`, `DAG/Noise`, `DAG/PolyCurves` | 高成本 | 需确认路径 |
| `ModelTrain/NoAttention/model_train_6ch/*.py` | 6 通道 CNN 训练 | `wavelet_dataset/*.npz` | `SavedModels_6ch`, `SavedGraphs_6ch` | 训练，高成本 | 主 baseline |
| `ModelTrain/NoAttention/model_train_4ch/*.py` | 4 通道裁剪训练 | `wavelet_dataset/*.npz` | `SavedModels_4ch`, `SavedGraphs_4ch` | 训练，高成本 | ablation |
| `ModelTrain/NoAttention/model_train_freq/*.py` | 频带裁剪训练 | `wavelet_dataset/*.npz` | `SavedModels_freq`, `SavedGraphs_freq` | 训练，高成本 | ablation |
| `ModelTrain/NoAttention/BaseModelTrain/*.py` | 通道+频带裁剪 baseline | `wavelet_dataset/*.npz` | `BaseModels`, `BaseGraphs` | 训练，高成本 | ablation |
| `ModelTrain/AttentionAndNoAttention/*.py` | attention 与 no-attention 并行训练 | `wavelet_dataset/wavelet_dataset.npz` | `SavedModels`, `SavedGraphs`, `SavedWeights` | 训练，高成本 | 部分入口可能缺输入 |
| `ModelTrain/AttentionAndNoAttention/ChannelWeights.py` | 学习全局通道权重 | `DAG/PureData` | `ChannelWeights.npz` | 优化/数据处理 | 核心辅助 |
| `CalculateShapValues/NoAttention/*.py` | no-attention SHAP | 模型 `.pth` + 小波数据 | 多 seed SHAP `.npz` | 高成本 | 已有结果 |
| `CalculateShapValues/AttentionAndNoAttention/*.py` | attention/no-attention SHAP 对照 | 两类模型 + 小波数据 | `*_ShapValues.npz` | 极高成本 | 已有结果 |
| `Common/PCDataset(wavelet).py` | SHAP 聚合到通道/频带 PC 节点 | no-attention SHAP + 小波数据 | `PC_Datasets/Seed_49/*.csv` | 读大 SHAP，高成本 | 双视角 DAG 前置 |
| `DAG/WaveletDAG.py` | 通道/频带 PC-KCI DAG 绘制 | `PC_Datasets/Seed_49`, SHAP | `DAG/PC_DAG/Seed_49/**.png` | 中高成本 | 论文图候选 |
| `Common/CausalAndDoWhy.py` | 可靠因果边生成 | `wavelet_dataset/*.npz` | `GNNCausal/*_reliable_edge.pt` | PC-KCI 很耗时 | GNN 前置 |
| `GNNCausal/CausalGNN.py` | 因果 GNN 训练与跨工况测试 | 小波数据 + reliable edges | `GNNCausal/trained_models`, `train_curves` | 训练，高成本 | 扩展主线 |
| `CWRU/WT.py` | CWRU `.mat` 小波预处理 | `DataSets/CWRU/**/*.mat` | `all_cond_wave.npz` | 高成本 | CWRU 支线 |
| `CWRU/ModelTrainAndCrossTest.py` | CWRU 因果 GNN 训练/跨工况 | `all_cond_wave.npz` | CWRU 模型/曲线 | 训练，高成本 | 支线 |

推荐后续复现实验入口优先级：

1. 先修正/统一数据键名与大小写，再以 `Common/LoadDatasets.py` 作为 SEU 预处理入口。
2. 以 `ModelTrain/NoAttention/model_train_6ch/*.py` 作为 CNN baseline。
3. 以 `CalculateShapValues/NoAttention/*.py` + `Common/PCDataset(wavelet).py` + `DAG/WaveletDAG.py` 支撑双视角审计图。
4. 以 `Common/CausalAndDoWhy.py` + `GNNCausal/CausalGNN.py` 支撑因果图注入 GNN 的新颖性。

## 7. 模型与算法实现

| 算法/模块 | 路径 | 输入输出 | 关键超参数 | 状态 |
| --- | --- | --- | --- | --- |
| 角域重采样 | `Common/LoadDatasets.py:angle_resample` | 时域振动 + rpm -> 角域信号 | `fs=5120`, `angle_fs=256` | 已实现 |
| 小波变换 | `Common/LoadDatasets.py:wavelet_function` | 窗口信号 -> `(6,6,1024)` | `db4`, `level=5`, 窗口 1024 | 已实现 |
| CNN no-attention | `Common/NetWorkFrame.py:CNNNetWorkNoAttention` | `(N,6,6,512/1024)` -> 类别 | Conv2d 6->12, maxpool, FC | 主 baseline，输入长度存在脚本差异 |
| SC attention CNN | `SCAttention2D`, `CNNNetWorkWithAttention` | 同上 | 通道归一化注意力 + 空间注意力 | 实现，但部分 SHAP 脚本调用参数与类签名不一致 |
| 4 通道模型 | `BaseModel4Channel` | `(N,4,6,1024)` | Conv2d 4->8 | ablation |
| 频带裁剪模型 | `BaseModelFreq` | `(N,6,4,1024)` | Conv2d 6->12 | ablation |
| 通道+频带裁剪 | `BaseModel` | `(N,4,4,1024)` 等 | `layer_number` | ablation |
| 训练器 | `Common/ModelTrainAndVisiable.py` | dataset dict -> 曲线 + `.pth` | lr `0.00015`, batch `256`, epoch `50` | 已实现 |
| SHAP | `CalculateShapValues/*` | 模型 + background + explain samples | GradientExplainer, background 100 | 已实现，结果很大 |
| PC-KCI 双视角 DAG | `Common/PCCausal.py:WaveletDAG` | 通道/频带 CSV -> DAG | KCI, alpha 0.05, BK 禁止 label->feature | 已实现 |
| PC-KCI + do 干预筛边 | `Common/CausalAndDoWhy.py` | `(N,120)` 统计特征 -> `.pt` edge index | alpha 0.05, RF 200 trees | 已实现 |
| 因果 GNN | `Common/NetWorkFrame.py:GNNCausalSEU`, `GNNTrainSEU` | `(N,30,4)` -> 类别 | lr 1e-4, batch 256, stage2 80 | 半成品风险：`GNNCausalSEU.forward` 当前分类器使用 `x` 而非融合后的 `out` |
| CWRU GNN | `GNNCausalCWRU`, `GNNTrainCWRU` | CWRU 小波节点 -> 类别 | epoch 300, batch 128 | 支线 |

## 8. 配置与超参数

项目没有集中配置系统，参数分散在脚本中。

关键硬编码参数：

- 数据读取：`skip_rows=16`, `use_cols=8`, 选通道 `[1,2,3,5,6,7]`。
- SEU 工况：`20_0` rpm 1200，`30_2` rpm 1800。
- 窗口：预处理 `segment_length=1024`；train/test step 常见 `768`，valid/finetune step `1024`；PC 特征 `window_length=512`, `step=512`。
- 小波：`db4`, `level=5`。
- CNN 训练：lr `0.00015`, batch `256`, epoch `50`，早停阈值约 `0.005/0.01/0.015`。
- GNN 训练：lr `0.0001`, batch `256` 或 `128`，SEU `stage2_epoch=80`，CWRU `epoch_sum=300`。
- SHAP：background `100`，每类解释样本 no-attention `100`，attention/no-attention `50`。
- 因果发现：PC-KCI `alpha=0.05`，`max_k=6` 或 `max_cond_vars=len(cols)-2`。
- 频域物理参数：`30_2`: `[313.2,870]`, `fr=30`; `20_0`: `[208.8,580]`, `fr=20`。
- seed：训练脚本常见 `42,49,56,63,70`，但预处理与部分脚本未全局统一。

## 9. 测试、验证与调试

未发现正式单元测试、集成测试、smoke test、CI、preflight 或依赖校验。现有验证主要是：

- 训练过程曲线图：`SavedGraphs*`, `train_curves`。
- 混淆矩阵/F1 图：`ModelTrain/**/ConfusionAndF1`。
- 跨工况准确率图：`ModelTrain/NoAttention/CrossCondTest/cross_condition.py`。
- 手工分析脚本：`Analysis/test.py`, `DAG/test.py`, `PC_Datasets/analysis.py`。

最低成本补充建议：

1. 添加 `scripts/preflight.py`：检查 Python 版本、依赖 import、关键路径存在性和大小写一致性。
2. 添加一个 `tests/test_npz_contract.py`：只读 `.npz` header，验证键名和形状，不加载大数组。
3. 添加 `tests/test_imports.py`：确保 `Common` 核心模块可导入。
4. 添加 `docs/REPRODUCE_MINIMAL.md`：明确从预处理到图表的命令顺序。

## 10. 风险、债务与可维护性

| 严重程度 | 问题 | 影响 | 证据路径 | 建议 |
| --- | --- | --- | --- | --- |
| 高 | 当前不是 Git 仓库 | 无法判断已跟踪大文件/敏感文件，GitHub 上传需先初始化 | `git status --short` 报错 | 初始化 Git 前先应用 `.gitignore`，再 `git status` |
| 高 | 无依赖文件 | 复现实验困难 | 未发现 `requirements.txt/pyproject` | 生成锁版本环境文件 |
| 高 | 数据/结果极大 | GitHub 普通仓库不可承受 | `CalculateShapValues` 40G，单 SHAP 2.8G，`wavelet_dataset/*.npz` 3.7-5.6G | 默认忽略，必要时 LFS/外部存储 |
| 高 | 文件命名大小写不一致 | Linux/CI/克隆后路径失效 | `Gear_30_2.npz` vs `gear_30_2.npz`; `train_set` vs `train_data` | 统一数据契约并迁移脚本 |
| 高 | `DAG/CreateDAG.py` 入口疑似不可运行 | 旧结果难复现 | import `GearBoxCausalAnalyzer`，但 `PCCausal.py` 中类被注释 | 标记历史遗留或恢复类 |
| 高 | `CNNNetWorkWithAttention` 调用签名不一致 | SHAP/训练脚本可能报错 | `CalculateShapValues/AttentionAndNoAttention/*.py` 传 `ratio=weights`，类未定义 `ratio` 参数 | 修正类或脚本 |
| 高 | `GNNCausalSEU.forward` 可能未使用因果融合结果 | 方法贡献可能被削弱 | `Common/NetWorkFrame.py` 中计算 `out` 后 `logits = self.classifier(x)` | 人工确认是否应为 `classifier(out)` |
| 中 | 相对路径/Windows 分隔符大量存在 | 跨平台复现差 | 多脚本 `..\\wavelet_dataset` | 使用 `pathlib` 和项目根定位 |
| 中 | 预处理随机微调片段允许与训练重叠 | 可能引入数据泄露质疑 | `Common/LoadDatasets.py` 注释说明允许重叠 | 论文中解释，或改为不重叠 |
| 中 | 结果文件与入口关系不总是清楚 | 图表可追溯性不足 | 多个 `analysis.py/test.py` | 给每个结果写 manifest |
| 中 | 大量重复脚本按工况复制 | 修改易漏 | `ModelTrain/**/Gear_*.py/Bearing_*.py` | 后续抽象为参数化 CLI |
| 中 | 参数分散且无配置 | 消融难控 | 全项目硬编码 lr/seed/path | 引入 `configs/*.yaml` |
| 中 | 多 seed 不完整覆盖 | 统计显著性证据不足 | Attention 仅 Seed 42/49/56，NoAttention 42/49/56/63/70 | 统一 seed 表与统计表 |
| 低 | `__pycache__` 已存在 | 仓库污染 | `Common/__pycache__`, `GNNCausal/__pycache__` | `.gitignore` 忽略 |
| 低 | 论文图 PNG/PDF 分散 | 发表材料整理成本高 | `DAG/PC_DAG`, `PC_Datasets/*.png`, `ConfusionAndF1` | 后续汇总到 `docs/figures` 或 `paper/figures` |

