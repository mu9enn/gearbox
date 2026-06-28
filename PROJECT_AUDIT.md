# GearBox Project Audit

生成时间：2026-06-28  
目标：冻结论文主线与实验矩阵，不改算法逻辑。  
证据原则：所有判断优先来自当前代码、脚本和结果路径；不确定处标注“需要人工确认”。

## 0. 仓库现状

- 当前目录：`/Volumes/mobile-disk/1/GearBox`。
- 当前目录不是 Git 仓库：`git status --short` 返回 `fatal: not a git repository`。
- 未发现 `requirements.txt`, `environment.yml`, `pyproject.toml`, `setup.py`, Dockerfile 或 shell 脚本。
- 主要代码目录：`Common/`, `ModelTrain/`, `CalculateShapValues/`, `DAG/`, `GNNCausal/`, `CWRU/`。
- 主要数据/结果目录：`DataSets/`, `wavelet_dataset/`, `PC_Datasets/`, `ModelTrain/**/Saved*`, `CalculateShapValues/**/Seed_*`, `DAG/PC_DAG`, `GNNCausal/trained_models`, `CWRU/trained_models`。

## 1. 数据流

### 1.1 原始数据读取

入口脚本：

- `Common/LoadDatasets.py`
- `DAG/data_treat.py`
- `CWRU/WT.py`

核心模块：

- `Common/LoadDatasets.py:DataSets.load_data` 用 `pd.read_csv(skiprows=16, usecols=range(8), sep=r'\s+|,')` 读取 SEU CSV（`Common/LoadDatasets.py:18-25`）。
- `Common/LoadDatasets.py:DataSets.wavelet_transform` 从原始 8 列中选 `[1,2,3,5,6,7]` 六个通道（`Common/LoadDatasets.py:116-118`）。
- `CWRU/WT.py` 读取 CWRU `.mat`，输出 `DataSets/CWRU/WaveletDataset/all_cond_wave.npz`，属于外部验证支线。

数据进入路径：

- SEU 原始：`DataSets/SEU/bearing_set/*.csv`, `DataSets/SEU/gear_set/*.csv`。
- CWRU 原始：`DataSets/CWRU/**/**/*.mat`。
- XJTU-SY 原始：`DataSets/XJTU-SY/.../*.csv`，当前未见主线训练脚本调用，暂列未来拓展/数据储备。

### 1.2 预处理与小波张量

主数据流：

```text
DataSets/SEU/*.csv
  -> DataSets.load_data
  -> angle_resample
  -> window_data
  -> wavelet_function
  -> wavelet_dataset/*.npz
```

关键证据：

- 角域重采样：`Common/LoadDatasets.py:DataSets.angle_resample`，默认 `fs=5120`, `angle_fs=256`（`Common/LoadDatasets.py:70-99`）。
- 小波：`pywt.swt(..., wavelet='db4', level=5)`，输出 `(n,6,6,1024)`（`Common/LoadDatasets.py:51-68`）。
- 划分：每类故障按时序 80% train、10% valid、10% test（`Common/LoadDatasets.py:123-128`）。
- cross-finetune：随机截取总长 10% 连续片段，注释明确“允许和 train 重叠”（`Common/LoadDatasets.py:153-163`）。
- 保存键：`train_set`, `valid_set`, `test_set`, `cross_finetune_set`, `cross_test_set`, `train_label`, `valid_label`, `test_label`, `cross_finetune_label`, `cross_test_label`, `label_mapping`（`Common/LoadDatasets.py:211-222`）。

结果保存路径：

- `wavelet_dataset/Bearing_20_0.npz`
- `wavelet_dataset/Bearing_30_2.npz`
- `wavelet_dataset/Gear_20_0.npz`
- `wavelet_dataset/Gear_30_2.npz`

潜在数据泄露点：

- `cross_finetune_set` 随机片段允许与 train 时间段重叠，需要人工确认论文是否可接受该设定。
- `Common/PCDatasets.py` 对 20 类纯净数据先全量拼接后做全局 min/max，再生成 PC 特征（`Common/PCDatasets.py:286-293`），这对因果可视化可接受性需说明；若用于训练/评估，则有泄露风险。
- `Common/PCDatasets.py` 后续对全部 feature_val 做 `StandardScaler.fit_transform` 与 `MinMaxScaler.fit_transform`（`Common/PCDatasets.py:341-347`），未按 train/test 区分。

重复逻辑：

- `window_data` 在 `Common/LoadDatasets.py`, `Common/PCDatasets.py`, `ModelTrain/AttentionAndNoAttention/ChannelWeights.py` 中重复。
- 时域/频域统计特征在 `Common/PCDatasets.py` 与 `ModelTrain/AttentionAndNoAttention/ChannelWeights.py` 中重复。

路径硬编码：

- `Common/LoadDatasets.py:257` 使用 `folder_dir = '../DataSets/SEU'`。
- `Common/LoadDatasets.py:301-304` 固定输出四个 `.npz`。
- 多处 Windows 风格相对路径，如 `GNNCausal/CausalGNN.py:224-231`, `ModelTrain/NoAttention/CrossCondTest/cross_condition.py:25`。

## 2. 模型流

### 2.1 CNN 与 attention 模型

核心模块：

- `Common/NetWorkFrame.py:CNNNetWorkNoAttention`，6 通道 CNN baseline（`Common/NetWorkFrame.py:136-155`）。
- `Common/NetWorkFrame.py:CNNNetWorkWithAttention`，SC attention CNN（`Common/NetWorkFrame.py:96-134`）。
- `Common/NetWorkFrame.py:BaseModel4Channel`, `BaseModelFreq`, `BaseModel`，对应通道裁剪、频带裁剪、通道+频带裁剪（`Common/NetWorkFrame.py:157-224`）。
- `Common/ModelTrainAndVisiable.py:ModelTrain.train_model`，单模型训练器（`Common/ModelTrainAndVisiable.py:62-186`）。
- `Common/ModelTrainAndVisiable.py:ParallelTraining.model_train`，attention/no-attention 并行训练器（`Common/ModelTrainAndVisiable.py:292-495`）。

入口脚本：

- `ModelTrain/NoAttention/model_train_6ch/*.py`
- `ModelTrain/NoAttention/model_train_4ch/*.py`
- `ModelTrain/NoAttention/model_train_freq/*.py`
- `ModelTrain/NoAttention/BaseModelTrain/*.py`
- `ModelTrain/AttentionAndNoAttention/*.py`
- `ModelTrain/AttentionAndNoAttention/AllData_of_SixChannels_CompareAttentionWithNoAttention.py`

关键超参数：

- CNN 单模型：lr `0.00015`, batch `256`, epoch `50`（`Common/ModelTrainAndVisiable.py:78-82`）。
- attention 并行训练：lr `0.00015`, batch `256`, epoch `50`（`Common/ModelTrainAndVisiable.py:311-315`）。
- 早停：单模型依据 train acc 变化和 train/test 差异保存（`Common/ModelTrainAndVisiable.py:164-178`）；并行训练依据 attention/no-attention 多组 acc 差异保存（`Common/ModelTrainAndVisiable.py:458-481`）。

结果保存路径：

- `ModelTrain/NoAttention/SavedModels_6ch/Seed_*/*.pth`
- `ModelTrain/NoAttention/SavedModels_4ch/Seed_42/*.pth`
- `ModelTrain/NoAttention/SavedModels_freq/Seed_42/*.pth`
- `ModelTrain/NoAttention/BaseModels/Seed_42/*.pth`
- `ModelTrain/AttentionAndNoAttention/SavedModels/Seed_*/*.pth`
- `ModelTrain/**/SavedGraphs*/Seed_*/*.png`

需要人工确认：

- 多个脚本读取 `train_data/train_labels`，但当前 `Common/LoadDatasets.py` 保存的是 `train_set/train_label`。例如 `ModelTrain/NoAttention/CrossCondTest/cross_condition.py:34-35` 读取 `test_data/test_labels`。这说明存在两套数据契约或历史产物，需要冻结主线前统一确认。
- `CalculateShapValues/AttentionAndNoAttention/*.py` 里曾出现 `CNNNetWorkWithAttention(class_number=class_num, ratio=weights)` 调用；当前 `Common/NetWorkFrame.py:CNNNetWorkWithAttention.__init__` 只接收 `class_number`（`Common/NetWorkFrame.py:96-103`）。需要确认脚本是否可运行。

### 2.2 因果 GNN 模型

核心模块：

- `Common/NetWorkFrame.py:GNNCausalSEU`（`Common/NetWorkFrame.py:308-436`）。
- `Common/NetWorkFrame.py:GNNCausalCWRU`（`Common/NetWorkFrame.py:439-514`）。
- `Common/GNNTrain.py:GNNTrainSEU.forward`，SEU GNN 训练（`Common/GNNTrain.py:180-290`）。
- `Common/GNNTrain.py:GNNTrainCWRU.forward`，CWRU GNN 训练（`Common/GNNTrain.py:20-111`）。

入口脚本：

- `GNNCausal/CausalGNN.py`
- `GNNCausal/cross_cond_test.py`
- `CWRU/ModelTrainAndCrossTest.py`

风险点：

- `GNNCausalSEU.forward` 计算了三阶因果传播 `out1/out2/out3` 并得到 `out = out + x`，但最后使用 `logits = self.classifier(x)`（`Common/NetWorkFrame.py:385-436`）。如果目标是验证因果传播有效性，这里需要人工确认是否应为 `classifier(out)`；本轮不改算法。
- `GNNCausal/CausalGNN.py:141` 将 `matrix_file` 写死为四个 `Bearing_20_0_reliable_edge.pt`，疑似临时调试或未完成泛化逻辑。

## 3. 解释流

### 3.1 SHAP 计算

入口脚本：

- `CalculateShapValues/NoAttention/Bearing_20_0.py`
- `CalculateShapValues/NoAttention/Bearing_30_2.py`
- `CalculateShapValues/NoAttention/Gear_20_0.py`
- `CalculateShapValues/NoAttention/Gear_30_2.py`
- `CalculateShapValues/AttentionAndNoAttention/*.py`
- `ModelTrain/NoAttention/SHAP_freq/*.py`

核心逻辑：

- `shap.GradientExplainer(model, background_datasets)` 与 `explainer.shap_values(X=explain_datasets)`（`CalculateShapValues/NoAttention/Gear_30_2.py:59-67`）。
- 背景样本 `background_sample_number=100`，每类解释样本 `explainable_number_every_fault=100`（`CalculateShapValues/NoAttention/Gear_30_2.py:51-53`）。
- 背景样本从正常类 train/test 按 80/20 抽取（`CalculateShapValues/NoAttention/Gear_30_2.py:91-105`）。
- 解释样本每类从 train/test 按 80/20 抽取（`CalculateShapValues/NoAttention/Gear_30_2.py:107-127`）。

结果保存路径：

- `CalculateShapValues/NoAttention/Seed_42|49|56|63|70/*.npz`
- `CalculateShapValues/AttentionAndNoAttention/Seed_42|49|56/*_ShapValues.npz`
- `ModelTrain/NoAttention/SHAP_freq/ShapResults/*.npz`
- `CalculateShapValues/NoAttention/*_AnalysisGraphs/*.png`
- `CalculateShapValues/NoAttention/TotalAnalysisGraphs/*.png`

潜在争议：

- SHAP 背景与解释样本混合 train/test。若 SHAP 只用于事后解释可接受，但如果由此生成的因果数据再用于模型选择或 claim，需要明确不参与训练。
- SHAP 结果体量极大，原始 `.npz` 不适合 Git 跟踪；论文主线应冻结聚合表和最终图，而非原始数组。

### 3.2 SHAP 聚合到通道/频带

核心脚本：

- `Common/PCDataset(wavelet).py`

核心逻辑：

- 读取五个 seed：`seeds = [42,49,56,63,70]`（`Common/PCDataset(wavelet).py:49-58`）。
- 从 SHAP 计算通道归因 `axis=(0,2,3,4)` 和频带归因 `axis=(0,1,3,4)`（`Common/PCDataset(wavelet).py:62-68`）。
- 对 seed 求平均，得到通道和频带权重（`Common/PCDataset(wavelet).py:73-80`）。
- 输出 PC-DAG 特征列 `ch2,ch3,ch4,ch6,ch7,ch8,cA5,cD5,cD4,cD3,cD2,label`（`Common/PCDataset(wavelet).py:126-140`）。

需要人工确认：

- `Common/PCDataset(wavelet).py:103-104` 读取 `all_data/all_labels`，但 `Common/LoadDatasets.py` 当前保存的是 `train_set/test_set` 等键。该脚本可能依赖历史版 `wavelet_dataset`。

## 4. 因果流

### 4.1 双视角 PC-DAG

入口脚本：

- `DAG/WaveletDAG.py`
- `Common/PCCausal.py:WaveletDAG`

核心逻辑：

- 通道节点：`ch2,ch3,ch4,ch6,ch7,ch8`；频带节点：`cD2,cD3,cD4,cD5,cA5`；结果节点：`label`（`Common/PCCausal.py:90-95`）。
- 背景知识：禁止 `label -> feature`（`Common/PCCausal.py:97-110`）。
- 条件独立检验：KCI（`Common/PCCausal.py:128-130`）。
- PC 参数：`significance_level=0.05`, `stable=True`, `max_cond_vars=len(cols)-2`（`Common/PCCausal.py:132-151`）。
- 绘图：边宽由 SHAP 值映射，原始 DAG 与过滤 DAG 分别保存（`Common/PCCausal.py:155-280`）。

结果保存路径：

- `DAG/PC_DAG/Seed_49/original/Significance_0.05/*.png`
- `DAG/PC_DAG/Seed_49/filtered/Significance_0.05/*.png`
- `DAG/PC_DAG/GearBox_PC_DAG_*.png/pdf`

历史遗留风险：

- `DAG/CreateDAG.py` import `GearBoxCausalAnalyzer`，但当前 `Common/PCCausal.py` 中该类实现被整段注释，入口可能不可运行，需要人工确认。

### 4.2 可靠因果边与虚拟 do 干预

入口脚本：

- `Common/CausalAndDoWhy.py`

核心逻辑：

- 读取 `wavelet_dataset/*.npz` 的 `train_set/valid_set/test_set/cross_*`，丢弃一个小波层后 reshape 为 `(N,30,1024)`（`Common/CausalAndDoWhy.py:190-206`）。
- 统计特征：方差、峰峰值、峭度、冲击因子，reshape 为 `(N,120)`（`Common/CausalAndDoWhy.py:209-224`）。
- 背景知识：同通道要求边、跨通道禁止边（`Common/CausalAndDoWhy.py:23-63`）。
- PC-KCI：`alpha=p`, `stable=True`, `max_k=6`（`Common/CausalAndDoWhy.py:65-81`）。
- 混杂候选：`find_all_confounders`（`Common/CausalAndDoWhy.py:83-97`）。
- 虚拟 do：`RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)`，对分位点干预并检查单调趋势（`Common/CausalAndDoWhy.py:119-160`）。
- 保存可靠边：`torch.save(reliable_edge, save_p)`（`Common/CausalAndDoWhy.py:162-185`）。

结果保存路径：

- `GNNCausal/Bearing_20_0_reliable_edge.pt`
- `GNNCausal/Bearing_30_2_reliable_edge.pt`
- `GNNCausal/Gear_20_0_reliable_edge.pt`
- `GNNCausal/Gear_30_2_reliable_edge.pt`

## 5. 评估流

入口脚本：

- `ModelTrain/NoAttention/ConfusionMatrix.py`
- `ModelTrain/AttentionAndNoAttention/ConfusionMatrix.py`
- `ModelTrain/NoAttention/CrossCondTest/cross_condition.py`
- `ModelTrain/NoAttention/ModelTest_ch/*.py`
- `ModelTrain/NoAttention/ModelTest_freq/*.py`
- `GNNCausal/cross_cond_test.py`
- `CWRU/ModelTrainAndCrossTest.py`

核心指标：

- 训练/测试 accuracy/loss：`Common/ModelTrainAndVisiable.py:133-162`, `Common/ModelTrainAndVisiable.py:395-449`。
- confusion matrix 和 classification report：`ModelTrain/NoAttention/ConfusionMatrix.py` 与 attention 版本。
- 固定工况与跨工况准确率：`ModelTrain/NoAttention/CrossCondTest/cross_condition.py:76-132`。
- 模型参数量/推理时间：`ModelTrain/NoAttention/ModelTest_ch/*.py`, `ModelTrain/NoAttention/ModelTest_freq/*.py` 中有 `count_params`, `infer_time`，需要人工确认输出是否完整留存。

评估缺口：

- 大多数脚本只保存图，不保存 CSV/JSON metrics。
- 没有统一 mean/std 汇总表。
- 没有显著性检验脚本。
- 没有 config snapshot、git hash、依赖版本记录。

## 6. 结果流

主要结果路径：

| 类型 | 路径 | 说明 |
| --- | --- | --- |
| 小波张量 | `wavelet_dataset/*.npz` | 预处理核心产物，体量 GB 级 |
| 模型权重 | `ModelTrain/**/SavedModels*`, `GNNCausal/trained_models`, `CWRU/trained_models` | `.pth` 权重 |
| 训练曲线 | `ModelTrain/**/SavedGraphs*`, `GNNCausal/train_curves`, `CWRU/train_curves` | accuracy/loss 图 |
| SHAP 原始结果 | `CalculateShapValues/**/Seed_*`, `ModelTrain/NoAttention/SHAP_freq/ShapResults` | `.npz`，体量极大 |
| SHAP 汇总图 | `CalculateShapValues/NoAttention/*AnalysisGraphs`, `TotalAnalysisGraphs` | 论文候选图 |
| PC 数据表 | `PC_Datasets/*.csv`, `PC_Datasets/Seed_49/*.csv` | 因果发现输入 |
| DAG 图 | `DAG/PC_DAG/**/*.png/pdf` | 双视角因果图候选 |
| 可靠边 | `GNNCausal/*_reliable_edge.pt` | GNN 因果结构输入 |

结果管理问题：

- 没有统一 `results/` 根目录。
- 没有实验 manifest，无法从结果文件反查完整命令、配置、commit、依赖版本。
- 同一结果类型分散在 `ModelTrain`, `CalculateShapValues`, `DAG`, `GNNCausal`, `CWRU`。
- 部分文件为空或入口不清，例如 `wavelet_dataset/causal_edge_intervention_result.csv` 当前读 CSV 报空文件，需人工确认。

## 7. 建议冻结的论文主线

推荐冻结为一条主线：

```text
SEU raw vibration
  -> angle-domain + SWT wavelet tensor
  -> CNN baseline / ablations
  -> SHAP channel-frequency attribution
  -> dual-view PC-KCI causal graph
  -> attribution-causality consistency audit
  -> optional intervention-verified causal edges + causal GNN validation
```

不建议把所有目录都写进主方法。`CWRU/` 可作为外部验证；`XJTU-SY/` 当前无主线脚本证据，暂不纳入主实验。

