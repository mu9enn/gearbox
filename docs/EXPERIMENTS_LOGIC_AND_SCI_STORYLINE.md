# Experiments Logic And SCI Storyline

生成时间：2026-06-28  
依据：当前项目代码、数据契约、结果文件。毕业论文仅作为背景，不作为当前实验逻辑的直接证据。

## 1. 实验总览

| 编号 | 实验名称 | 入口 | 数据输入 | 方法 | 输出 | 可支撑结论 | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | SEU 小波数据构建 | `Common/LoadDatasets.py` | `DataSets/SEU/*/*.csv` | 角域重采样 + SWT 小波 | `wavelet_dataset/*.npz` | 统一多工况多尺度输入 | 已实现，重跑成本高 |
| E1 | 6 通道 CNN baseline | `ModelTrain/NoAttention/model_train_6ch/*.py` | `wavelet_dataset/*.npz` | `CNNNetWorkNoAttention` | `SavedModels_6ch`, `SavedGraphs_6ch` | 高性能诊断 baseline | 已有多 seed 结果 |
| E2 | 4 通道裁剪 | `ModelTrain/NoAttention/model_train_4ch/*.py` | 小波数据 | `BaseModel4Channel` | `SavedModels_4ch` | 通道裁剪消融 | 已实现 |
| E3 | 频带裁剪 | `ModelTrain/NoAttention/model_train_freq/*.py` | 小波数据 | `BaseModelFreq` | `SavedModels_freq` | 频带裁剪消融 | 已实现 |
| E4 | 通道+频带裁剪 baseline | `ModelTrain/NoAttention/BaseModelTrain/*.py` | 小波数据 | `BaseModel` | `BaseModels`, `BaseGraphs` | 双维裁剪消融 | 已实现 |
| E5 | 固定/跨工况泛化比较 | `ModelTrain/NoAttention/CrossCondTest/cross_condition.py` | E1-E4 模型 + test set | 准确率柱状图 | `Cross_condition_acc_comparison.png` | 特征裁剪对泛化影响 | 有脚本 |
| E6 | 混淆矩阵与 F1 | `ModelTrain/NoAttention/ConfusionMatrix.py`, `ModelTrain/AttentionAndNoAttention/ConfusionMatrix.py` | 模型 + test set | confusion matrix, precision/recall/F1 | `ConfusionAndF1/**` | 分类性能细粒度证明 | 已有结果目录 |
| E7 | attention vs no-attention | `ModelTrain/AttentionAndNoAttention/*.py` | `wavelet_dataset/wavelet_dataset.npz` | 并行训练两模型 | `SavedModels`, `SavedWeights`, `SavedGraphs` | attention 是否改变诊断和归因 | 有结果但入口输入需确认 |
| E8 | SHAP no-attention | `CalculateShapValues/NoAttention/*.py` | E1 模型 + 小波数据 | SHAP GradientExplainer | `CalculateShapValues/NoAttention/Seed_*/*.npz` | 通道/频带归因证据 | 已有大结果 |
| E9 | SHAP attention/no-attention 对照 | `CalculateShapValues/AttentionAndNoAttention/*.py` | E7 模型 | SHAP GradientExplainer | `*_ShapValues.npz` | attention 对归因分布影响 | 已有结果，脚本签名需确认 |
| E10 | SHAP 汇总图 | `CalculateShapValues/NoAttention/analysis.py`, `total_analysis.py` | E8 SHAP | 绝对 SHAP 均值聚合 | `TotalAnalysisGraphs/*.png` | 归因主导通道/频带 | 已实现 |
| E11 | SHAP/attention 权重相关性 | `Analysis/ChannelWeights.py` | E9 SHAP + `ChannelWeights.npz` | 相关系数 | 终端输出 | attention 权重与 SHAP 一致性 | 有脚本 |
| E12 | SHAP 加权 PC 数据集 | `Common/PCDataset(wavelet).py` | E8 SHAP + 小波数据 | 通道/频带能量加权 | `PC_Datasets/Seed_49/*.csv` | 双视角因果图输入 | 已实现 |
| E13 | 手工时频特征 PC 数据集 | `Common/PCDatasets.py` | `DAG/PureData` + 通道权重 | 11 维时频特征 | `PC_Datasets/*.csv` | 全局因果特征表 | 已实现 |
| E14 | 通道/频带双视角 PC-DAG | `DAG/WaveletDAG.py` + `Common/PCCausal.py` | E12 CSV + SHAP | PC-KCI + BK | `DAG/PC_DAG/Seed_49/**.png` | 高归因特征是否直接连到 label | 已实现 |
| E15 | 可靠因果边生成 | `Common/CausalAndDoWhy.py` | `wavelet_dataset/*.npz` | PC-KCI + 混杂检查 + 虚拟 do | `GNNCausal/*_reliable_edge.pt` | 因果可靠边集合 | 已实现 |
| E16 | 因果 GNN | `GNNCausal/CausalGNN.py` | 小波统计特征 + reliable edges | `GNNCausalSEU` | GNN 模型/曲线/跨工况 acc | 因果结构能否改善泛化 | 半成品风险需确认 |
| E17 | CWRU 小波/GNN 支线 | `CWRU/WT.py`, `CWRU/ModelTrainAndCrossTest.py` | CWRU `.mat` | 小波 + 因果 GNN | CWRU 模型/曲线 | 外部数据验证 | 有结果，需补文档 |
| E18 | 去趋势/纯净信号 | `DAG/data_treat.py` | SEU CSV | 滑动线性拟合去趋势 | `DAG/PureData`, `DAG/Noise` | 去趋势是否增强物理因果解释 | 已有结果，入口路径需确认 |

## 2. 当前真实技术路线

当前项目真实路线不是单纯“训练一个诊断模型”，而是：

1. 原始 SEU CSV 从 8 列中取 6 个振动通道。
2. 按工况转速做角域重采样，减少转速差导致的时域相位/频率偏移。
3. 按故障类别时序切分 train/valid/test，并额外抽取 cross-finetune/cross-test。
4. 对每个窗口做 SWT 小波，得到 `(样本, 通道, 小波层, 时间点)` 张量。
5. 训练多种 CNN baseline：全通道、4 通道、频带裁剪、通道+频带裁剪、attention/no-attention。
6. 用 SHAP 解释诊断模型，对通道和频带做绝对贡献度聚合。
7. 一条路线把 SHAP 权重转为通道/频带节点，做 PC-KCI 双视角 DAG，观察哪些高归因节点真正与 label 建立因果边。
8. 另一条路线把小波节点压缩成统计特征，做 PC-KCI + 背景知识 + 虚拟 do 干预，生成可靠边，再注入 GNN 做诊断与跨工况泛化。
9. 最终论文叙事应围绕“归因强度、因果连通性、模型泛化”三者的一致或冲突关系。

从 raw data 到 figures/tables：

```text
DataSets/SEU/*.csv
  -> Common/LoadDatasets.py
  -> wavelet_dataset/*.npz
  -> ModelTrain/NoAttention and AttentionAndNoAttention
  -> .pth models + train curves + confusion/F1
  -> CalculateShapValues/*
  -> SHAP npz + SHAP graphs
  -> Common/PCDataset(wavelet).py
  -> PC_Datasets/Seed_49/*.csv
  -> DAG/WaveletDAG.py
  -> Channel/Frequency DAG figures

wavelet_dataset/*.npz
  -> Common/CausalAndDoWhy.py
  -> GNNCausal/*_reliable_edge.pt
  -> GNNCausal/CausalGNN.py
  -> causal GNN models + curves + cross-condition scores
```

## 3. 实验卡片

### E0 SEU 小波数据构建

- 研究问题：如何把不同工况下的原始振动信号变成统一诊断输入。
- 假设：角域重采样 + SWT 小波能增强跨转速工况的可比性。
- 入口：`Common/LoadDatasets.py`。
- 调用链：`load_data` -> `angle_resample` -> `window_data` -> `wavelet_function` -> `np.savez`。
- 输入：`DataSets/SEU/bearing_set/*.csv`, `DataSets/SEU/gear_set/*.csv`。
- 输出：`wavelet_dataset/Bearing_20_0.npz`, `Bearing_30_2.npz`, `Gear_20_0.npz`, `Gear_30_2.npz`。
- 关键配置：`skip_rows=16`, `use_cols=8`, 通道 `[1,2,3,5,6,7]`, `db4`, `level=5`, train/valid/test = 80/10/10。
- 证据强度：强，有数据产物和明确代码。
- 风险：入口未固定 seed；cross-finetune 允许与 train 重叠。

### E1-E4 CNN baseline 与裁剪消融

- 研究问题：全特征、通道裁剪、频带裁剪、双裁剪对诊断性能和泛化有什么影响。
- 假设：高归因或因果可靠的特征子集可以在较低输入维度下保持性能，甚至改善跨工况。
- 入口：`ModelTrain/NoAttention/model_train_6ch/*.py`, `model_train_4ch/*.py`, `model_train_freq/*.py`, `BaseModelTrain/*.py`。
- 调用链：工况脚本 -> `ModelTrain.train_model` -> `Common.NetWorkFrame` 中模型 -> `.pth` 与曲线。
- 输入：`wavelet_dataset/*.npz`。
- 输出：`SavedModels_6ch`, `SavedModels_4ch`, `SavedModels_freq`, `BaseModels`, 对应图目录。
- 关键配置：lr `0.00015`, batch `256`, epoch `50`, seed 常见 `42/70`。
- 评价指标：train/test accuracy/loss，后续 confusion/F1 与 cross-condition accuracy。
- 证据强度：中到强，已有模型和曲线，但脚本存在大小写/键名不一致风险。

### E5 跨工况泛化比较

- 研究问题：模型在相同类型但不同转速/载荷工况间是否泛化。
- 入口：`ModelTrain/NoAttention/CrossCondTest/cross_condition.py`。
- 输入：`SavedModels_6ch/Seed_42`, `SavedModels_freq/Seed_42`, `BaseModels/Seed_42` 和 `wavelet_dataset/*.npz`。
- 输出：`Cross_condition_acc_comparison.png`。
- 关系：使用 E1-E4 的模型进行固定工况和交叉工况测试。
- 证据强度：中；脚本只生成图，未保存数值表，建议补 CSV。

### E6 混淆矩阵/F1

- 研究问题：总体准确率背后哪些故障类型容易混淆。
- 入口：`ModelTrain/NoAttention/ConfusionMatrix.py`, `ModelTrain/AttentionAndNoAttention/ConfusionMatrix.py`。
- 输出：`ConfusionAndF1/Seed_*/ConfusionMatrix/*.png`, `F1/*.png`。
- 指标：precision, recall, f1-score, normalized confusion matrix。
- 证据强度：中；建议保存 `classification_report.csv/json`。

### E7 Attention vs No-Attention

- 研究问题：attention 机制是否改变模型性能和归因分布。
- 入口：`ModelTrain/AttentionAndNoAttention/*.py` 与 `AllData_of_SixChannels_CompareAttentionWithNoAttention.py`。
- 输入：脚本写的是 `wavelet_dataset/wavelet_dataset.npz`，但当前顶层扫描未见该文件；已有结果目录存在。
- 输出：`SavedModels/Seed_42|49|56`, `SavedGraphs`, `SavedWeights`, `ChannelWeights`。
- 证据强度：中；需要人工确认缺失的统一 `wavelet_dataset.npz` 是否在其他位置或曾被删除。

### E8-E10 SHAP 归因

- 研究问题：诊断模型主要依赖哪些通道和频带。
- 入口：`CalculateShapValues/NoAttention/*.py`, `CalculateShapValues/NoAttention/analysis.py`, `total_analysis.py`。
- 输入：`SavedModels_6ch/Seed_*/*.pth`, `wavelet_dataset/*.npz`。
- 输出：多 seed SHAP `.npz` 和 `TotalAnalysisGraphs/*.png`。
- 关键配置：`GradientExplainer`, background 100, 每故障解释 100；结果示例 `(500,6,6,512,5)`。
- 证据强度：强，但文件极大；建议论文只跟踪聚合表/图，不跟踪原始 SHAP。

### E11 SHAP 与 attention 权重相关性

- 研究问题：attention 学到的通道权重是否与 SHAP 归因一致。
- 入口：`Analysis/ChannelWeights.py`。
- 输入：`CalculateShapValues/AttentionAndNoAttention/Seed_56/*` 与 `ModelTrain/AttentionAndNoAttention/ChannelWeights/ChannelWeights.npz`。
- 输出：当前主要为终端打印结果。
- 证据强度：弱到中；建议保存相关系数表并扩展到多 seed。

### E12-E14 双视角 PC-DAG

- 研究问题：高 SHAP 通道/频带是否在因果图中真正指向诊断 label。
- 入口：`Common/PCDataset(wavelet).py`, `DAG/WaveletDAG.py`, `Common/PCCausal.py:WaveletDAG`。
- 输入：NoAttention SHAP 多 seed、`wavelet_dataset/*.npz`。
- 输出：`PC_Datasets/Seed_49/*.csv`, `DAG/PC_DAG/Seed_49/original`, `filtered`。
- 方法：通道视角 `ch2,ch3,ch4,ch6,ch7,ch8`；频带视角 `cA5,cD5,cD4,cD3,cD2`；PC-KCI；背景知识禁止 `label -> feature`；边宽由 SHAP 控制。
- 证据强度：强，最贴近原 topic；建议补“高 SHAP 但无 label 因果边”的统计表。

### E15-E16 可靠因果边与 GNN

- 研究问题：经过因果可靠性筛选的结构能否提升或解释跨工况诊断。
- 入口：`Common/CausalAndDoWhy.py`, `GNNCausal/CausalGNN.py`, `GNNCausal/cross_cond_test.py`。
- 输入：`wavelet_dataset/*.npz`。
- 输出：`GNNCausal/*_reliable_edge.pt`, `GNNCausal/trained_models/GNN/*.pth`, `train_curves/GNN/*.png`。
- 方法：30 个物理小波节点，每节点 4 个统计量；PC-KCI + 机械背景知识；随机森林虚拟 do 干预判断单调趋势；GNN 传播因果边。
- 证据强度：有潜力但需人工确认；`GNNCausalSEU.forward` 目前计算了 `out` 但分类器用 `x`，可能没有真正使用因果传播结果。

### E17 CWRU 外部验证

- 研究问题：方法是否可迁移到公开 CWRU 轴承数据。
- 入口：`CWRU/WT.py`, `CWRU/ModelTrainAndCrossTest.py`。
- 输入：`DataSets/CWRU/**/*.mat`。
- 输出：`DataSets/CWRU/WaveletDataset/all_cond_wave.npz`, `CWRU/trained_models/GNN`, `CWRU/train_curves/GNN`。
- 证据强度：中；建议补与 SEU 主线一致的指标表。

### E18 去趋势/纯净信号

- 研究问题：去除慢变趋势后，时频/因果特征是否更稳定。
- 入口：`DAG/data_treat.py`。
- 输入：脚本写 `../DataSets/bearing_set` 和 `../DataSets/gear_set`，而当前数据在 `DataSets/SEU/...`，入口路径需确认。
- 输出：`DAG/PureData`, `DAG/Noise`, `DAG/PolyCurves`。
- 证据强度：中；已有结果，但重跑路径可能失效。

## 4. Baseline / Ablation / Comparison

当前已实现：

- 主诊断 baseline：6 通道 no-attention CNN。
- attention 对照：SC attention CNN vs no-attention CNN。
- 输入消融：4 通道、频带裁剪、通道+频带裁剪。
- 多 seed：NoAttention SHAP/模型覆盖 `42,49,56,63,70`；AttentionAndNoAttention 覆盖 `42,49,56`。
- 跨工况：`CrossCondTest/cross_condition.py` 与 `GNNCausal/cross_cond_test.py`。
- CWRU 外部数据支线。
- 因果发现对照：通道视角 PC-DAG、频带视角 PC-DAG、统计特征可靠边。

SCI 一区建议补充但当前代码未完整实现：

- 将所有准确率、F1、跨工况结果保存为 CSV，并做 mean/std。
- 明确多 seed 统计显著性检验，如 paired t-test 或 Wilcoxon。
- 噪声鲁棒性实验：已有 `DAG/Noise` 数据，但未见完整诊断鲁棒性入口。
- 计算复杂度：参数量、FLOPs、推理时间已经在 `ModelTest_*` 中有雏形，建议统一表格。
- 替代因果发现：当前主要 PC-KCI；可加 NOTEARS/FCI/GES 作为对比，但当前未实现。
- 替代解释方法：当前主要 SHAP；可加 Integrated Gradients/Grad-CAM 但当前未实现。

## 5. 结果文件反查

| 结果类型 | 路径 | 归属实验 | 是否适合论文 | 备注 |
| --- | --- | --- | --- | --- |
| 小波张量 | `wavelet_dataset/*.npz` | E0 | 不直接入 Git | 3.7-5.6GB/文件 |
| 训练模型 | `ModelTrain/**/SavedModels*/*.pth` | E1-E7 | 不入 Git，记录指标 | 需 LFS/外部存储 |
| 训练曲线 | `ModelTrain/**/SavedGraphs*/*.png` | E1-E7 | 可筛选入论文 | 建议复制最终图到 `paper/figures` |
| SHAP 原始值 | `CalculateShapValues/**/Seed_*/*.npz` | E8-E9 | 不入 Git | 最大 2.8GB/文件 |
| SHAP 汇总图 | `CalculateShapValues/NoAttention/TotalAnalysisGraphs/*.png` | E10 | 可入论文 | 需确认生成 seed |
| PC CSV | `PC_Datasets/*.csv`, `PC_Datasets/Seed_49/*.csv` | E12-E13 | 可考虑精简版 | 约 10MB 总表 |
| PC-DAG 图 | `DAG/PC_DAG/**/*.png/pdf` | E14 | 高价值 | 需统一命名/筛最终图 |
| reliable edge | `GNNCausal/*_reliable_edge.pt` | E15 | 可入补充材料 | 建议导出 CSV |
| GNN 模型/曲线 | `GNNCausal/trained_models`, `GNNCausal/train_curves` | E16 | 曲线可筛选 | 模型不入 Git |
| CWRU 结果 | `CWRU/trained_models`, `CWRU/train_curves` | E17 | 可作外部验证 | 需补指标表 |
| 空 CSV | `wavelet_dataset/causal_edge_intervention_result.csv` | 未确认 | 不建议直接使用 | 当前 pandas 报空文件 |

## 6. SCI 一区论文叙事建议

候选题目：

- Causality-Audited Explainable Fault Diagnosis for Rotating Machinery via SHAP-Guided Dual-View Causal Graphs
- Auditing Shortcut Learning in Rotating Machinery Fault Diagnosis with SHAP Attribution and Intervention-Verified Causal Graphs
- From Attribution to Causation: Dual-View Causal Reliability Analysis for Gearbox and Bearing Fault Diagnosis

核心科学问题：

深度诊断模型在高准确率下是否依赖了与故障机理弱相关的捷径特征？SHAP 高归因特征与 PC-KCI/干预验证的因果可靠特征是否一致？这种一致性是否能预测或提升跨工况泛化？

方法贡献点：

1. 角域重采样 + 小波多尺度张量作为统一诊断输入。
2. 通道与频带双视角 SHAP 归因，量化模型关注对象。
3. SHAP 加权 PC-KCI 因果图，识别“高归因且有因果边”的可信特征。
4. PC-KCI + 背景知识 + 虚拟 do 干预，筛选可靠因果边。
5. 将可靠因果边注入 GNN 或用于跨工况审计，连接可解释性与泛化。

图表规划：

- Fig. 1：整体 pipeline。
- Fig. 2：小波张量构建与通道/频带定义。
- Fig. 3：baseline/ablation 诊断性能表。
- Fig. 4：SHAP 通道与频带归因热图/柱图。
- Fig. 5：双视角 PC-DAG，突出高 SHAP 但弱因果节点。
- Fig. 6：可靠因果边生成与 do 干预示意。
- Fig. 7：跨工况泛化和因果 GNN 对比。
- Table 1：数据集与工况。
- Table 2：模型性能 mean/std。
- Table 3：归因-因果一致性指标。

最强证据：

- 多 seed SHAP 原始结果和汇总图已经存在。
- 双视角 PC-DAG 代码和图结果已经存在。
- 多组 baseline/ablation 模型和图结果已经存在。
- SEU 与 CWRU 两套数据均有实验痕迹。

最薄弱环节：

- 缺少统一依赖、README、复现命令和指标 CSV。
- 部分入口路径或类签名不一致，结果可追溯性不足。
- 因果 GNN 代码需确认是否真正使用了因果融合结果。
- 多 seed 统计没有形成论文级表格。
- 数据划分与 cross-finetune 可能被审稿人质疑，需要解释或补实验。

最值得保留和强化的 5 个实验方向：

1. SHAP 通道/频带归因与 PC-DAG 因果连通性的冲突分析。
2. 全特征 vs 通道裁剪 vs 频带裁剪 vs 双裁剪的诊断和跨工况泛化。
3. 多 seed SHAP 稳定性与归因排名稳定性。
4. PC-KCI + 虚拟 do 干预筛边对 GNN 泛化的增益验证。
5. CWRU 作为外部公开数据集验证，增强论文说服力。

需要人工确认的问题：

1. `wavelet_dataset/wavelet_dataset.npz` 是否曾存在，还是应统一改为四个工况 `.npz`。
2. `train_data/train_labels` 与 `train_set/train_label` 两套键名哪一套是当前标准。
3. `GNNCausalSEU.forward` 中 `classifier(x)` 是否应改为 `classifier(out)`。
4. `CNNNetWorkWithAttention(class_number, ratio=weights)` 的 `ratio` 参数是否是旧版本遗留。
5. `DAG/CreateDAG.py` 依赖的 `GearBoxCausalAnalyzer` 是否应恢复，或该脚本应标为废弃。

