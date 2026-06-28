# Paper Experiment Matrix

生成时间：2026-06-28  
目标：按论文主线冻结现有实验脚本，不生成论文正文，不改算法。

## 0. Claim 分层

建议冻结 claim 时按证据强弱分层：

| Claim ID | 建议表述方向 | 当前证据强度 | 备注 |
| --- | --- | --- | --- |
| C1 | 角域重采样 + SWT 小波张量可作为 SEU 多工况故障诊断输入 | 强 | `Common/LoadDatasets.py:70-222` 与 `wavelet_dataset/*.npz` |
| C2 | CNN baseline 能在四个 SEU 工况上完成故障诊断 | 中-强 | 多个 `ModelTrain/NoAttention/model_train_6ch/*.py` 与权重/曲线结果 |
| C3 | 通道/频带/双裁剪会改变固定工况与跨工况性能 | 中 | `CrossCondTest/cross_condition.py` 有图但缺 CSV/mean/std |
| C4 | SHAP 能量归因能定位模型关注的通道与频带 | 强 | 多 seed SHAP 结果与聚合脚本 |
| C5 | SHAP 高归因特征与 PC-KCI 因果边的一致/冲突可用于逻辑审计 | 强 | `Common/PCDataset(wavelet).py`, `Common/PCCausal.py`, `DAG/WaveletDAG.py` |
| C6 | PC-KCI + 虚拟 do 干预可筛出可靠因果边 | 中 | `Common/CausalAndDoWhy.py` 有完整链路，但缺稳定性/表格 |
| C7 | 可靠因果边注入 GNN 可提升/解释泛化 | 弱-中 | `GNNCausalSEU.forward` 是否真正使用因果融合需人工确认 |
| C8 | 方法能推广到 CWRU | 中 | 有 CWRU 支线结果，但主线指标未统一 |

## 1. 主线实验

这些实验建议作为 SCI 论文核心实验链。重跑优先级最高。

| 实验 | 入口脚本/模块 | 回答什么问题 | 支撑 claim | 证据强度 | 是否重跑 | mean/std | 多 seed | 显著性检验 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SEU 小波数据构建 | `Common/LoadDatasets.py:DataSets.wavelet_transform` | 原始振动如何变成可训练小波张量 | C1 | 强 | 不一定，除非修正契约 | 不需要 | 预处理 seed 需固定 | 不需要 | 需记录 split 与 cross-finetune 策略 |
| 6 通道 CNN baseline | `ModelTrain/NoAttention/model_train_6ch/*.py`, `Common/ModelTrainAndVisiable.py:train_model` | 全特征 baseline 性能如何 | C2 | 中-强 | 建议重跑一次干净版 | 需要 | 需要，至少 5 seeds | 需要 | 当前结果存在，但 metrics 未结构化导出 |
| 通道/频带/双裁剪消融 | `model_train_4ch`, `model_train_freq`, `BaseModelTrain` | 削减输入维度是否影响性能/泛化 | C3 | 中 | 建议重跑 | 需要 | 需要 | 需要 | 需统一输入选择和命名 |
| 固定/跨工况比较 | `ModelTrain/NoAttention/CrossCondTest/cross_condition.py` | 裁剪策略是否影响跨工况泛化 | C3 | 中 | 建议重跑并保存 CSV | 需要 | 需要 | 需要 | 当前只保存 `Cross_condition_acc_comparison.png` |
| NoAttention SHAP | `CalculateShapValues/NoAttention/*.py` | 模型关注哪些通道/频带 | C4 | 强 | 不一定，若模型重跑则重算 | 需要聚合 std | 已有 5 seeds | 排名稳定性检验建议 | 原始 SHAP 很大，冻结聚合结果 |
| SHAP -> PC 双视角数据 | `Common/PCDataset(wavelet).py` | 如何把归因转成通道/频带因果节点 | C5 | 强但脚本契约需确认 | 需要轻量重跑/修契约后重跑 | 需要 | 已写 5 seeds | 可做 edge stability | `all_data/all_labels` 键需人工确认 |
| 通道/频带 PC-KCI DAG | `DAG/WaveletDAG.py`, `Common/PCCausal.py:WaveletDAG` | 高归因节点是否有 label 因果边 | C5 | 强 | 建议重跑最终图 | 需要边稳定表 | 需要 | 需要 edge consistency | 论文核心图 |

主线最小闭环：

```text
E0 preprocess -> E1 CNN baseline -> E4 SHAP -> E5 dual-view PC-DAG -> audit table
```

这条闭环足以支撑“归因-因果一致性审计”主线，不依赖 GNN 是否成熟。

## 2. 支撑实验

这些实验可以增强论文说服力，但不应压过主线。

| 实验 | 入口脚本/模块 | 回答什么问题 | 支撑 claim | 证据强度 | 是否重跑 | mean/std | 多 seed | 显著性检验 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 混淆矩阵/F1 | `ModelTrain/NoAttention/ConfusionMatrix.py`, `ModelTrain/AttentionAndNoAttention/ConfusionMatrix.py` | 哪些故障类别容易混淆 | C2/C3 | 中 | 建议重跑最终模型 | 需要 | 可按 seed 汇总 | 可选 | 当前主要保存图 |
| 参数量/推理时间 | `ModelTrain/NoAttention/ModelTest_ch/*.py`, `ModelTrain/NoAttention/ModelTest_freq/*.py` | 裁剪是否带来轻量化 | C3 | 中 | 建议重跑并导出表 | 需要 | 可选 | 可选 | 脚本有 `count_params`, `infer_time` 函数 |
| Attention vs NoAttention | `ModelTrain/AttentionAndNoAttention/*.py`, `ParallelTraining.model_train` | attention 是否改变性能与归因 | C4/C5 | 中 | 需要人工确认输入契约后重跑 | 需要 | 已有 3 seeds 结果 | 建议 | `wavelet_dataset.npz` 与 `ratio` 参数存在风险 |
| SHAP 与 attention 权重相关性 | `Analysis/ChannelWeights.py` | attention 权重是否与 SHAP 一致 | C4 | 弱-中 | 建议重跑为 CSV | 需要 | 需要 | 相关显著性可选 | 当前多为终端打印 |
| PC_Datasets 全局时频特征 | `Common/PCDatasets.py` | 手工时频特征能否辅助因果解释 | C5 | 中 | 需要明确用途后重跑 | 需要 | 不明确 | 可选 | 全局标准化存在泄露争议，建议只做可视化/附录 |
| SHAP 汇总图 | `CalculateShapValues/NoAttention/analysis.py`, `total_analysis.py` | 归因排名和通道/频带图如何呈现 | C4 | 强 | 建议用最终 seed 结果重画 | 需要 | 需要 | 排名稳定性 | 论文图候选 |

## 3. 风险实验

这些实验有潜在高价值，但现在不能直接作为强 claim 使用。

| 实验 | 入口脚本/模块 | 回答什么问题 | 可能支撑 claim | 证据强度 | 是否重跑 | mean/std | 多 seed | 显著性检验 | 风险 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 可靠因果边生成 | `Common/CausalAndDoWhy.py:CausalAnalysis.generate_and_save_reliable_edges` | PC-KCI 边能否经虚拟 do 筛选 | C6 | 中 | 建议重跑小规模/最终全量 | 需要边数/稳定性 | 需要 | 需要 edge stability | PC-KCI 成本高，日志未结构化 |
| 因果 GNN 诊断 | `GNNCausal/CausalGNN.py`, `Common/GNNTrain.py:GNNTrainSEU` | 因果结构能否提升诊断/泛化 | C7 | 弱-中 | 需要修正/确认后重跑 | 需要 | 需要 | 需要 | `GNNCausalSEU.forward` 当前 `classifier(x)` 可能绕开因果融合 |
| GNN 跨工况微调 | `GNNCausal/cross_cond_test.py` | 可靠边是否提升跨工况迁移 | C7 | 弱-中 | 需要重跑 | 需要 | 需要 | 需要 | 依赖前一个风险实验 |
| CWRU 因果 GNN | `CWRU/ModelTrainAndCrossTest.py` | 方法是否能外部验证 | C8 | 中 | 建议最后重跑 | 需要 | 需要 | 需要 | 与 SEU 主线的数据契约不同 |
| 去趋势纯净信号 | `DAG/data_treat.py` | 去趋势是否让因果特征更清晰 | C5/C6 | 中 | 需要先修路径 | 需要 | 不明确 | 可选 | 当前脚本路径疑似少 `SEU` 层 |

风险实验的冻结建议：

- `C6` 可以作为“方法组件”写，但需要可靠边 CSV/表格支撑。
- `C7` 暂时不要作为论文核心 claim，除非人工确认并修复 GNN 是否真正使用因果传播。
- `C8` 可作为 external validation，但不要承担主创新。

## 4. 历史遗留实验

这些脚本或结果应先标为历史遗留，不建议直接进入主论文。

| 实验/脚本 | 位置 | 原因 | 处理建议 |
| --- | --- | --- | --- |
| 旧 PC DAG 总图 | `DAG/CreateDAG.py` | import `GearBoxCausalAnalyzer`，但当前 `Common/PCCausal.py` 中该类被注释 | 若要保留，恢复类或重写入口；否则标废弃 |
| `Frame/Attention.py`, `Frame/NoAttention.py` | `Frame/` | 与 `Common/NetWorkFrame.py` 重复，另有 ONNX 文件 | 作为模型导出/历史版本，不进主线 |
| ONNX 导出 | `Frame/*.onnx`, `*.onnx.data` | 未见主线调用 | 不进论文实验，Git 忽略 |
| 零散测试脚本 | `Analysis/test.py`, `DAG/test.py`, `GNNCausal/analysis.py`, `GNNCausal/trained_models/GNN/analysis.py` | 调试性质，输出链路不完整 | 保留但不引用 |
| `wavelet_dataset/analysis.py` | `wavelet_dataset/analysis.py` | 使用 DoWhy/PC 做干预 CSV，但结果文件有空文件风险 | 需人工确认是否来自旧路线 |
| `ModelTrain/AttentionAndNoAttention/AllData_of_SixChannels_CompareAttentionWithNoAttention.py` | 单文件全数据对比 | 与工况脚本重复，保存路径不完全统一 | 只作历史参考 |

## 5. 未来拓展实验

这些方向当前代码或数据有痕迹，但不足以纳入当前论文主线。

| 方向 | 当前证据 | 可回答问题 | 纳入条件 |
| --- | --- | --- | --- |
| XJTU-SY 退化数据拓展 | `DataSets/XJTU-SY/**.csv` | 方法是否适合寿命退化/预测 | 需要完整预处理、训练、解释、因果链 |
| 噪声鲁棒性 | `DAG/Noise/*.npz`, `DAG/PolyCurves/Noise/*.png` | 方法在噪声下是否稳定 | 需要训练/评估入口和结果表 |
| 替代解释方法 | 当前主要 SHAP | IG/Grad-CAM 与 SHAP 是否一致 | 需要新增实现，不属本轮 |
| 替代因果发现 | 当前主要 PC-KCI | FCI/GES/NOTEARS 对边稳定性影响 | 需要新增实现，不属本轮 |
| 复杂度与部署 | `ModelTest_*` 有参数量/耗时雏形，`Frame/*.onnx` | 能否轻量部署 | 需要统一 benchmark |

## 6. 重跑优先级

### P0 必须重跑或至少重算表

1. 6 通道 CNN baseline：输出每 seed 每工况 accuracy/loss/F1 CSV。
2. 4 通道、频带、双裁剪消融：输出固定/跨工况 CSV。
3. SHAP 聚合：输出通道/频带 mean/std/rank CSV。
4. PC-DAG：输出边列表、边稳定性、SHAP-因果一致性表。

### P1 建议重跑

1. attention/no-attention 对照。
2. confusion/F1 最终图。
3. 参数量和推理时间表。
4. CWRU 外部验证。

### P2 暂不重跑

1. 因果 GNN，直到确认 `classifier(x)` 问题。
2. `DAG/CreateDAG.py` 旧路线。
3. XJTU-SY 拓展。

## 7. 最终论文实验矩阵建议

建议冻结矩阵：

| Section | 实验 | Claim | 图/表 |
| --- | --- | --- | --- |
| Dataset/Preprocessing | SEU angle-domain SWT | C1 | 数据集表、pipeline 图 |
| Diagnostic Baseline | 6ch CNN + ablations | C2/C3 | 性能表、跨工况图 |
| Explanation | SHAP channel/frequency | C4 | 归因热图/柱图、rank stability |
| Causal Audit | Dual-view PC-KCI DAG | C5 | 通道 DAG、频带 DAG、一致性表 |
| Validation | Reliable edges / CWRU / complexity | C6/C8 | 附加表，谨慎 claim |

不要把“因果 GNN 提升泛化”作为主结论，除非完成风险确认和重跑。

