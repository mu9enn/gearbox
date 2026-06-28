# Reproducibility Gap Report

生成时间：2026-06-28  
目标：检查当前 GearBox 项目的可复现性缺口，不改算法逻辑。

## 0. 总体结论

当前项目有较完整的实验产物，但还不是“论文可复现包”。主要缺口是：

1. 数据契约不统一：`train_set/train_label` 与 `train_data/train_labels` 并存。
2. 随机种子分散在脚本里，预处理和部分抽样未统一记录。
3. metrics 多数只存在内存、终端输出或图片，没有结构化 CSV/JSON。
4. 没有 config snapshot、git hash、依赖版本、命令行参数记录。
5. 图表生成链路存在，但从图反查模型、seed、数据版本困难。

## 1. 随机种子

### 已有证据

- `Common/NetWorkFrame.py` 模块级设置 `seed = 49`，并设置 `random`, `np`, `PYTHONHASHSEED`, `torch`, `torch.cuda`, `cudnn.deterministic=True`, `cudnn.benchmark=False`（`Common/NetWorkFrame.py:10-24`）。
- 多个训练脚本各自设置 seed，例如 `CalculateShapValues/NoAttention/Gear_30_2.py:9-23` 使用 seed `70`。
- `GNNCausal/CausalGNN.py:13-16` 设置 `np.random.seed(42)`, `torch.manual_seed(42)`, `random.seed(42)`。
- `Common/CausalAndDoWhy.py:227-232` 在主入口中设置 `np.random.seed(42)` 和 `torch.manual_seed(42)`。
- `Common/PCDataset(wavelet).py:49-58` 显式聚合 `seeds = [42,49,56,63,70]` 的 SHAP。

### 缺口

| 项 | 状态 | 风险 | 建议 |
| --- | --- | --- | --- |
| 全局 seed registry | 缺失 | 无法一眼确认每个实验的 seed | 建议 `configs/seeds.yaml` |
| 预处理 seed | 不完整 | `cross_finetune` 用 `np.random.randint`，入口无固定 seed | 在预处理入口固定并记录 seed |
| DataLoader worker seed | 缺失 | 若未来使用多 worker，复现会漂移 | 增加 `worker_init_fn` |
| SHAP 抽样 seed | 脚本内有，但不导出 | 无法从结果文件反查抽样 | 结果保存 `seed`, `background_ids`, `explain_ids` |
| Optuna seed | 写法可能无效 | `optuna.samplers.RandomSampler(seed=42)` 被调用但未传入 study | 若启用 Optuna，显式 `sampler=RandomSampler(seed=42)` |

需要人工确认：

- 当前最终论文结果究竟对应哪些 seed：NoAttention 看起来有 `42,49,56,63,70`；AttentionAndNoAttention 看起来有 `42,49,56`。

## 2. Train/Test Split

### 已有证据

- SEU 预处理按每个故障类别的原始时间顺序切分：前 80% train，接着 10% valid，最后 10% test（`Common/LoadDatasets.py:123-128`）。
- 窗口步长：train/test `step=768`，valid/cross_finetune `step=1024`，cross_test `step=768`（`Common/LoadDatasets.py:179-184`）。
- cross-finetune 是全局随机连续 10% 片段，注释写明允许与 train 重叠（`Common/LoadDatasets.py:153-163`）。
- `GNNTrainCWRU.forward` 对输入再做 `train_test_split(..., test_size=0.2, random_state=42)`（`Common/GNNTrain.py:37-45`）。
- `ModelTrain/NoAttention/CrossCondTest/cross_condition.py` 使用固定工况 test set 和交叉工况 test set 做比较（`ModelTrain/NoAttention/CrossCondTest/cross_condition.py:24-42`, `76-132`）。

### 缺口

| 项 | 状态 | 风险 | 建议 |
| --- | --- | --- | --- |
| split manifest | 缺失 | 无法追溯每个样本窗口来源 | 保存每个 `.npz` 的 split metadata |
| cross_finetune 重叠 | 明确存在 | 审稿人可能质疑数据泄露 | 主文中说明用途，或新增 non-overlap ablation |
| train/test 键名 | 不统一 | 脚本可能读取不到数据或读历史产物 | 冻结标准为 `train_set/train_label` 或迁移 |
| 文件大小写 | 不统一 | Linux/CI 会失败 | 统一 `Bearing_20_0.npz` 与脚本路径 |
| CWRU 二次 split | 有 | 与 SEU split 逻辑不同 | 单独记录，不混入 SEU 主线 |

需要人工确认：

- `wavelet_dataset` 目录当前实际 `.npz` 是 `train_set` 契约，但不少 `ModelTrain` 脚本读取 `train_data`。这些脚本是否依赖旧版未列出的 `wavelet_dataset.npz`？

## 3. 标准化统计量来源

### 已有证据

- SEU 小波 CNN 训练器 `ModelTrain.train_model` 未在公共训练器中做标准化；直接把数据转为 tensor（`Common/ModelTrainAndVisiable.py:62-76`）。
- `Common/CausalAndDoWhy.py:causal_data` 对训练数据统计特征使用传入的 `StandardScaler.fit_transform`（`Common/CausalAndDoWhy.py:209-224`, `238-252`）。
- `GNNCausal/CausalGNN.py:data_transfer` 对 train/valid/test 使用 train scaler，但对 finetune/cross_test 重新 `fit_transform` finetune，再 transform cross_test（`GNNCausal/CausalGNN.py:71-79`）。
- `Common/PCDataset(wavelet).py` 每个工况对通道节点和频带节点各自 `scaler.fit_transform`（`Common/PCDataset(wavelet).py:129-140`）。
- `Common/PCDatasets.py` 将所有纯净数据拼接后全局 min/max 归一化（`Common/PCDatasets.py:286-293`），再对全部 feature 做 `StandardScaler.fit_transform` 和 `MinMaxScaler.fit_transform`（`Common/PCDatasets.py:341-347`）。
- `GNNTrainCWRU.forward` 使用 `x_train_orig.mean/std` 标准化 train/test（`Common/GNNTrain.py:37-45`）。

### 风险分级

| 风险 | 位置 | 影响 | 建议 |
| --- | --- | --- | --- |
| 高 | `Common/PCDatasets.py:286-347` | 如果用于评估，存在全数据统计量泄露 | 明确仅用于因果可视化，或改为 train-only scaler |
| 中 | `GNNCausal/CausalGNN.py:76-77` | finetune/cross_test 使用独立 scaler，跨工况评估口径需说明 | 保存 scaler 来源并统一策略 |
| 中 | `Common/PCDataset(wavelet).py:129-140` | 每工况 fit scaler，跨工况比较解释需谨慎 | 记录每工况 scaler |
| 低 | CNN 训练无标准化 | 可复现但性能解释依赖原始尺度 | 记录“未标准化”作为配置 |

## 4. Checkpoint

### 已有证据

- 单模型训练在早停或 epoch 结束保存 `torch.save(self.model.state_dict(), save_path)`（`Common/ModelTrainAndVisiable.py:164-178`）。
- 并行 attention/no-attention 训练保存两个 `state_dict`（`Common/ModelTrainAndVisiable.py:466-481`）。
- GNN SEU 在 valid acc 最优时保存模型（`Common/GNNTrain.py:277-280`）。
- GNN CWRU 训练结束保存模型（`Common/GNNTrain.py:97`）。

### 缺口

| 项 | 当前状态 | 建议 |
| --- | --- | --- |
| checkpoint metadata | 只保存 `state_dict` | 另存 `checkpoint.json`：model, seed, data hash, config, best epoch, metrics |
| best epoch | 未统一保存 | 保存 `best_epoch` 和 `selection_metric` |
| optimizer state | 未保存 | 若只复现最终评估可不需要；若续训需保存 |
| model registry | 缺失 | 建议 registry 映射 `cnn_6ch`, `cnn_4ch`, `cnn_freq`, `cnn_att`, `gnn_causal` |
| 权重与结果绑定 | 弱 | 每个 result dir 保存 `model_path` 和 `checkpoint_sha256` |

## 5. Metrics 导出

### 已有证据

- `ModelTrain.train_model` 返回 `train_acc`, `test_acc`, `train_loss`, `test_loss`，但公共函数只保存 `.pth`，图由调用脚本保存（`Common/ModelTrainAndVisiable.py:133-186`, `188-250`）。
- `ParallelTraining.model_train` 返回 attention/no-attention 的 acc/loss 和权重数组（`Common/ModelTrainAndVisiable.py:395-490`）。
- `CrossCondTest/cross_condition.py` 将准确率放在 `full_cond_acc`, `full_cross_cond_acc`，最后只保存图（`ModelTrain/NoAttention/CrossCondTest/cross_condition.py:76-189`）。
- ConfusionMatrix 脚本用 `classification_report(..., output_dict=True)`，但当前主要保存图，未见 report CSV/JSON 导出。

### 缺口

| 指标 | 当前保存 | 缺口 |
| --- | --- | --- |
| train/test loss/acc | 曲线 PNG | 缺 CSV/JSON |
| best test acc | 可能在曲线中可读 | 缺 `metrics.json` |
| confusion matrix | PNG | 缺 raw matrix CSV |
| precision/recall/F1 | PNG | 缺 classification report JSON |
| cross-condition acc | PNG | 缺固定/跨工况数值表 |
| SHAP rank | 图/npz | 缺 mean/std/rank stability CSV |
| PC-DAG edges | 图 | 缺 edge list CSV 和 edge stability |
| reliable edges | `.pt` | 缺可读 CSV/JSON |

最低补齐标准：

- 每个实验目录必须有 `metrics.json`, `metrics.csv`, `config.yaml`, `run_manifest.json`。
- 每张论文图必须能从一个 CSV 或 JSON 复现。

## 6. Config Snapshot

当前状态：

- 没有集中配置文件。
- 超参数散落在脚本中：lr、batch、epoch、seed、路径、通道选择、频带选择、SHAP 样本数、PC alpha。
- 路径硬编码明显：如 `GNNCausal/CausalGNN.py:224-231`, `ModelTrain/NoAttention/CrossCondTest/cross_condition.py:25`, `CalculateShapValues/NoAttention/Gear_30_2.py:26-40`。

建议最小 config schema：

```yaml
run:
  seed: 42
  device: auto
  output_dir: results/<experiment>/<condition>/<seed>
dataset:
  name: SEU
  condition: Gear_30_2
  npz_path: wavelet_dataset/Gear_30_2.npz
  contract: train_set
model:
  name: cnn_6ch
  checkpoint: null
train:
  lr: 0.00015
  batch_size: 256
  epochs: 50
explain:
  method: shap_gradient
  background_samples: 100
  explain_samples_per_class: 100
causal:
  method: pc_kci
  alpha: 0.05
```

## 7. Git Hash 与环境记录

当前状态：

- 目录不是 Git 仓库，因此无法记录当前 commit hash。
- 无依赖文件，无法复现 Python 包版本。
- 当前运行环境可执行 Python，但项目本身未声明 Python 版本。

建议：

- 初始化 Git 后，每个 run 保存 `git rev-parse HEAD`。
- 若工作区 dirty，保存 `git status --short` 到 `run_manifest.json`。
- 保存 `python --version`, `pip freeze`, `torch.__version__`, CUDA 可用性。
- 保存关键大文件 hash：输入 `.npz`, checkpoint, SHAP `.npz`。

## 8. 图表生成链路

### 已有链路

| 图表类型 | 入口 | 输出路径 |
| --- | --- | --- |
| 训练曲线 | `ModelTrainAndVisiable.visible`, `ParallelTraining.visible`, `GNNTrain.visible` | `SavedGraphs*`, `train_curves` |
| Confusion/F1 | `ModelTrain/**/ConfusionMatrix.py` | `ConfusionAndF1/**` |
| 跨工况柱图 | `ModelTrain/NoAttention/CrossCondTest/cross_condition.py` | `Cross_condition_acc_comparison.png` |
| SHAP 图 | `CalculateShapValues/NoAttention/analysis.py`, `total_analysis.py` | `*_AnalysisGraphs`, `TotalAnalysisGraphs` |
| PC-DAG | `DAG/WaveletDAG.py`, `Common/PCCausal.py` | `DAG/PC_DAG/Seed_49/**` |
| 纯净信号拟合图 | `DAG/data_treat.py` | `DAG/PolyCurves`, `DAG/PolyCurves/Noise` |

### 缺口

- 图表脚本大多直接读相对路径，没有 config。
- 图表未记录源 CSV/NPZ/checkpoint hash。
- 图表目录没有 `figure_manifest.json`。
- 部分图只保存 PNG，论文可能还需要 PDF/SVG。
- 旧图和新图混在结果目录，最终论文图缺少冻结目录。

建议冻结：

```text
results/
  paper_freeze_YYYYMMDD/
    tables/
    figures/
    manifests/
```

每个 figure 保存：

- `figure_id`
- `script`
- `input_files`
- `input_hashes`
- `config`
- `git_hash`
- `created_at`

## 9. 优先修复清单

| 优先级 | 缺口 | 最小动作 |
| --- | --- | --- |
| P0 | 数据契约不统一 | 写 `DATA_CONTRACT.md`，确认 `train_set` vs `train_data` |
| P0 | metrics 不导出 | 给训练/评估脚本旁加只读导出 wrapper，不改模型 |
| P0 | config 缺失 | 每次运行保存 config snapshot |
| P0 | result manifest 缺失 | 每个结果目录写 `run_manifest.json` |
| P1 | seed 分散 | 建 `configs/seeds.yaml` 和统一 `set_seed()` |
| P1 | 图表不可追溯 | 每张图旁保存源数据 CSV/JSON |
| P1 | Git hash 缺失 | 初始化 Git 后纳入 manifest |
| P2 | CI 缺失 | 先做 import/smoke/header tests |

