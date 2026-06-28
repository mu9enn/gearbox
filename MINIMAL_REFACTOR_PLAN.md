# Minimal Refactor Plan

生成时间：2026-06-28  
目标：在不改算法逻辑的前提下，让 GearBox 项目能支撑论文复现、实验冻结和 GitHub 上传。  
原则：先包裹现有脚本，再逐步收敛重复逻辑；不直接重写模型、不改变实验结果口径。

## 0. 不做什么

- 不改 `Common/NetWorkFrame.py` 中模型结构。
- 不改 `Common/LoadDatasets.py` 的数据划分策略，除非另立新实验。
- 不重算大规模 SHAP、PC-KCI、GNN。
- 不删除历史结果。
- 不把所有脚本一次性改成新框架。

## 1. 目标目录结构

建议新增最小结构：

```text
configs/
  datasets.yaml
  models.yaml
  experiments/
    cnn_6ch.yaml
    shap_noattention.yaml
    pc_dag.yaml
    causal_gnn.yaml
gearbox/
  __init__.py
  cli.py
  registry.py
  paths.py
  seed.py
  io_contracts.py
  metrics.py
scripts/
  run_experiment.py
results/
  README.md
tests/
  test_imports.py
  test_npz_contract_headers.py
  test_registry.py
```

现有目录先保留：

- `Common/`, `ModelTrain/`, `CalculateShapValues/`, `DAG/`, `GNNCausal/`, `CWRU/` 不立即移动。
- 新 `gearbox/` 只做 wrapper 和 registry，不承接算法实现。

## 2. 统一 CLI

### 2.1 最小 CLI 命令

新增 `gearbox/cli.py`，提供：

```bash
python -m gearbox.cli train --config configs/experiments/cnn_6ch.yaml --condition Gear_30_2 --seed 42
python -m gearbox.cli evaluate --config configs/experiments/cnn_6ch.yaml --checkpoint <path>
python -m gearbox.cli explain --config configs/experiments/shap_noattention.yaml --condition Gear_30_2 --seed 42
python -m gearbox.cli causal-dag --config configs/experiments/pc_dag.yaml --condition Gear_30_2
python -m gearbox.cli manifest --result-dir results/...
```

### 2.2 第一阶段只做 wrapper

CLI 先调用现有函数/脚本逻辑，不改算法：

- train -> 调用 `Common.ModelTrainAndVisiable.ModelTrain.train_model` 或 `ParallelTraining.model_train`。
- explain -> 包装 `CalculateShapValues/NoAttention/*.py` 的核心逻辑。
- causal-dag -> 调用 `Common.PCCausal.WaveletDAG`。
- causal-edge -> 调用 `Common.CausalAndDoWhy.CausalAnalysis`。

### 2.3 为什么需要 CLI

当前问题：

- 每个工况一个脚本，路径和 seed 写死。
- 结果无法记录命令。
- 多 seed 批跑无法统一。

最小收益：

- 每次运行都有固定命令。
- 每个结果目录能保存 `command.txt`。
- 论文补实验时不再手工复制脚本。

## 3. Config

### 3.1 配置范围

先覆盖最小参数：

```yaml
experiment:
  name: cnn_6ch
  group: main
run:
  seed: 42
  device: auto
  output_dir: results/cnn_6ch/Gear_30_2/seed_42
dataset:
  name: SEU
  condition: Gear_30_2
  npz_path: wavelet_dataset/Gear_30_2.npz
  keys:
    train_x: train_set
    train_y: train_label
    test_x: test_set
    test_y: test_label
model:
  name: cnn_noattention
  class_number: 5
train:
  lr: 0.00015
  batch_size: 256
  epochs: 50
```

### 3.2 数据契约先冻结

新增 `gearbox/io_contracts.py`：

- 支持 `train_set/train_label`。
- 支持历史 `train_data/train_labels`，但要 warning。
- 只读 `.npz` header 检查键名，不加载大数组。

需要人工确认：

- 最终标准键名建议冻结为 `train_set/train_label`，因为 `Common/LoadDatasets.py:211-222` 当前这样保存。

### 3.3 Config snapshot

每次运行复制配置到：

```text
results/<experiment>/<condition>/seed_<seed>/config.resolved.yaml
```

并保存：

- `command.txt`
- `run_manifest.json`
- `environment.txt`
- `git_status.txt`

## 4. Results 目录

### 4.1 统一结果结构

建议新结果目录：

```text
results/
  cnn_6ch/
    Gear_30_2/
      seed_42/
        checkpoints/
        metrics/
        figures/
        manifests/
  shap_noattention/
  pc_dag/
  causal_edges/
  paper_freeze_YYYYMMDD/
```

旧结果目录不移动，先在 manifest 中引用：

- `ModelTrain/**/SavedModels*`
- `CalculateShapValues/**/Seed_*`
- `DAG/PC_DAG/**`

### 4.2 必需输出文件

每次实验至少输出：

```text
metrics/metrics.json
metrics/metrics.csv
figures/*.png
manifests/run_manifest.json
manifests/input_files.json
manifests/checksums.json
config.resolved.yaml
command.txt
```

### 4.3 图表冻结

论文图统一复制或再生成到：

```text
results/paper_freeze_YYYYMMDD/figures/
results/paper_freeze_YYYYMMDD/tables/
results/paper_freeze_YYYYMMDD/manifests/
```

每张图必须有同名 `.json` 说明输入文件、脚本、配置和 hash。

## 5. Dataset Registry

新增 `configs/datasets.yaml`：

```yaml
SEU:
  root: DataSets/SEU
  wavelet:
    Bearing_20_0: wavelet_dataset/Bearing_20_0.npz
    Bearing_30_2: wavelet_dataset/Bearing_30_2.npz
    Gear_20_0: wavelet_dataset/Gear_20_0.npz
    Gear_30_2: wavelet_dataset/Gear_30_2.npz
  channels:
    raw_indices: [1, 2, 3, 5, 6, 7]
    names: [ch2, ch3, ch4, ch6, ch7, ch8]
  conditions:
    Bearing_20_0:
      rpm: 1200
      type: bearing
    Bearing_30_2:
      rpm: 1800
      type: bearing
    Gear_20_0:
      rpm: 1200
      type: gear
    Gear_30_2:
      rpm: 1800
      type: gear
CWRU:
  wavelet: DataSets/CWRU/WaveletDataset/all_cond_wave.npz
```

用途：

- 替代脚本内的硬编码路径。
- 防止大小写不一致。
- 记录每个条件的 rpm、类别、标签。

## 6. Model Registry

新增 `configs/models.yaml` 或 `gearbox/registry.py`：

```yaml
cnn_noattention:
  class: Common.NetWorkFrame.CNNNetWorkNoAttention
  input: [6, 6, 512]
cnn_attention:
  class: Common.NetWorkFrame.CNNNetWorkWithAttention
  input: [6, 6, 1024]
base_4ch:
  class: Common.NetWorkFrame.BaseModel4Channel
  input: [4, 6, 1024]
base_freq:
  class: Common.NetWorkFrame.BaseModelFreq
  input: [6, 4, 1024]
base_ch_freq:
  class: Common.NetWorkFrame.BaseModel
  args:
    layer_number: 4
gnn_causal_seu:
  class: Common.NetWorkFrame.GNNCausalSEU
```

需要人工确认：

- `CNNNetWorkNoAttention` 的线性层当前按 `512` 长度写 `12*6*128`（`Common/NetWorkFrame.py:143-151`），但部分脚本使用 `1024` 长度。需确认最终输入长度。
- `CNNNetWorkWithAttention` 线性层按 `1024` 长度写 `12*6*256`（`Common/NetWorkFrame.py:113-119`）。

## 7. Tests

### 7.1 第一批测试，不跑训练

新增：

```text
tests/test_imports.py
tests/test_npz_contract_headers.py
tests/test_registry.py
tests/test_path_case.py
```

测试内容：

- `Common.LoadDatasets`, `Common.NetWorkFrame`, `Common.ModelTrainAndVisiable`, `Common.PCCausal` 可导入。
- `.npz` header 包含预期键，不加载大数组。
- registry 中的路径存在或被标记为 external。
- 检查 `Gear_30_2.npz` vs `gear_30_2.npz` 这种大小写风险。

### 7.2 第二批 smoke tests

在 `examples/tiny/` 放极小假数据：

- 构造 `(8,6,6,512)` 或 `(8,6,6,1024)` 随机张量。
- 测试模型 forward。
- 测试 metrics 导出。
- 测试 PC-DAG wrapper 能处理小表。

### 7.3 不建议立刻做的测试

- 不跑完整训练。
- 不跑完整 SHAP。
- 不跑完整 PC-KCI。
- 不跑 CWRU 全流程。

## 8. .gitignore 与 .gitattributes

当前已存在：

- `.gitignore`
- `.gitattributes`
- `.env.example`

建议继续保持：

- 默认忽略 `DataSets/`, `wavelet_dataset/*.npz`, `CalculateShapValues/**/Seed_*`, `*.pth`, `*.pt`, `*.npz`, `*.mat`。
- 只提交源码、配置、文档、小型示例数据和最终论文图/表。
- `.gitattributes` 对大文件类型配置 LFS，但实际是否跟踪由 `.gitignore` 和人工选择决定。

需要补充：

- `results/README.md`，说明结果默认不入 Git。
- `data/README.md` 或 `DataSets/README.md`，说明数据下载方式和许可。

## 9. CI

### 9.1 最小 GitHub Actions

新增 `.github/workflows/ci.yml`：

```yaml
name: ci
on: [push, pull_request]
jobs:
  smoke:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: python -m pip install -r requirements.txt
      - run: python -m pytest tests/test_imports.py tests/test_registry.py
```

### 9.2 CI 不做

- 不下载 SEU/CWRU 大数据。
- 不跑 GPU。
- 不跑 SHAP/PC-KCI/训练。

### 9.3 需要先补

- `requirements.txt` 或 `environment.yml`。
- tiny fixtures。
- 导入路径：当前多处 `from GearBox...`，CI 需要确认包路径。

## 10. 分阶段实施路线

### Phase 1：冻结契约与文档

产物：

- `DATA_CONTRACT.md`
- `configs/datasets.yaml`
- `configs/models.yaml`
- `results/README.md`

不改算法。

### Phase 2：加 wrapper 与 manifest

产物：

- `gearbox/cli.py`
- `gearbox/paths.py`
- `gearbox/seed.py`
- `gearbox/metrics.py`
- `gearbox/io_contracts.py`

只包裹现有函数。

### Phase 3：统一评估导出

产物：

- `metrics.json`
- `metrics.csv`
- `confusion_matrix.csv`
- `classification_report.json`
- `cross_condition_accuracy.csv`

优先改评估脚本，不改模型。

### Phase 4：主线脚本参数化

迁移顺序：

1. `ModelTrain/NoAttention/model_train_6ch/*.py`
2. `ModelTrain/NoAttention/model_train_4ch/*.py`
3. `ModelTrain/NoAttention/model_train_freq/*.py`
4. `CalculateShapValues/NoAttention/*.py`
5. `DAG/WaveletDAG.py`

### Phase 5：风险实验确认

只在主线稳定后处理：

- `GNNCausalSEU.forward` 是否使用 `out`。
- `GNNCausal/CausalGNN.py:matrix_file` 是否应按工况读取可靠边。
- `DAG/CreateDAG.py` 是否废弃。

## 11. 验收标准

最小重构完成后，应能回答：

1. 每个论文图来自哪个脚本、哪个 config、哪个输入文件。
2. 每个实验的 seed、数据 split、checkpoint、metrics 都能追溯。
3. 任意一个主线实验可用 CLI 复跑。
4. GitHub 仓库不含大数据和原始 SHAP。
5. CI 能在无大数据环境下完成 import、registry、contract smoke test。

