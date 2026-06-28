# GitHub Ready Checklist

生成时间：2026-06-28  
项目根目录：`/Volumes/mobile-disk/1/GearBox`

## 本轮新增/修改文件

- `.gitignore`
- `.gitattributes`
- `.env.example`
- `docs/PROJECT_CODEBASE_KNOWLEDGE.md`
- `docs/EXPERIMENTS_LOGIC_AND_SCI_STORYLINE.md`
- `docs/GITHUB_READY_CHECKLIST.md`

## 当前 Git 状态

当前目录不是 Git 仓库，`git status --short` 返回：

```text
fatal: not a git repository (or any parent up to mount point /Volumes)
Stopping at filesystem boundary (GIT_DISCOVERY_ACROSS_FILESYSTEM not set).
```

因此，本轮无法判断哪些大文件已经被 Git 跟踪。建议先保留 `.gitignore`，再初始化或复制到 Git 仓库中。

## .gitignore 策略

默认忽略：

- Python 缓存：`__pycache__/`, `*.py[cod]`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`。
- 本地环境：`.env`, `.env.*`, `.vscode/`, `.idea/`, `.DS_Store`。
- 大型原始/中间数据：`DataSets/`, `wavelet_dataset/*.npz`, `DAG/PureData/`, `DAG/Noise/`。
- 大型解释结果：`CalculateShapValues/**/Seed_*/`, `CalculateShapValues/**/ShapResults/`。
- 模型和训练产物：`*.pth`, `*.pt`, `*.ckpt`, `*.onnx`, `*.onnx.data`, `ModelTrain/**/SavedModels*/`, `GNNCausal/trained_models/`, `CWRU/trained_models/`。
- 大型数组/数据表：`*.npy`, `*.npz`, `*.mat`, `*.h5`, `*.pkl`, `*.joblib`。

保留：

- `docs/**`
- Python 源码
- `PC_Datasets/README.md` 或其他未来手写说明
- `examples/**`、`configs/**` 如果后续新增

注意：当前 `.gitignore` 没有全局忽略所有 `*.csv`/`*.png`/`*.pdf`，因为项目中有论文图和小型结果表可能需要人工筛选提交；但大型数据目录和生成结果目录已定向忽略。

## 大文件风险

已发现的典型大文件：

- `CalculateShapValues/AttentionAndNoAttention/Seed_*/*_ShapValues.npz`：单个约 2.8GB。
- `CalculateShapValues/NoAttention/Seed_*/*.npz`：单个约 351MB。
- `ModelTrain/NoAttention/SHAP_freq/ShapResults/*.npz`：单个约 961MB。
- `wavelet_dataset/Bearing_30_2.npz`, `Gear_30_2.npz`：单个约 5.6GB。
- `DataSets/SEU/*/*.csv`：单个约 76MB。
- `DAG/PureData/*.npz`, `DAG/Noise/*.npz`：单个约 48MB。

建议：

- 原始数据、SHAP 原始数组、小波 `.npz`、模型权重默认不要上传普通 Git。
- 需要公开复现时，用 Zenodo/OSF/Google Drive/学校数据仓库，并在 README 写下载链接和校验 hash。
- 如果必须放入 GitHub，使用 Git LFS，但应控制总量，避免把 40GB SHAP 结果直接纳入仓库。

## 敏感信息扫描

轻量扫描范围：Python、文本、配置候选文件。未发现 `api key`, `secret`, `password`, `credential`, 私钥、`/Users/`、`/Volumes/` 等敏感内容。

发现的路径问题主要是相对路径和 Windows 分隔符，不是密钥：

- `..\\wavelet_dataset`
- `.\\trained_models\\GNN`
- `../DataSets/...`

## 建议使用 Git LFS 的模式

`.gitattributes` 已为以下类型提供 LFS 规则：

- `*.pth`, `*.pt`, `*.ckpt`, `*.onnx`, `*.onnx.data`
- `*.npz`, `*.npy`, `*.mat`, `*.h5`, `*.pkl`, `*.joblib`
- `*.rar`, `*.zip`, `*.7z`

如果这些文件被 `.gitignore` 忽略，则不会进入 Git；LFS 规则只在你人工选择跟踪某些大文件时生效。

## 上传 GitHub 前人工确认

1. 是否要把当前目录初始化为仓库，还是把整理后的源码复制到一个干净仓库。
2. 是否公开原始 SEU/CWRU/XJTU-SY 数据；需确认数据许可。
3. 哪些论文图需要提交到 `docs/figures` 或 `paper/figures`。
4. 是否补一个 `README.md` 和 `requirements.txt/environment.yml`。
5. 是否先统一大小写路径和 `.npz` 键名，避免 Linux 用户无法复现。
6. 是否修复 `DAG/CreateDAG.py`, `CNNNetWorkWithAttention` 参数、`GNNCausalSEU.forward` 后再公开。

## 推荐命令

```bash
git init
git status --short
git check-ignore -v wavelet_dataset/Bearing_30_2.npz
git check-ignore -v CalculateShapValues/AttentionAndNoAttention/Seed_42/Bearing_20_0_ShapValues.npz
git ls-files | sort
find . -type f -size +50M
```

如果已经误跟踪大文件，不要直接删除工作区文件。建议使用：

```bash
git rm --cached <large-file>
git status --short
```

如需迁移历史中的大文件，需另行使用 `git filter-repo` 或 BFG，执行前应备份仓库。

