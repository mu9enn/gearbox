
# Author Result Request Packet

## 11.1 WeChat Short Message

老师/同学好，我这边现在不是在质疑之前的结果，而是在做投稿前的 evidence lock：把论文里每一个表、图、指标都和原始结果文件、seed、split、config 对上，避免后面写 SCI 时证据链不完整。现在本地已经能整理出 SHAP、PC-DAG 和一致性审计主线，但缺少几份原始数值结果和设置确认。麻烦你方便时把下面清单里的原始指标表/日志/配置发我，或者确认哪些图就是最终版本。我们不会提交大模型或大缓存到 Git，只需要确认论文证据来源。

## 11.2 Detailed Checklist

1. NoAttention 6ch baseline: original per-condition test accuracy, macro-F1, weighted-F1, precision, recall.
2. NoAttention 6ch baseline: per-seed metrics for Seed_42/49/56/63/70, if available.
3. Original confusion matrix numeric values or classification report for each SEU condition.
4. Exact dataset/split used for the reported baseline results.
5. Whether original checkpoint input width was 512, 1024, or another preprocessing setting.
6. Whether current `wavelet_dataset/*.npz` matches the result-producing dataset.
7. Confirmation that `CalculateShapValues/NoAttention/Seed_*/*.npz` are final SHAP caches for the intended NoAttention checkpoints.
8. Confirmation of SHAP background/test samples and random seed/config.
9. Confirmation of PC-DAG input: Seed_49 PC dataset, average PC dataset, or both.
10. Confirmation of final PC-KCI alpha/significance setting for paper figures.
11. Confirmation whether filtered PC-DAG or original PC-DAG is the final figure style.
12. Confirmation whether physical prior/background knowledge is part of the final method or only exploratory.
13. If attention comparison is retained: original attention/no-attention numeric metrics.
14. If CWRU/GNN/cross-condition is retained as supplement: original numeric metrics and caveat notes.

## 11.3 Requested Files

- `baseline_noattention_6ch_metrics.csv` or original logs containing accuracy/F1/classification report.
- `baseline_noattention_6ch_confusion_matrices.csv` if available.
- `train_test_split_description.md` or config file describing split and dataset generation.
- `shap_generation_config.md` or original SHAP script/log command.
- `pcdag_generation_config.md` or original PC-KCI command/alpha/filter setting.
- Any final figure source tables for SHAP/PC-DAG if different from the local repository.
- Optional supplement metric files for attention, CWRU, GNN, and cross-condition.

## 11.4 Placement Convention

Please place small confirmed tables/config notes under:

`reports/collaborator_confirmed_results/`

Suggested names:

- `reports/collaborator_confirmed_results/baseline_noattention_6ch_metrics.csv`
- `reports/collaborator_confirmed_results/baseline_noattention_6ch_confusion_matrices.csv`
- `reports/collaborator_confirmed_results/dataset_split_contract.md`
- `reports/collaborator_confirmed_results/shap_config_confirmation.md`
- `reports/collaborator_confirmed_results/pcdag_setting_confirmation.md`

Do not commit large `.npz`, `.pth`, `.pt`, SHAP caches, checkpoint folders, or raw datasets. Large files can stay local or be shared separately; Git should only track small tables and text confirmations.
