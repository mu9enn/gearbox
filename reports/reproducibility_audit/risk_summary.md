# Reproducibility Audit Risk Summary

## P0

- Detected sampled exact overlap between `train` and an evaluation/cross split; examples: wavelet_dataset/Bearing_20_0.npz cross_test~train dup=27; wavelet_dataset/Gear_20_0.npz cross_test~train dup=26. Treat affected results as non-paper-grade until strict split hashes are regenerated.
- `Common/PCDataset(wavelet).py`-style `all_data/all_labels` expectations are not found in local scanned npz files; SHAP-to-PC data contract needs confirmation.
- `GNNCausalSEU.forward` computes propagated `out` but feeds `x` into the classifier; do not use SEU GNN as core causal-enhancement evidence.
- `GNNCausal/CausalGNN.py` contains multiple `fit_transform` calls, including finetune refit; scaler provenance is inconsistent.

## P1

- 61 large local research assets are not versioned; keep them out of Git but record hash/config/manifest before using as evidence.
- 6 npz key usages were not matched to scanned local npz keys; inspect `npz_key_usage_report.csv` before reruns.

## P2

- Reports are metadata only; they do not replace config snapshots, metrics JSON/CSV, checkpoint manifests, or git-hash capture.
- Asset manifest uses file metadata and simple role heuristics; paper-grade provenance still needs explicit experiment manifests.

## Suggested Next Minimal Engineering Tasks

- Add a strict no-overlap split generator and persist split indices/hash summaries.
- Add a small experiment manifest writer for seed, config, metrics, checkpoint, git hash, and source data hash.
- Fix or quarantine SEU GNN causal propagation before any causal-enhancement claim.
- Normalize SHAP-to-PC input/output contracts before rerunning attribution-causality experiments.
