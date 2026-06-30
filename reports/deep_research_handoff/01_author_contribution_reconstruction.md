
# Author Contribution Reconstruction

This file reconstructs the collaborator's work from local code, existing result assets, and prior audit/planning reports. It is not manuscript text.

## 1.1 One-Sentence Summary

The collaborator built a rotating-machinery fault-diagnosis and diagnostic-logic audit workflow that combines angle-domain/wavelet multichannel vibration representations, CNN diagnosis, SHAP channel/frequency attribution, SHAP-weighted dual-view PC-DAG construction, and conservative attribution-graph consistency analysis, with additional exploratory branches for attention comparison, causal reliable-edge GNNs, CWRU validation, cross-condition testing, and noise/pure-signal processing.

## 1.2 Implemented Core Technical Modules

### Data Preprocessing / Wavelet Features

- Local evidence paths: `Common/LoadDatasets.py`, `wavelet_dataset/*.npz`, `DataSets/SEU/`.
- Experiment assets: four SEU condition wavelet datasets; audit notes in `PROJECT_AUDIT.md` and `docs/EXPERIMENTS_LOGIC_AND_SCI_STORYLINE.md`.
- Possible contribution: transform raw multichannel vibration into consistent wavelet tensors for diagnosis and attribution.
- Credibility/limits: implementation exists, but current local data contract has historical width and split-provenance caveats; cross-condition splits should not support strong robustness claims.

### CNN / NoAttention Baseline

- Local evidence paths: `Common/NetWorkFrame.py`, `Common/ModelTrainAndVisiable.py`, `ModelTrain/NoAttention/model_train_6ch/`, `ModelTrain/NoAttention/SavedGraphs_6ch/`, `ModelTrain/NoAttention/ConfusionAndF1/`.
- Experiment assets: training curves, checkpoints, confusion/F1 plots.
- Possible contribution: provide the diagnostic backbone to be audited by SHAP and PC-DAG.
- Credibility/limits: collaborator visual assets exist, but trusted structured baseline metrics are not currently indexed; local crop-width evaluation is diagnostic only.

### Attention Comparison

- Local evidence paths: `ModelTrain/AttentionAndNoAttention/`, `Analysis/ChannelWeights.py`.
- Experiment assets: attention/no-attention saved graphs, confusion/F1 figures, channel-weight caches.
- Possible contribution: auxiliary comparison showing how attention changes model behavior or channel weighting.
- Credibility/limits: useful supplement candidate, not part of the frozen main contribution unless collaborator metrics are recovered.

### SHAP Attribution

- Local evidence paths: `CalculateShapValues/NoAttention/`, `CalculateShapValues/AttentionAndNoAttention/`, `reports/paper_evidence/shap_rank_summary.csv`, `reports/paper_evidence/shap_seed_stability_summary.csv`.
- Experiment assets: SHAP caches, per-seed SHAP figures, total SHAP channel/frequency figures.
- Possible contribution: quantify model reliance on vibration channels and wavelet frequency bands.
- Credibility/limits: strong local evidence exists; source SHAP cache lacks full config snapshot and is too large to version.

### SHAP-to-PC Dataset

- Local evidence paths: `Common/PCDataset(wavelet).py`, `PC_Datasets/Seed_49/`, `PC_Datasets/average/`.
- Experiment assets: condition-level channel/frequency PC input CSVs and feature-distribution figures.
- Possible contribution: bridge attribution and graph discovery by weighting PC-DAG features using SHAP-derived importance.
- Credibility/limits: central to the intended contribution; generation protocol and seed/average choice need final documentation.

### Channel-View PC-DAG

- Local evidence paths: `Common/PCCausal.py`, `DAG/WaveletDAG.py`, `DAG/PC_DAG/Seed_49/**/_ch.png`, `reports/paper_evidence/pcdag_edges/`.
- Experiment assets: channel-view original/filtered PC-DAG figures and exported edge tables.
- Possible contribution: sensor/channel-level graph audit of attribution logic.
- Credibility/limits: edge direction must be conservative; only `feature_to_label` is strict directional support.

### Frequency-View PC-DAG

- Local evidence paths: `DAG/PC_DAG/Seed_49/**/_freq.png`, `reports/paper_evidence/pcdag_edges/`.
- Experiment assets: frequency-view original/filtered PC-DAG figures and edge summaries.
- Possible contribution: wavelet-scale/frequency-level graph audit of attribution logic.
- Credibility/limits: strong main-topic asset, but final alpha/filtered/average setting should be frozen.

### Consistency / Shortcut-Risk Analysis

- Local evidence paths: `reports/paper_evidence/consistency/`, `reports/paper_evidence/final_tables/`.
- Experiment assets: strict consistency table, weak adjacency table, shortcut-risk candidates, underused graph-supported candidates.
- Possible contribution: identify attribution-graph agreement, conflict, shortcut-risk candidates, and underused graph-supported features.
- Credibility/limits: the analysis is an evidence-organization layer over collaborator outputs; do not overclaim confirmed shortcuts or causal mechanisms.

### GNN / Reliable Edge

- Local evidence paths: `Common/CausalAndDoWhy.py`, `Common/NetWorkFrame.py`, `GNNCausal/`.
- Experiment assets: reliable-edge tensors, GNN training curves, GNN checkpoints.
- Possible contribution: an exploratory causal-structure-enhanced diagnosis extension.
- Credibility/limits: future-work/risk branch; previous audit flagged that `GNNCausalSEU.forward` appears to compute propagated features but classify from `x`, so it should not be a main contribution now.

### CWRU

- Local evidence paths: `CWRU/WT.py`, `CWRU/ModelTrainAndCrossTest.py`, `CWRU/train_curves/GNN/`, `CWRU/trained_models/GNN/`.
- Experiment assets: CWRU GNN training curves and checkpoints.
- Possible contribution: external validation attempt on public bearing data.
- Credibility/limits: currently figure/model assets without a paper-ready structured metric table; supplement/future only unless collaborator metrics are recovered.

### Cross-Condition

- Local evidence paths: `ModelTrain/NoAttention/CrossCondTest/`, `reports/reproducibility_audit/full_split_overlap_report.csv`.
- Experiment assets: cross-condition accuracy comparison figure.
- Possible contribution: explored robustness/generalization.
- Credibility/limits: current split overlap risk means it should be limitation/risk, not strong robustness evidence.

### Noise / Pure Data / DAG Processing

- Local evidence paths: `DAG/data_treat.py`, `DAG/PureData/`, `DAG/Noise/`, `DAG/PolyCurves/`.
- Experiment assets: pure/noise caches and curve visualizations.
- Possible contribution: exploratory physical signal processing and robustness/causal visualization.
- Credibility/limits: not connected to a final metrics table; future/support only.

## 1.3 Most Likely Original Paper Mainline

The most faithful reconstruction is not a pure model-performance paper and not a GNN paper. The code and result layout suggest that the collaborator intended to move from a wavelet-CNN diagnostic model to explanation and causal-graph auditing:

1. Build wavelet multichannel inputs from rotating-machinery vibration.
2. Train a CNN/no-attention diagnostic model and compare optional attention/ablation branches.
3. Compute SHAP attribution over channels and wavelet bands.
4. Convert SHAP-weighted features into channel-view and frequency-view PC-DAG inputs.
5. Compare high-attribution features with label-related graph structure.
6. Use agreement/conflict patterns to discuss diagnostic logic, shortcut-risk candidates, and graph-supported but underused features.

The strongest current mainline is therefore an attribution-graph consistency audit for rotating-machinery fault diagnosis, with GNN, CWRU, cross-condition, attention, and robustness as supporting or future branches.

## 1.4 Later Evidence-Organization Tools Added By Codex

These tools/reports help organize evidence but should not be presented as the collaborator's original method contribution:

- `tools/extract_paper_evidence_tables.py`
- `tools/export_pcdag_edges.py`
- `tools/build_consistency_table.py`
- `tools/summarize_pcdag_edge_policy.py`
- `tools/diagnose_checkpoint_dataset_mismatch.py`
- `tools/export_baseline_metrics.py`
- `tools/build_paper_evidence_package.py`
- `tools/build_final_paper_evidence_package.py`
- `reports/paper_evidence/**`
- `reports/paper_planning/**`
- `reports/deep_research_handoff/**`

The collaborator's original contribution is embodied in the preprocessing, CNN/attention training, SHAP generation, PC_Datasets, PC-DAG figures, reliable-edge/GNN/CWRU branches, and existing result assets. Codex's later scripts are audit, extraction, indexing, and planning aids.

## 1.5 Quantitative Local Evidence Snapshot

- SHAP seed-stability rows: 44.
- SHAP per-seed rank rows: 220.
- Exported PC-DAG edges: 159 total, 31 label-related.
- Strict consistency counts: {'low_shap_without_strict_causal_support': 32, 'low_shap_with_strict_causal_support_underused_feature': 3, 'high_shap_without_strict_causal_support_potential_shortcut': 8, 'high_shap_strict_causal_support': 1}.
- Weak adjacency support counts: {'no_label_adjacency': 18, 'strict_directed_support': 4, 'weak_label_adjacency_only': 22}.
- Shortcut-risk candidate rows: 10.
- Underused graph-supported candidate rows: 22.

## 1.6 Key Caveat

The local `--crop-width 512` baseline evaluation is diagnostic-only and should not be treated as collaborator evidence. It signals a provenance gap in the current local code/data/checkpoint bundle.
