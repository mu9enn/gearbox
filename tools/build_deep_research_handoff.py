#!/usr/bin/env python3
"""Build the final local evidence handoff package for ChatGPT deep research."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import pandas as pd


CONDITIONS = ["Bearing_20_0", "Bearing_30_2", "Gear_20_0", "Gear_30_2"]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def exists(root: Path, rel: str) -> str:
    return rel if (root / rel).exists() else "needs_asset_path_confirmation"


def shap_figure(root: Path, condition: str) -> str:
    return exists(root, f"CalculateShapValues/NoAttention/TotalAnalysisGraphs/SHAP_{condition}.png")


def dag_figure(root: Path, condition: str, view: str) -> str:
    suffix = "ch" if view == "channel" else "freq"
    candidates = [
        f"DAG/PC_DAG/Seed_49/filtered/Significance_0.001/average/{condition}_{suffix}.png",
        f"DAG/PC_DAG/Seed_49/filtered/Significance_0.001/{condition}_{suffix}.png",
        f"DAG/PC_DAG/Seed_49/filtered/Significance_0.05/average/{condition}_{suffix}.png",
        f"DAG/PC_DAG/Seed_49/filtered/Significance_0.05/{condition}_{suffix}.png",
    ]
    for candidate in candidates:
        if (root / candidate).exists():
            return candidate
    return "needs_asset_path_confirmation"


def val_counts(df: pd.DataFrame, column: str) -> dict:
    if df.empty or column not in df.columns:
        return {}
    return dict(Counter(df[column].fillna("").astype(str)))


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return ""
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")).replace("\n", " ") for col in columns) + " |")
    return "\n".join([header, sep, *body])


def build_author_contribution(root: Path, out: Path, stats: dict) -> None:
    text = f"""
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

- SHAP seed-stability rows: {stats['shap_rows']}.
- SHAP per-seed rank rows: {stats['shap_rank_rows']}.
- Exported PC-DAG edges: {stats['pc_edges_total']} total, {stats['pc_label_edges']} label-related.
- Strict consistency counts: {stats['strict_counts']}.
- Weak adjacency support counts: {stats['adj_counts']}.
- Shortcut-risk candidate rows: {stats['shortcut_rows']}.
- Underused graph-supported candidate rows: {stats['underused_rows']}.

## 1.6 Key Caveat

The local `--crop-width 512` baseline evaluation is diagnostic-only and should not be treated as collaborator evidence. It signals a provenance gap in the current local code/data/checkpoint bundle.
"""
    write_text(out / "01_author_contribution_reconstruction.md", text)


def build_digest(root: Path, out: Path, tables: dict, stats: dict) -> None:
    rows = [
        {
            "result_id": "R001",
            "result_family": "wavelet_dataset",
            "asset_path": "wavelet_dataset/*.npz",
            "condition": "four SEU conditions",
            "dataset": "SEU",
            "model_or_method": "angle resampling + SWT/db4 wavelet",
            "view": "multichannel wavelet tensor",
            "seed_or_setting": "data preprocessing",
            "metric_or_content": "wavelet tensor files",
            "numeric_value_or_summary": "4 condition npz files; current local width 1024",
            "result_form": "large_npz_data",
            "is_original_collaborator_result": "true",
            "is_later_extraction_or_diagnostic": "false",
            "paper_use_status": "planning_reference",
            "known_caveat": "large local data not versioned; split/provenance caveats",
            "notes": "Use as method premise, not as direct paper table.",
        },
        {
            "result_id": "R002",
            "result_family": "baseline_training_curves",
            "asset_path": "ModelTrain/NoAttention/SavedGraphs_6ch/Seed_*",
            "condition": "four SEU conditions",
            "dataset": "SEU",
            "model_or_method": "NoAttention CNN 6ch",
            "view": "diagnostic baseline",
            "seed_or_setting": "Seed_42/49/56/63/70 visible",
            "metric_or_content": "training/test curves",
            "numeric_value_or_summary": "plot-only collaborator assets",
            "result_form": "png_figures",
            "is_original_collaborator_result": "true",
            "is_later_extraction_or_diagnostic": "false",
            "paper_use_status": "needs_collaborator_numeric_confirmation",
            "known_caveat": "no trusted structured metrics indexed",
            "notes": "Needed for baseline context; do not substitute crop-width diagnostic numbers.",
        },
        {
            "result_id": "R003",
            "result_family": "confusion_f1",
            "asset_path": "ModelTrain/NoAttention/ConfusionAndF1/Seed_49",
            "condition": "four SEU conditions",
            "dataset": "SEU",
            "model_or_method": "NoAttention CNN 6ch",
            "view": "class-level diagnosis",
            "seed_or_setting": "Seed_49",
            "metric_or_content": "confusion matrix and F1 figures",
            "numeric_value_or_summary": "plot-only; classification report table missing",
            "result_form": "png_figures",
            "is_original_collaborator_result": "true",
            "is_later_extraction_or_diagnostic": "false",
            "paper_use_status": "supplement_candidate",
            "known_caveat": "plot-only evidence",
            "notes": "Use in supplement after numeric report recovery.",
        },
        {
            "result_id": "R004",
            "result_family": "shap_rank_summary",
            "asset_path": "reports/paper_evidence/shap_rank_summary.csv",
            "condition": "four SEU conditions",
            "dataset": "SEU",
            "model_or_method": "NoAttention CNN + SHAP",
            "view": "channel/frequency",
            "seed_or_setting": "5 seeds",
            "metric_or_content": "per-seed SHAP rank",
            "numeric_value_or_summary": f"{stats['shap_rank_rows']} rows",
            "result_form": "csv_table",
            "is_original_collaborator_result": "false",
            "is_later_extraction_or_diagnostic": "true",
            "paper_use_status": "main_candidate",
            "known_caveat": "derived from collaborator SHAP cache",
            "notes": "Extraction table supports stability and rank interpretation.",
        },
        {
            "result_id": "R005",
            "result_family": "shap_seed_stability",
            "asset_path": "reports/paper_evidence/shap_seed_stability_summary.csv",
            "condition": "four SEU conditions",
            "dataset": "SEU",
            "model_or_method": "NoAttention CNN + SHAP",
            "view": "channel/frequency",
            "seed_or_setting": "5 seeds",
            "metric_or_content": "mean importance, mean rank, top-k frequency",
            "numeric_value_or_summary": f"{stats['shap_rows']} rows; top-k frequency uses rank<=3",
            "result_form": "csv_table",
            "is_original_collaborator_result": "false",
            "is_later_extraction_or_diagnostic": "true",
            "paper_use_status": "main_candidate",
            "known_caveat": "source SHAP cache lacks full config manifest",
            "notes": "Best current table for attribution stability.",
        },
        {
            "result_id": "R006",
            "result_family": "shap_figures",
            "asset_path": "CalculateShapValues/NoAttention/TotalAnalysisGraphs",
            "condition": "four SEU conditions",
            "dataset": "SEU",
            "model_or_method": "NoAttention CNN + SHAP",
            "view": "channel/frequency",
            "seed_or_setting": "total analysis",
            "metric_or_content": "SHAP attribution plots",
            "numeric_value_or_summary": "all-condition and per-condition SHAP figures",
            "result_form": "png_figures",
            "is_original_collaborator_result": "true",
            "is_later_extraction_or_diagnostic": "false",
            "paper_use_status": "main_candidate",
            "known_caveat": "may need publication redraw",
            "notes": "Candidate Fig. 2 material.",
        },
        {
            "result_id": "R007",
            "result_family": "pc_dataset",
            "asset_path": "PC_Datasets/Seed_49; PC_Datasets/average",
            "condition": "four SEU conditions",
            "dataset": "SEU",
            "model_or_method": "SHAP-weighted PC input",
            "view": "channel/frequency",
            "seed_or_setting": "Seed_49 and average",
            "metric_or_content": "feature-label CSVs",
            "numeric_value_or_summary": "8 condition-view CSVs plus average variants",
            "result_form": "csv_tables",
            "is_original_collaborator_result": "true",
            "is_later_extraction_or_diagnostic": "false",
            "paper_use_status": "main_candidate",
            "known_caveat": "generation contract should be documented",
            "notes": "Method bridge from SHAP to PC-DAG.",
        },
        {
            "result_id": "R008",
            "result_family": "pcdag_edges",
            "asset_path": "reports/paper_evidence/pcdag_edges/pcdag_edges_long.csv",
            "condition": "four SEU conditions",
            "dataset": "SEU",
            "model_or_method": "PC-KCI PC-DAG",
            "view": "channel/frequency",
            "seed_or_setting": "Seed_49 and average; alpha 0.05",
            "metric_or_content": "edge export",
            "numeric_value_or_summary": f"{stats['pc_edges_total']} total edges; {stats['pc_label_edges']} label-related; relations {stats['label_relation_counts']}",
            "result_form": "csv_table",
            "is_original_collaborator_result": "false",
            "is_later_extraction_or_diagnostic": "true",
            "paper_use_status": "main_candidate",
            "known_caveat": "direction policy must be conservative",
            "notes": "Use only feature_to_label as strict directional support.",
        },
        {
            "result_id": "R009",
            "result_family": "pcdag_figures",
            "asset_path": "DAG/PC_DAG/Seed_49",
            "condition": "four SEU conditions",
            "dataset": "SEU",
            "model_or_method": "PC-KCI PC-DAG",
            "view": "channel/frequency",
            "seed_or_setting": "multiple alpha settings; filtered/original; average variants",
            "metric_or_content": "PC-DAG figures",
            "numeric_value_or_summary": "figure directories exist for channel/frequency PC-DAGs",
            "result_form": "png_figures",
            "is_original_collaborator_result": "true",
            "is_later_extraction_or_diagnostic": "false",
            "paper_use_status": "main_candidate",
            "known_caveat": "final alpha/filtered setting must be frozen",
            "notes": "Candidate Fig. 3 material.",
        },
        {
            "result_id": "R010",
            "result_family": "strict_consistency",
            "asset_path": "reports/paper_evidence/consistency/strict_feature_to_label_consistency_table.csv",
            "condition": "four SEU conditions",
            "dataset": "SEU",
            "model_or_method": "SHAP + PC-DAG audit",
            "view": "channel/frequency",
            "seed_or_setting": "top-k=2 strict policy",
            "metric_or_content": "strict consistency types",
            "numeric_value_or_summary": str(stats["strict_counts"]),
            "result_form": "csv_table",
            "is_original_collaborator_result": "false",
            "is_later_extraction_or_diagnostic": "true",
            "paper_use_status": "main_candidate",
            "known_caveat": "strict support sparse",
            "notes": "Strongest directional-support layer.",
        },
        {
            "result_id": "R011",
            "result_family": "weak_adjacency",
            "asset_path": "reports/paper_evidence/consistency/adjacency_weak_consistency_table.csv",
            "condition": "four SEU conditions",
            "dataset": "SEU",
            "model_or_method": "SHAP + PC-DAG audit",
            "view": "channel/frequency",
            "seed_or_setting": "top-k=2 weak adjacency policy",
            "metric_or_content": "adjacency support levels",
            "numeric_value_or_summary": str(stats["adj_counts"]),
            "result_form": "csv_table",
            "is_original_collaborator_result": "false",
            "is_later_extraction_or_diagnostic": "true",
            "paper_use_status": "main_candidate",
            "known_caveat": "weak adjacency is not causal direction",
            "notes": "Broad graph-consistency/conflict audit layer.",
        },
        {
            "result_id": "R012",
            "result_family": "shortcut_candidates",
            "asset_path": "reports/paper_evidence/final_tables/table_shortcut_risk_candidates.csv",
            "condition": "four SEU conditions",
            "dataset": "SEU",
            "model_or_method": "SHAP + PC-DAG audit",
            "view": "channel/frequency",
            "seed_or_setting": "derived from strict and weak consistency",
            "metric_or_content": "high-SHAP graph-unsupported candidates",
            "numeric_value_or_summary": f"{stats['shortcut_rows']} rows; {stats['shortcut_counts']}",
            "result_form": "csv_table",
            "is_original_collaborator_result": "false",
            "is_later_extraction_or_diagnostic": "true",
            "paper_use_status": "main_candidate",
            "known_caveat": "candidate only, not confirmed shortcut",
            "notes": "Candidate Fig. 4 case source.",
        },
        {
            "result_id": "R013",
            "result_family": "underused_candidates",
            "asset_path": "reports/paper_evidence/final_tables/table_underused_causal_candidates.csv",
            "condition": "four SEU conditions",
            "dataset": "SEU",
            "model_or_method": "SHAP + PC-DAG audit",
            "view": "channel/frequency",
            "seed_or_setting": "derived from strict and weak consistency",
            "metric_or_content": "low-SHAP graph-supported candidates",
            "numeric_value_or_summary": f"{stats['underused_rows']} rows; {stats['underused_counts']}",
            "result_form": "csv_table",
            "is_original_collaborator_result": "false",
            "is_later_extraction_or_diagnostic": "true",
            "paper_use_status": "main_candidate",
            "known_caveat": "candidate interpretation, not intervention-confirmed mechanism",
            "notes": "Secondary audit output.",
        },
        {
            "result_id": "R014",
            "result_family": "attention_comparison",
            "asset_path": "ModelTrain/AttentionAndNoAttention",
            "condition": "four SEU conditions",
            "dataset": "SEU",
            "model_or_method": "Attention vs NoAttention CNN",
            "view": "diagnostic and channel weights",
            "seed_or_setting": "Seed_42/49/56 visible",
            "metric_or_content": "training graphs, confusion/F1, weights",
            "numeric_value_or_summary": "plot/cache assets; structured metric table missing",
            "result_form": "figures_and_npz_cache",
            "is_original_collaborator_result": "true",
            "is_later_extraction_or_diagnostic": "false",
            "paper_use_status": "supplement_candidate",
            "known_caveat": "not central contribution",
            "notes": "Use only as support if needed.",
        },
        {
            "result_id": "R015",
            "result_family": "cross_condition",
            "asset_path": "ModelTrain/NoAttention/CrossCondTest/Cross_condition_acc_comparison.png",
            "condition": "cross-condition",
            "dataset": "SEU",
            "model_or_method": "NoAttention variants",
            "view": "generalization",
            "seed_or_setting": "legacy result",
            "metric_or_content": "cross-condition accuracy plot",
            "numeric_value_or_summary": "plot-only; split overlap risk",
            "result_form": "png_figure",
            "is_original_collaborator_result": "true",
            "is_later_extraction_or_diagnostic": "false",
            "paper_use_status": "limitation_only",
            "known_caveat": "train/cross_test exact duplicate risk",
            "notes": "Do not use as robustness claim.",
        },
        {
            "result_id": "R016",
            "result_family": "gnn",
            "asset_path": "GNNCausal/train_curves/GNN; GNNCausal/trained_models/GNN; GNNCausal/*_reliable_edge.pt",
            "condition": "four SEU conditions",
            "dataset": "SEU",
            "model_or_method": "reliable-edge GNN",
            "view": "causal GNN",
            "seed_or_setting": "legacy GNN branch",
            "metric_or_content": "train curves, checkpoints, edge tensors",
            "numeric_value_or_summary": "assets exist; trusted metrics absent",
            "result_form": "figures_models_cache",
            "is_original_collaborator_result": "true",
            "is_later_extraction_or_diagnostic": "false",
            "paper_use_status": "future_work",
            "known_caveat": "implementation risk documented",
            "notes": "Do not force into main contribution.",
        },
        {
            "result_id": "R017",
            "result_family": "cwru",
            "asset_path": "CWRU/train_curves/GNN; CWRU/trained_models/GNN",
            "condition": "CWRU conditions",
            "dataset": "CWRU",
            "model_or_method": "GNN external branch",
            "view": "external validation",
            "seed_or_setting": "cond_0..3",
            "metric_or_content": "training curves and checkpoints",
            "numeric_value_or_summary": "plot/model assets; no structured metrics indexed",
            "result_form": "figures_models",
            "is_original_collaborator_result": "true",
            "is_later_extraction_or_diagnostic": "false",
            "paper_use_status": "future_work",
            "known_caveat": "not aligned with frozen main contribution yet",
            "notes": "Can become supplement if metrics are recovered.",
        },
        {
            "result_id": "R018",
            "result_family": "noise_pure_data",
            "asset_path": "DAG/PureData; DAG/Noise; DAG/PolyCurves",
            "condition": "fault-class signal assets",
            "dataset": "SEU",
            "model_or_method": "pure/noise preprocessing and curve visualization",
            "view": "physical/noise analysis",
            "seed_or_setting": "legacy DAG preprocessing",
            "metric_or_content": "pure/noise caches and curve plots",
            "numeric_value_or_summary": "assets exist; formal robustness metrics absent",
            "result_form": "npz_cache_and_png_figures",
            "is_original_collaborator_result": "true",
            "is_later_extraction_or_diagnostic": "false",
            "paper_use_status": "future_work",
            "known_caveat": "not connected to final evidence chain",
            "notes": "Optional future robustness/physical-support branch.",
        },
        {
            "result_id": "R019",
            "result_family": "crop_width_diagnostic",
            "asset_path": "reports/paper_evidence/baseline_metrics/baseline_metrics_paper_table.csv",
            "condition": "four SEU conditions",
            "dataset": "current local wavelet_dataset",
            "model_or_method": "NoAttention checkpoints + local crop-width eval",
            "view": "local reproducibility",
            "seed_or_setting": "crop_width=512",
            "metric_or_content": "near-random diagnostic metrics",
            "numeric_value_or_summary": "diagnostic only, not paper evidence",
            "result_form": "csv_table",
            "is_original_collaborator_result": "false",
            "is_later_extraction_or_diagnostic": "true",
            "paper_use_status": "diagnostic_only",
            "known_caveat": "provenance gap between current code/data/checkpoints",
            "notes": "Do not use as collaborator result.",
        },
    ]
    fields = [
        "result_id",
        "result_family",
        "asset_path",
        "condition",
        "dataset",
        "model_or_method",
        "view",
        "seed_or_setting",
        "metric_or_content",
        "numeric_value_or_summary",
        "result_form",
        "is_original_collaborator_result",
        "is_later_extraction_or_diagnostic",
        "paper_use_status",
        "known_caveat",
        "notes",
    ]
    write_csv(out / "02_local_experiment_result_digest.csv", rows, fields)


def build_claim_matrix(out: Path) -> None:
    rows = [
        ["C01", "Wavelet multichannel representation supports rotating machinery diagnosis", "moderate", "Common/LoadDatasets.py; wavelet_dataset/*.npz; ModelTrain result assets", "data contract and training assets, but not a final numeric table", "wavelet tensor and training curve assets", "trusted baseline metrics/provenance still need confirmation", "could overstate performance if local provenance gap ignored", "moderate_claim_with_caveat", "Use as method premise and recover collaborator metrics."],
        ["C02", "CNN/NoAttention model provides diagnostic backbone", "moderate pending numeric confirmation", "ModelTrain/NoAttention/SavedGraphs_6ch; ConfusionAndF1; checkpoints", "trusted collaborator metrics missing", "training curves and confusion/F1 figures", "numeric metrics table or formal rerun", "crop-width diagnostic could be mistaken for true result", "moderate_claim_with_caveat", "Recover collaborator baseline numeric metrics first."],
        ["C03", "SHAP reveals channel/frequency attribution patterns", "strong", "CalculateShapValues/NoAttention; SHAP figures; shap_rank_summary.csv", "220 per-seed rank rows; 44 stability rows", "TotalAnalysisGraphs and seed analysis graphs", "publication-style figure selection", "do not overinterpret SHAP as causal", "strong_claim_possible", "Use as main attribution evidence."],
        ["C04", "SHAP attribution is stable across seeds", "strong", "shap_seed_stability_summary.csv", "44 rows with mean/std rank and top-k frequency", "SHAP summary figures", "format final top-k table", "source cache config incomplete", "strong_claim_possible", "Use as main Table 2 candidate."],
        ["C05", "SHAP-weighted features can construct dual-view PC-DAG inputs", "strong", "PC_Datasets/Seed_49; PC_Datasets/average; Common/PCDataset(wavelet).py", "condition-view feature-label CSVs exist", "feature distribution plots", "document seed/average protocol", "generation contract/key history should be clarified", "strong_claim_possible", "Use as method bridge."],
        ["C06", "Channel-view PC-DAG provides sensor-level graph audit", "moderate-to-strong", "DAG/PC_DAG/Seed_49/*_ch.png; pcdag edge summary", "edge export includes channel label-related edges", "channel DAG figures", "freeze alpha and filtered/average setting", "edge direction overclaim", "moderate_claim_with_caveat", "Use conservative graph-audit wording."],
        ["C07", "Frequency-view PC-DAG provides scale/frequency-level graph audit", "moderate-to-strong", "DAG/PC_DAG/Seed_49/*_freq.png; pcdag edge summary", "frequency label-related edges and strict cD2/cD5 cases", "frequency DAG figures", "freeze final figure setting", "weak adjacency not causal direction", "moderate_claim_with_caveat", "Use conservative graph-audit wording."],
        ["C08", "Attribution-graph consistency can identify graph-supported high-attribution features", "moderate", "strict_feature_to_label_consistency_table.csv; adjacency_weak_consistency_table.csv", "1 high-SHAP strict support; 7 high-SHAP label adjacency", "SHAP and DAG paired figures", "select case and explain sparse strict support", "broad causal direction claim would be too strong", "moderate_claim_with_caveat", "Use case-study-level support."],
        ["C09", "High-attribution but graph-unsupported features are shortcut-risk candidates", "moderate", "table_shortcut_risk_candidates.csv", "10 candidate rows: 8 strict, 2 weak", "Gear_20_0 SHAP/DAG figures", "avoid confirmed-shortcut language", "shortcut not validated by intervention", "moderate_claim_with_caveat", "Use screening/candidate language."],
        ["C10", "Low-attribution but graph-supported features are underused candidate features", "moderate", "table_underused_causal_candidates.csv", "22 candidate rows; 3 strict", "Bearing_20_0/Bearing_30_2 paired figures", "choose concise secondary case", "underused feature not proven mechanism", "weak_claim_only", "Use as supplement or secondary audit output."],
        ["C11", "Attention comparison supports interpretability claims", "weak-to-moderate", "ModelTrain/AttentionAndNoAttention; ChannelWeights", "cache/figures present; structured metrics missing", "attention/no-attention curves and confusion figures", "numeric summary and relation to SHAP needed", "could distract from main contribution", "weak_claim_only", "Keep supplement-only."],
        ["C12", "CWRU supports external validation", "weak now", "CWRU/train_curves/GNN; CWRU/trained_models/GNN", "structured metric table absent", "CWRU GNN train curves", "protocol and metrics needed", "plot-only validation would be weak", "future_work_only", "Treat as future/supplement until metrics recovered."],
        ["C13", "GNN reliable-edge extension improves diagnosis", "do not claim now", "GNNCausal; gnn_risk_report", "assets exist but implementation risk documented", "GNN training curves", "repair and formal verification needed", "would overextend paper and create credibility risk", "future_work_only", "Keep as future work."],
        ["C14", "Cross-condition robustness can be strongly claimed", "do not claim", "CrossCondTest figure; full_split_overlap_report.csv", "overlap audit supports limitation, not robustness", "cross-condition plot", "clean split/rerun needed", "leakage risk", "limitation_only", "Mention only as limitation/risk."],
    ]
    fields = [
        "claim_id",
        "candidate_claim",
        "claim_strength_if_written_now",
        "supporting_assets",
        "supporting_numeric_evidence",
        "supporting_figures",
        "current_gap",
        "risk_if_overclaimed",
        "recommended_wording_level",
        "recommended_next_action",
    ]
    write_csv(out / "03_claim_evidence_gap_matrix.csv", [dict(zip(fields, r)) for r in rows], fields)


def build_case_pack(root: Path, out: Path) -> None:
    cases = [
        ("CASE01", "strict_consistent_case", "Bearing_30_2", "frequency", "cD2", "mean_rank=1.0; high SHAP; topk_frequency=1.0", "feature_to_label strict support", "strict", "strict table: high_shap_strict_causal_support", "High attribution feature with strict PC-DAG support", "Best strict positive case; useful to show that attribution can align with graph support.", "Strict support is sparse; do not generalize to all high-SHAP features.", "main case candidate"),
        ("CASE02", "weak_adjacency_consistent_case", "Bearing_20_0", "frequency", "cD2", "mean_rank=1.0; high SHAP; topk_frequency=1.0", "weak_label_adjacency_only", "weak", "weak table: high_shap_with_label_adjacency", "High attribution feature has label-related graph adjacency without directional claim", "Useful to explain weak adjacency layer.", "No strict feature_to_label support; avoid causal-direction wording.", "main or supplement case candidate"),
        ("CASE03", "weak_adjacency_consistent_case", "Bearing_20_0", "frequency", "cD3", "mean_rank=2.0; high SHAP; topk_frequency=1.0", "weak_label_adjacency_only", "weak", "weak table: high_shap_with_label_adjacency", "High SHAP plus weak graph adjacency", "Pairs naturally with cD2 to show frequency-view audit.", "Weak adjacency only.", "supplement case"),
        ("CASE04", "shortcut_risk_candidate", "Gear_20_0", "channel", "ch8", "mean_rank=1.4; high SHAP", "no_label_adjacency", "weak negative", "shortcut table: weak_shortcut_risk_candidate", "High-attribution but graph-unsupported shortcut-risk candidate", "Strong candidate because both strict direction and weak adjacency are absent.", "Candidate only; not a confirmed shortcut.", "main Fig. 4 candidate"),
        ("CASE05", "shortcut_risk_candidate", "Gear_20_0", "frequency", "cD4", "mean_rank=2.0; high SHAP", "no_label_adjacency", "weak negative", "shortcut table: weak_shortcut_risk_candidate", "High-attribution but graph-unsupported frequency shortcut-risk candidate", "Useful if frequency-view case is preferred over channel-view case.", "Candidate only; no intervention confirmation.", "main or supplement case"),
        ("CASE06", "shortcut_risk_candidate", "Gear_30_2", "frequency", "cD2", "mean_rank=1.0; high SHAP", "no strict feature_to_label support but weak adjacency exists", "strict negative / weak positive", "shortcut table: strict_shortcut_risk_candidate; weak adjacency table", "Strictly unsupported high SHAP feature", "Shows why strict and weak policies differ.", "Not as strong a shortcut candidate because weak adjacency exists.", "supplement case"),
        ("CASE07", "underused_graph_supported_candidate", "Bearing_20_0", "channel", "ch3", "low SHAP; mean_rank=4.2", "strict_feature_to_label_support", "strict", "underused table: strict_underused_graph_supported_candidate", "Low-attribution feature with strict graph support", "Useful to show model may underuse graph-supported channel.", "Underused is a candidate interpretation.", "secondary main or supplement case"),
        ("CASE08", "underused_graph_supported_candidate", "Bearing_20_0", "channel", "ch7", "low SHAP; mean_rank=5.0", "strict_feature_to_label_support", "strict", "underused table: strict_underused_graph_supported_candidate", "Low-attribution feature with strict graph support", "Second channel-view strict underused case.", "May be redundant with ch3.", "supplement case"),
        ("CASE09", "underused_graph_supported_candidate", "Bearing_30_2", "frequency", "cD5", "low SHAP; strict support", "strict_feature_to_label_support", "strict", "underused table: strict_underused_graph_supported_candidate", "Low-attribution frequency feature with strict graph support", "Pairs with Bearing_30_2/cD2 to show high and low SHAP strict support in one condition.", "Needs careful explanation.", "supplement case"),
        ("CASE10", "negative_or_limitation_case", "cross-condition", "generalization", "not_applicable", "not_applicable", "split overlap risk", "limitation", "full_split_overlap_report.csv; CrossCondTest figure", "Cross-condition evidence cannot support strong robustness", "Prevents overclaiming and shows honest limitation.", "Do not frame as positive result.", "limitation note"),
    ]
    rows = []
    for case in cases:
        case_id, case_type, condition, view, feature, shap_summary, support, strict_or_weak, related, claim, why, risk, use = case
        shap = shap_figure(root, condition) if condition in CONDITIONS else "not_applicable"
        dag = dag_figure(root, condition, view) if condition in CONDITIONS and view in {"channel", "frequency"} else "not_applicable"
        rows.append(
            {
                "case_id": case_id,
                "case_type": case_type,
                "condition": condition,
                "view": view,
                "feature_name": feature,
                "shap_rank_or_importance_summary": shap_summary,
                "graph_support_type": support,
                "strict_or_weak": strict_or_weak,
                "related_table_rows": related,
                "shap_figure_candidates": shap,
                "dag_figure_candidates": dag,
                "paper_claim_supported": claim,
                "why_this_case_is_useful": why,
                "risk_or_caveat": risk,
                "recommended_use": use,
            }
        )
    fields = [
        "case_id",
        "case_type",
        "condition",
        "view",
        "feature_name",
        "shap_rank_or_importance_summary",
        "graph_support_type",
        "strict_or_weak",
        "related_table_rows",
        "shap_figure_candidates",
        "dag_figure_candidates",
        "paper_claim_supported",
        "why_this_case_is_useful",
        "risk_or_caveat",
        "recommended_use",
    ]
    write_csv(out / "04_main_case_study_asset_pack.csv", rows, fields)


def build_figure_table_pack(out: Path) -> None:
    rows = [
        ["FIG1", "main_figure", "Framework overview: wavelet-CNN-SHAP-PC-DAG audit workflow", "to_draw_by_author; code pipeline; docs/EXPERIMENTS_LOGIC_AND_SCI_STORYLINE.md", "deep_research_handoff/01_author_contribution_reconstruction.md", "all", "pipeline", "End-to-end collaborator workflow and audit logic", "needs_drawing", "yes", "Method overview", "No polished figure exists", "Draw after ChatGPT finalizes storyline"],
        ["FIG2", "main_figure", "SHAP channel/frequency attribution examples", "CalculateShapValues/NoAttention/TotalAnalysisGraphs/*.png", "shap_seed_stability_summary.csv; table_shap_stability_main.csv", "four SEU conditions", "channel/frequency", "Model attribution patterns and multi-seed ranking", "existing asset candidate", "yes", "Attribution analysis", "SHAP figures may need style unification", "Select 1 all-condition figure plus 1 condition panel"],
        ["FIG3", "main_figure", "Dual-view PC-DAG examples", "DAG/PC_DAG/Seed_49/filtered/Significance_0.001/average/*.png", "pcdag_edge_label_summary.csv; pcdag_edges_long.csv", "four SEU conditions", "channel/frequency", "Channel-view and frequency-view graph audit", "existing asset candidate", "yes", "Graph audit", "Alpha/filtered/average policy must be frozen", "Use selected average filtered DAGs"],
        ["FIG4", "main_figure", "Attribution-graph consistency and shortcut-risk case study", "SHAP figures + DAG figures for selected cases", "04_main_case_study_asset_pack.csv; shortcut/underused tables", "selected cases", "channel/frequency", "Consistency, conflict, shortcut-risk, and underused candidates", "needs composition", "yes", "Case study", "Candidates are not confirmed causal mechanisms", "Build multi-panel case figure after ChatGPT chooses case"],
        ["TBL1", "main_table", "Collaborator baseline metric status", "ModelTrain/NoAttention/SavedGraphs_6ch; ConfusionAndF1", "paper_planning/codex_local_repro_gap_note.md", "four SEU conditions", "diagnostic baseline", "Shows baseline metrics are needed but local crop eval is diagnostic-only", "missing trusted numeric table", "no", "Baseline context", "Do not use crop-width metrics", "Recover collaborator numeric metrics or plan formal rerun"],
        ["TBL2", "main_table", "SHAP top-k stability ranking", "CalculateShapValues/NoAttention/Seed_*", "shap_seed_stability_summary.csv; table_shap_stability_main.csv", "four SEU conditions", "channel/frequency", "Multi-seed attribution stability", "ready with cleanup", "minor", "Attribution analysis", "Config provenance caveat", "Round/rename columns for manuscript"],
        ["TBL3", "main_table", "Strict/weak attribution-graph audit", "PC_Datasets; DAG/PC_DAG", "strict_feature_to_label_consistency_table.csv; adjacency_weak_consistency_table.csv", "four SEU conditions", "channel/frequency", "Strict support and weak adjacency consistency/conflict", "ready with caveat", "minor", "Graph audit", "Weak adjacency is not causal direction", "Use conservative wording"],
        ["TBL4", "optional_table", "Support/ablation summary", "ModelTrain/NoAttention/ModelTest_ch; ModelTest_freq; AttentionAndNoAttention", "needs recovered metrics", "four SEU conditions", "channel/frequency/attention", "Optional support for engineering variants", "not ready", "no", "Supplement", "plot-only currently", "Use only if numeric metrics recovered"],
        ["SUP1", "supplement", "Confusion/F1 figures", "ModelTrain/**/ConfusionAndF1", "needs classification_report table", "four SEU conditions", "diagnosis", "Class-level behavior", "plot-only", "maybe", "Supplement", "No structured report", "Recover/export class-level metrics"],
        ["SUP2", "supplement", "Attention comparison", "ModelTrain/AttentionAndNoAttention", "needs metric table", "four SEU conditions", "attention/no-attention", "Auxiliary model comparison", "partial", "maybe", "Supplement", "Could distract from main topic", "Keep concise"],
        ["SUP3", "supplement", "CWRU external branch", "CWRU/train_curves/GNN; CWRU/trained_models/GNN", "needs metric table", "CWRU", "GNN/external", "Potential external validation", "future/supplement", "yes", "Supplement/future", "GNN-centric and incomplete", "Use only after metrics/protocol recovered"],
        ["SUP4", "supplement", "GNN risk/future branch", "GNNCausal/**", "gnn_risk_report.md", "SEU", "causal GNN", "Reliable-edge extension idea", "future only", "no", "Future work", "Implementation risk", "Do not include as main claim"],
        ["SUP5", "limitation", "Cross-condition limitation", "ModelTrain/NoAttention/CrossCondTest", "full_split_overlap_report.csv", "SEU", "cross-condition", "Why robustness is not claimed strongly", "ready as limitation", "no", "Limitations", "Overlap/leakage risk", "Mention honestly, do not use as proof"],
        ["SUP6", "supplement", "PC-DAG edge policy", "reports/paper_evidence/pcdag_edges; consistency", "edge_policy_summary.csv; edge_policy_notes.md", "four SEU conditions", "graph policy", "Strict vs weak edge interpretation", "ready", "no", "Methods/supplement", "Requires conservative wording", "Use to preempt causal overclaim"],
        ["SUP7", "limitation", "Local reproducibility/provenance gap", "baseline_metrics_paper_table.csv", "codex_local_repro_gap_note.md", "local current assets", "baseline diagnostic", "Why local crop eval is not paper evidence", "ready", "no", "Limitations/internal note", "Do not overemphasize in manuscript", "Use for planning and reproducibility appendix only"],
    ]
    fields = [
        "paper_item_id",
        "paper_item_type",
        "proposed_title_or_role",
        "source_assets",
        "data_tables",
        "conditions",
        "views",
        "what_it_demonstrates",
        "current_readiness",
        "needs_redraw_or_cleanup",
        "paper_section_candidate",
        "risk_or_caveat",
        "recommended_next_action",
    ]
    write_csv(out / "05_figure_table_materials_pack.csv", [dict(zip(fields, r)) for r in rows], fields)


def build_followup_blueprint(out: Path) -> None:
    rows = [
        ["P0", "Recover collaborator baseline numeric metrics", "Establish trustworthy diagnostic baseline context", "High-level journal reviewers will expect performance context before explanation/audit", "ModelTrain/NoAttention curves, checkpoints, confusion/F1 figures", "Search local logs; ask collaborator; parse original result exports if found", "condition-level accuracy/F1 mean/std and classification reports", "small-to-medium", "metrics may be unavailable", "Baseline remains plot-only and weaker", "true", "true"],
        ["P0", "Formal baseline rerun if original metrics cannot be recovered", "Produce trusted metrics under frozen protocol", "Needed if baseline table is required and collaborator metrics absent", "training scripts and wavelet data contract", "Define protocol first; do not run in current phase", "metrics table, config snapshot, seed list, git hash", "medium-to-large", "data/provenance mismatch must be resolved first", "Paper needs caveated or delayed baseline table", "true", "true"],
        ["P0", "Freeze SHAP + PC-DAG consistency main table", "Lock main evidence for audit contribution", "Central contribution must be clean and conservative", "SHAP stability, PC-DAG edges, strict/weak tables", "Finalize columns, labels, selected top-k policy", "publication-ready Table 2/3", "small", "overly complex table could confuse", "ChatGPT lacks stable evidence skeleton", "true", "false"],
        ["P0", "Select main case study", "Provide concrete narrative example", "SCI paper needs interpretable case, not only aggregate tables", "04_main_case_study_asset_pack.csv", "Choose strict/weak/shortcut/underused cases", "final case shortlist with figure assets", "small", "case choice may need human taste", "Story remains abstract", "true", "true"],
        ["P0", "Clarify PC-DAG edge policy", "Prevent causal overclaim", "Causal claims must be defensible", "edge_policy_notes.md; pcdag_direction_sanity_report.md", "Freeze wording: strict feature_to_label vs weak adjacency", "methods note and supplement policy table", "small", "reviewers may challenge PC-DAG direction", "Causal story becomes risky", "true", "true"],
        ["P1", "Numerical classification reports", "Support baseline/confusion figures", "Supplement needs structured metrics", "ConfusionAndF1 figures, checkpoints/logs", "Recover existing table or plan formal export", "classification_report_long.csv from trusted source", "small-to-medium", "may require original data protocol", "F1 evidence remains plot-only", "true", "false"],
        ["P1", "SHAP top-k table polishing", "Make attribution evidence manuscript-ready", "Improves readability and claim clarity", "shap_seed_stability_summary.csv", "Rename/round columns, map feature labels, add interpretation labels", "clean Table 2 candidate", "small", "none significant", "Manual cleanup later", "true", "false"],
        ["P1", "Redraw selected PC-DAG figures", "Make graph evidence publication-quality", "Current figures may be busy/inconsistent", "selected DAG PNG/PDF assets", "Prepare figure spec, not rerun PC-KCI unless later requested", "clean Fig. 3 panels", "medium", "requires style/design decision", "Figures may look exploratory", "true", "true"],
        ["P1", "Attention comparison numeric整理", "Optional supplement and discussion", "May help explain why no-attention is audited", "AttentionAndNoAttention assets", "Recover or export metrics later", "supplementary table", "medium", "could distract main story", "Attention remains background only", "true", "true"],
        ["P1", "Config/provenance manifest", "Improve reproducibility narrative", "High-level journal expects traceability", "existing scripts and reports", "Record data version, input width, checkpoint, seed, SHAP and PC settings", "manifest CSV/MD", "small-to-medium", "some historical details missing", "Methods section remains caveated", "true", "true"],
        ["P2", "CWRU external validation", "Assess transfer to public bearing dataset", "Could strengthen SCI evidence if clean", "CWRU assets", "Recover metrics or design rerun", "external validation table", "medium-to-large", "currently GNN-centric/incomplete", "Paper lacks external validation", "true", "true"],
        ["P2", "Noise robustness", "Test audit stability under noise", "Robustness improves application relevance", "DAG/Noise; DAG/PolyCurves/Noise", "Design formal robustness protocol", "noise sensitivity table", "medium", "new experiment not current phase", "Robustness claim omitted", "true", "true"],
        ["P2", "Missing channel / sensor failure", "Evaluate practical sensor degradation", "Rotating machinery deployments face sensor loss", "channel ablation assets; model variants", "Design formal missing-channel test", "robustness table", "medium", "requires protocol and metrics", "Application claim weaker", "true", "true"],
        ["P2", "GNN repair and validation", "Test reliable-edge extension", "Only needed if GNN becomes future paper/support", "GNNCausal assets and risk report", "Fix/verify algorithm then rerun later", "trusted GNN metrics", "large", "implementation risk", "GNN remains future work", "true", "true"],
        ["P2", "Clean cross-condition rerun", "Support limited robustness claim", "Could strengthen paper if leakage-free", "CrossCondTest scripts and split audit", "Define clean split first", "cross-condition metrics with no overlap", "large", "data protocol risk", "Cross-condition remains limitation only", "true", "true"],
    ]
    fields = [
        "priority",
        "experiment_or_task",
        "scientific_purpose",
        "why_needed_for_sci",
        "existing_foundation",
        "implementation_entry_points",
        "expected_outputs",
        "estimated_effort",
        "risk",
        "if_not_done_consequence",
        "codex_can_execute_later",
        "requires_human_or_collaborator_input",
    ]
    write_csv(out / "06_followup_experiment_blueprint.csv", [dict(zip(fields, r)) for r in rows], fields)


def build_questions(out: Path) -> None:
    text = """
# Original Author / Collaborator Intent Questions

These questions are intended for the human researcher or collaborator. They focus only on issues that can change the paper evidence chain.

## Baseline And Data Provenance

1. Where are the original baseline numeric result tables for the NoAttention CNN? Are they in logs, notebooks, spreadsheets, or only figures?
2. Which model family, seed set, and condition set were intended as the final diagnostic baseline in the original work?
3. What is the historical relationship between the 512-width checkpoint/SHAP input and the current 1024-width `wavelet_dataset` files?
4. Which data split contract should be considered the official one for the original paper: historical lowercase `train_data/test_data`, current uppercase `train_set/test_set`, or another archive?
5. Were the existing confusion/F1 figures produced from the same checkpoints used for SHAP?

## SHAP And Attribution

6. Which checkpoints exactly correspond to the existing SHAP caches under `CalculateShapValues/NoAttention/Seed_*`?
7. Were SHAP background and explanation samples chosen from train/test intentionally, and how should this be described?
8. Were channel and frequency SHAP rankings intended as final quantitative results or mainly as visualization?
9. Should the paper emphasize channel attribution, frequency attribution, or the contrast between both?

## PC-DAG And Causal Graphs

10. Which PC-DAG setting was intended for the final paper: Seed_49, average, alpha 0.05, alpha 0.001, filtered, original, or another combination?
11. How did the original author intend to interpret `label_to_feature` edges? Should they be discarded, treated as adjacency, or discussed as orientation instability?
12. Is the physical prior/background knowledge meant to strictly forbid label -> feature in the final graph, or only guide graph visualization?
13. Should the final claim use causal-direction language or safer graph-consistency/audit language?
14. Were shortcut-risk candidates an intended original contribution, or are they a post-hoc framing of attribution-graph conflicts?

## Auxiliary Branches

15. Was GNN intended as a main contribution, a future extension, or only an exploratory experiment?
16. If GNN was intended as a contribution, should `GNNCausalSEU.forward` classify from propagated features rather than the raw `x` tensor?
17. Was CWRU intended as external validation for this paper or as a separate follow-up?
18. Were cross-condition results intended for the paper despite the current split-overlap risk?
19. Are noise/pure-data assets part of a planned robustness experiment or only signal-processing exploration?

## Paper Framing

20. What should the paper primarily emphasize: diagnosis performance, interpretability, causal/graph audit, shortcut-risk screening, or engineering application?
21. Which 1-2 conditions/cases does the collaborator consider most convincing?
22. Which figures were originally planned as main figures?
23. Are there unpublished slides, thesis figures, spreadsheets, or logs that better capture the original result story?
24. What claim would the collaborator most want reviewers to remember?
"""
    write_text(out / "07_original_author_intent_questions.md", text)


def build_prompt(out: Path, stats: dict) -> None:
    text = f"""
# ChatGPT Deep Research Prompt

You are ChatGPT taking over the deep research, literature positioning, contribution design, and SCI Q1 paper planning for the GearBox project. Use the local facts prepared by Codex, but do not treat Codex's diagnostic scripts as the original scientific contribution.

## Current Project Facts

- Topic area: rotating machinery / gearbox / bearing fault diagnosis.
- Main available pipeline: raw multichannel vibration -> angle-domain and wavelet tensor representation -> CNN/no-attention diagnosis -> SHAP channel/frequency attribution -> SHAP-weighted PC_Datasets -> channel-view and frequency-view PC-DAG -> attribution-graph consistency/conflict audit.
- Key local evidence directories: `ModelTrain/`, `CalculateShapValues/`, `PC_Datasets/`, `DAG/PC_DAG/`, `GNNCausal/`, `CWRU/`, `DAG/PureData`, `DAG/Noise`.
- Key handoff files: `reports/deep_research_handoff/*`, `reports/paper_planning/*`, `reports/paper_evidence/*`.

## Collaborator Core Contribution To Respect

The collaborator's contribution appears to be a diagnostic-logic audit framework, not merely a classifier:

1. Build wavelet multichannel representations for rotating-machinery diagnosis.
2. Train/use CNN diagnosis models as the target of explanation.
3. Compute SHAP attribution over channels and wavelet frequency bands.
4. Convert SHAP-weighted features into dual-view PC-DAG inputs.
5. Use channel-view and frequency-view graph structures to audit attribution logic.
6. Identify consistency, conflict, shortcut-risk candidates, and underused graph-supported features.

Do not rewrite this as a GNN paper unless the human researcher explicitly asks.

## Existing Evidence

- SHAP stability: {stats['shap_rows']} rows in `shap_seed_stability_summary.csv`; {stats['shap_rank_rows']} per-seed SHAP rank rows.
- PC-DAG edge export: {stats['pc_edges_total']} total edges and {stats['pc_label_edges']} label-related edges.
- Strict feature-to-label support counts: {stats['strict_counts']}.
- Weak adjacency support counts: {stats['adj_counts']}.
- Shortcut-risk candidates: {stats['shortcut_rows']} rows.
- Underused graph-supported candidates: {stats['underused_rows']} rows.
- Existing figures: SHAP total graphs, PC-DAG figures, training curves, confusion/F1 figures, attention comparison figures, GNN/CWRU curves.

## Evidence Gaps

- Trusted collaborator baseline numeric metrics are not currently indexed.
- Local `--crop-width 512` baseline evaluation is diagnostic only and near-random; do not treat it as collaborator result failure.
- PC-DAG direction must be conservative: `feature_to_label` is strict support; `label_to_feature` and undirected label-related edges are weak adjacency only.
- Cross-condition evidence has split-overlap risk and should not support strong robustness claims.
- GNN has implementation/provenance risks and should remain future work unless repaired and verified.
- CWRU and attention assets are currently support/future unless numeric tables are recovered.

## Recommended Storyline

Prioritize Storyline A:

Attribution-graph consistency audit for trustworthy rotating machinery diagnosis.

Possible phrasing direction: an explainable diagnostic logic audit framework that combines SHAP attribution with dual-view causal/graph discovery to identify attribution-graph agreement, conflict, shortcut-risk candidates, and underused graph-supported features.

## Literature Directions To Research

1. Explainable AI for rotating machinery fault diagnosis: SHAP, Grad-CAM, integrated gradients, attention explanations.
2. Shortcut learning and spurious correlation auditing in fault diagnosis / condition monitoring.
3. Causal discovery or causal graph use in mechanical fault diagnosis.
4. PC/PC-KCI, causal adjacency, and limitations of causal direction interpretation in observational data.
5. Wavelet/time-frequency representations for gearbox/bearing diagnosis.
6. Trustworthy/reliable AI for rotating machinery and industrial PHM.
7. Multi-view explanation: sensor/channel view and frequency/time-scale view.
8. Papers combining attribution with causal discovery, or comparing attribution and causal graph structure.

## Academic Questions To Decide

- Is the strongest novelty the SHAP-to-dual-view-PC-DAG bridge, the attribution-graph consistency audit, or shortcut-risk screening?
- Is "causal" acceptable in the title, or should the wording use "graph consistency" / "causality-adjacency" to stay conservative?
- How much baseline performance is required for a high-level journal if the main focus is diagnostic logic audit?
- Should attention, CWRU, GNN, and cross-condition be supplement/future only?
- Which 1-2 case studies best demonstrate the method without overclaiming?

## Follow-Up Experiments To Design

P0:

- Recover collaborator baseline numeric metrics, or define a formal rerun protocol later.
- Freeze main SHAP + PC-DAG consistency tables.
- Select main case study.
- Freeze PC-DAG edge policy and graph figure settings.

P1:

- Structured classification report / confusion matrix table.
- SHAP top-k table polishing.
- Publication-quality SHAP/DAG/case figures.
- Config/provenance manifest for baseline/SHAP/PC-DAG.

P2:

- CWRU external validation if metrics can be recovered.
- Noise robustness / missing-channel robustness.
- GNN repair and verification only if a future extension is desired.
- Clean cross-condition rerun only as support, not current main evidence.

## Do Not Drift

- Do not treat the local crop-width reproduction gap as proof that collaborator results failed.
- Do not force GNN, CWRU, or cross-condition into the main contribution.
- Do not overclaim causal direction; use strict feature_to_label only for directional support.
- Do not write the paper as a generic high-accuracy classifier paper.
- Do not present Codex's extraction/audit scripts as the collaborator's original scientific method.
- Keep the paper aligned with the collaborator's original assets and likely intent.

## Requested Deep Research Output Structure

Please produce a report with:

1. Recommended SCI Q1 positioning and target-journal style.
2. Final main storyline and 3-5 contribution bullets.
3. Claim-evidence matrix with conservative wording.
4. Required figures/tables and what each should show.
5. Minimal follow-up experiment plan with P0/P1/P2.
6. Literature map and how this work differs from prior SHAP/XAI/causal-diagnosis papers.
7. Risks, limitations, and wording constraints.
8. Questions for the human researcher/collaborator before drafting.
"""
    write_text(out / "08_chatgpt_deep_research_prompt.md", text)


def build_readme(out: Path) -> None:
    text = """
# Deep Research Handoff Package

This directory is the final local-facts handoff from Codex to ChatGPT before deep literature research and SCI paper planning.

Recommended reading order:

1. `README.md` - this guide.
2. `01_author_contribution_reconstruction.md` - reconstructed collaborator contribution and original-intent interpretation.
3. `02_local_experiment_result_digest.csv` - asset/result inventory, including main/supplement/risk/future classifications.
4. `03_claim_evidence_gap_matrix.csv` - candidate paper claims, evidence, gaps, and safe wording level.
5. `04_main_case_study_asset_pack.csv` - concrete case candidates for the main story and figures.
6. `05_figure_table_materials_pack.csv` - proposed figures/tables and available assets.
7. `06_followup_experiment_blueprint.csv` - P0/P1/P2 follow-up tasks for a high-level SCI paper.
8. `07_original_author_intent_questions.md` - questions that should be asked of the collaborator/human researcher.
9. `08_chatgpt_deep_research_prompt.md` - prompt for ChatGPT's next-stage deep research report.

Important caveats:

- The local `--crop-width 512` evaluation is diagnostic only and not paper evidence.
- The strongest current mainline is SHAP attribution plus dual-view PC-DAG attribution-graph consistency audit.
- GNN, CWRU, cross-condition, and attention should not be forced into the main contribution unless the human researcher later decides otherwise.
- Causal wording must stay conservative: `feature_to_label` is strict support; `label_to_feature` and undirected label-related edges are weak adjacency only.
"""
    write_text(out / "README.md", text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="reports/deep_research_handoff")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    out = (root / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    tables = {
        "shap": read_csv(root / "reports/paper_evidence/shap_seed_stability_summary.csv"),
        "shap_rank": read_csv(root / "reports/paper_evidence/shap_rank_summary.csv"),
        "pc_edges": read_csv(root / "reports/paper_evidence/pcdag_edges/pcdag_edges_long.csv"),
        "pc_summary": read_csv(root / "reports/paper_evidence/pcdag_edges/pcdag_edge_label_summary.csv"),
        "strict": read_csv(root / "reports/paper_evidence/consistency/strict_feature_to_label_consistency_table.csv"),
        "adj": read_csv(root / "reports/paper_evidence/consistency/adjacency_weak_consistency_table.csv"),
        "shortcut": read_csv(root / "reports/paper_evidence/final_tables/table_shortcut_risk_candidates.csv"),
        "underused": read_csv(root / "reports/paper_evidence/final_tables/table_underused_causal_candidates.csv"),
    }
    stats = {
        "shap_rows": len(tables["shap"]),
        "shap_rank_rows": len(tables["shap_rank"]),
        "pc_edges_total": len(tables["pc_edges"]),
        "pc_label_edges": int(tables["pc_edges"].get("is_label_related", pd.Series(dtype=bool)).astype(str).str.lower().eq("true").sum()) if not tables["pc_edges"].empty else 0,
        "label_relation_counts": val_counts(tables["pc_edges"], "label_relation_type"),
        "strict_counts": val_counts(tables["strict"], "strict_consistency_type"),
        "adj_counts": val_counts(tables["adj"], "adjacency_support_level"),
        "shortcut_rows": len(tables["shortcut"]),
        "shortcut_counts": val_counts(tables["shortcut"], "candidate_type"),
        "underused_rows": len(tables["underused"]),
        "underused_counts": val_counts(tables["underused"], "candidate_type"),
    }

    build_author_contribution(root, out, stats)
    build_digest(root, out, tables, stats)
    build_claim_matrix(out)
    build_case_pack(root, out)
    build_figure_table_pack(out)
    build_followup_blueprint(out)
    build_questions(out)
    build_prompt(out, stats)
    build_readme(out)
    print(f"Wrote deep research handoff package to {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
