#!/usr/bin/env python3
"""Build the Stage 2 evidence-lock package from existing local assets.

This script is deliberately read-only with respect to experiments: it does not
train, evaluate checkpoints, recompute SHAP, run PC-KCI, or touch datasets.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import pandas as pd


CONDITIONS = ["Bearing_20_0", "Bearing_30_2", "Gear_20_0", "Gear_30_2"]
SEEDS = [42, 49, 56, 63, 70]
VIEWS = ["channel", "frequency"]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def rel_if_exists(root: Path, rel: str) -> str:
    return rel if (root / rel).exists() else "missing_or_needs_path_confirmation"


def join_paths(paths: list[str]) -> str:
    return "; ".join(paths)


def counts_by(df: pd.DataFrame, column: str) -> str:
    if df.empty or column not in df.columns:
        return "not_available"
    counts = Counter(df[column].fillna("NA").astype(str))
    return "; ".join(f"{k}={v}" for k, v in sorted(counts.items()))


def pcdag_figure(root: Path, condition: str, view: str, seed_or_average: str) -> str:
    suffix = "ch" if view == "channel" else "freq"
    if seed_or_average == "average":
        candidates = [
            f"DAG/PC_DAG/Seed_49/filtered/Significance_0.05/average/{condition}_{suffix}.png",
            f"DAG/PC_DAG/Seed_49/filtered/Significance_0.001/average/{condition}_{suffix}.png",
            f"DAG/PC_DAG/Seed_49/original/Significance_0.05/average/{condition}_{suffix}.png",
            f"DAG/PC_DAG/Seed_49/original/Significance_0.001/average/{condition}_{suffix}.png",
        ]
    else:
        candidates = [
            f"DAG/PC_DAG/Seed_49/filtered/Significance_0.05/{condition}_{suffix}.png",
            f"DAG/PC_DAG/Seed_49/filtered/Significance_0.001/{condition}_{suffix}.png",
            f"DAG/PC_DAG/Seed_49/original/Significance_0.05/{condition}_{suffix}.png",
            f"DAG/PC_DAG/Seed_49/original/Significance_0.001/{condition}_{suffix}.png",
        ]
    for candidate in candidates:
        if (root / candidate).exists():
            return candidate
    return "missing_or_needs_path_confirmation"


def shap_figure(root: Path, condition: str) -> str:
    return rel_if_exists(root, f"CalculateShapValues/NoAttention/TotalAnalysisGraphs/SHAP_{condition}.png")


def load_tables(root: Path) -> dict[str, pd.DataFrame]:
    return {
        "baseline_inventory": read_csv(root / "reports/paper_evidence/baseline_metric_inventory.csv"),
        "crop512_metrics": read_csv(root / "reports/paper_evidence/baseline_metrics/baseline_metrics_paper_table.csv"),
        "checkpoint_mismatch": read_csv(root / "reports/paper_evidence/baseline_metrics/checkpoint_dataset_mismatch_report.csv"),
        "shap_stability": read_csv(root / "reports/paper_evidence/shap_seed_stability_summary.csv"),
        "shap_rank": read_csv(root / "reports/paper_evidence/shap_rank_summary.csv"),
        "pcdag_edges": read_csv(root / "reports/paper_evidence/pcdag_edges/pcdag_edges_long.csv"),
        "pcdag_summary": read_csv(root / "reports/paper_evidence/pcdag_edges/pcdag_edge_label_summary.csv"),
        "strict": read_csv(root / "reports/paper_evidence/consistency/strict_feature_to_label_consistency_table.csv"),
        "weak": read_csv(root / "reports/paper_evidence/consistency/adjacency_weak_consistency_table.csv"),
        "consistency_summary": read_csv(root / "reports/paper_evidence/consistency/consistency_summary.csv"),
        "case_pack": read_csv(root / "reports/deep_research_handoff/04_main_case_study_asset_pack.csv"),
        "figure_pack": read_csv(root / "reports/deep_research_handoff/05_figure_table_materials_pack.csv"),
    }


def build_baseline_metrics_status(root: Path, out: Path, tables: dict[str, pd.DataFrame]) -> None:
    fields = [
        "metric_item",
        "condition",
        "dataset",
        "model_or_method",
        "seed_or_setting",
        "asset_path",
        "asset_type",
        "numeric_metric_available",
        "metric_fields_found",
        "paper_use_status",
        "known_caveat",
        "recommended_action",
        "requires_collaborator_confirmation",
        "notes",
    ]
    rows: list[dict] = []

    for cond in CONDITIONS:
        rows.append(
            {
                "metric_item": "trusted_no_attention_baseline_numeric_metric",
                "condition": cond,
                "dataset": "SEU wavelet_dataset historical",
                "model_or_method": "NoAttention 6ch",
                "seed_or_setting": "multi_seed_expected",
                "asset_path": "missing_in_local_versioned_assets",
                "asset_type": "missing_needs_collaborator",
                "numeric_metric_available": "no",
                "metric_fields_found": "",
                "paper_use_status": "blocked_needs_collaborator_confirmation",
                "known_caveat": "Local search found plots/checkpoints but not a trusted collaborator original numeric metrics table.",
                "recommended_action": "Ask collaborator for original test accuracy/F1/classification report with seed/config/split provenance.",
                "requires_collaborator_confirmation": "yes",
                "notes": "Do not fill this gap with the local crop-width-512 diagnostic artifact.",
            }
        )

    crop = tables["crop512_metrics"]
    for _, row in crop.iterrows():
        rows.append(
            {
                "metric_item": "local_crop512_reproducibility_diagnostic",
                "condition": row.get("condition", ""),
                "dataset": "local wavelet_dataset plus crop-width 512 assumption",
                "model_or_method": row.get("model_family", "no_attention_6ch"),
                "seed_or_setting": f"n_seeds={row.get('n_seeds', '')}; crop_width={row.get('crop_width', '')}",
                "asset_path": "reports/paper_evidence/baseline_metrics/baseline_metrics_paper_table.csv",
                "asset_type": "local_diagnostic_crop512_artifact",
                "numeric_metric_available": "yes",
                "metric_fields_found": "accuracy_mean,accuracy_std,macro_f1_mean,macro_f1_std,weighted_f1_mean,weighted_f1_std",
                "paper_use_status": "local_diagnostic_crop512_artifact_only_not_paper_evidence",
                "known_caveat": "Near-random local values reflect current code/data/checkpoint provenance gap; human researcher instructed not to pursue as paper evidence.",
                "recommended_action": "Keep only as provenance diagnostic; do not cite in manuscript performance table.",
                "requires_collaborator_confirmation": "no",
                "notes": str(row.get("notes", "")),
            }
        )

    plot_assets = [
        (
            "no_attention_training_curves",
            "ModelTrain/NoAttention/SavedGraphs_6ch",
            "collaborator_plot_only_result",
            "Training curves exist for Seed_42/49/56/63/70; numeric endpoint table not found.",
        ),
        (
            "no_attention_confusion_f1_figures",
            "ModelTrain/NoAttention/ConfusionAndF1/Seed_49",
            "collaborator_plot_only_result",
            "Confusion matrix/F1 PNG assets exist; OCR-free numeric table not found.",
        ),
        (
            "attention_vs_no_attention_figures",
            "ModelTrain/AttentionAndNoAttention/SavedGraphs",
            "collaborator_plot_only_result",
            "Attention/no-attention curves and figures exist; supplement only unless metrics are recovered.",
        ),
        (
            "cross_condition_accuracy_plot",
            "ModelTrain/NoAttention/CrossCondTest/Cross_condition_acc_comparison.png",
            "collaborator_plot_only_result",
            "Cross split has duplicate/overlap risk; not strong robustness evidence.",
        ),
        (
            "gnn_training_curves",
            "GNNCausal/train_curves/GNN",
            "collaborator_plot_only_result",
            "GNN branch is future/supplement only due implementation-risk audit.",
        ),
        (
            "cwru_gnn_curves",
            "CWRU/train_curves/GNN",
            "collaborator_plot_only_result",
            "CWRU branch lacks trusted paper-ready structured metrics locally.",
        ),
    ]
    for item, path, asset_type, caveat in plot_assets:
        rows.append(
            {
                "metric_item": item,
                "condition": "multiple_or_external",
                "dataset": "local existing assets",
                "model_or_method": item,
                "seed_or_setting": "existing local result assets",
                "asset_path": rel_if_exists(root, path),
                "asset_type": asset_type,
                "numeric_metric_available": "plot_only",
                "metric_fields_found": "",
                "paper_use_status": "supplement_or_reference_only_needs_numeric_recovery",
                "known_caveat": caveat,
                "recommended_action": "Recover original metric CSV/log from collaborator before using as quantitative evidence.",
                "requires_collaborator_confirmation": "yes",
                "notes": "Treat plots as provenance pointers, not final metric sources.",
            }
        )

    ckpt = tables["checkpoint_mismatch"]
    if not ckpt.empty:
        for _, row in ckpt.head(8).iterrows():
            rows.append(
                {
                    "metric_item": "checkpoint_dataset_width_provenance",
                    "condition": row.get("condition", ""),
                    "dataset": row.get("dataset_path", ""),
                    "model_or_method": "NoAttention checkpoint",
                    "seed_or_setting": row.get("seed", ""),
                    "asset_path": row.get("checkpoint_path", ""),
                    "asset_type": "checkpoint_only_no_metric",
                    "numeric_metric_available": "no",
                    "metric_fields_found": "",
                    "paper_use_status": "provenance_diagnostic_only",
                    "known_caveat": "Checkpoint inferred width differs from current dataset width.",
                    "recommended_action": "Use only to explain why local reproduction is blocked; ask for original dataset/config snapshot.",
                    "requires_collaborator_confirmation": "yes",
                    "notes": row.get("notes", ""),
                }
            )

    write_csv(out / "03_baseline_numeric_metrics_status.csv", rows, fields)


def build_provenance_map(root: Path, out: Path, tables: dict[str, pd.DataFrame]) -> None:
    fields = [
        "provenance_id",
        "condition",
        "seed",
        "stage",
        "asset_path",
        "asset_type",
        "upstream_asset",
        "downstream_asset",
        "link_confidence",
        "evidence_for_link",
        "paper_grade_status",
        "known_caveat",
        "required_confirmation",
        "notes",
    ]
    rows: list[dict] = []
    pid = 1
    for cond in CONDITIONS:
        for seed in SEEDS:
            checkpoint = f"ModelTrain/NoAttention/SavedModels_6ch/Seed_{seed}/{cond}.pth"
            shap_cache = f"CalculateShapValues/NoAttention/Seed_{seed}/{cond}.npz"
            shap_fig = f"CalculateShapValues/NoAttention/Seed_{seed}_AnalysisGraphs/SHAP_{cond}.png"
            rows.extend(
                [
                    {
                        "provenance_id": f"P{pid:03d}",
                        "condition": cond,
                        "seed": seed,
                        "stage": "checkpoint",
                        "asset_path": rel_if_exists(root, checkpoint),
                        "asset_type": "pth_checkpoint",
                        "upstream_asset": f"wavelet_dataset/{cond}.npz",
                        "downstream_asset": shap_cache,
                        "link_confidence": "medium",
                        "evidence_for_link": "matched condition and seed in local path naming",
                        "paper_grade_status": "needs_collaborator_confirmation",
                        "known_caveat": "No config snapshot tying checkpoint, dataset width, and split contract.",
                        "required_confirmation": "Confirm original training dataset/split/crop width and checkpoint provenance.",
                        "notes": "Do not evaluate in this sprint.",
                    },
                    {
                        "provenance_id": f"P{pid + 1:03d}",
                        "condition": cond,
                        "seed": seed,
                        "stage": "shap_cache",
                        "asset_path": rel_if_exists(root, shap_cache),
                        "asset_type": "ignored_npz_cache",
                        "upstream_asset": checkpoint,
                        "downstream_asset": shap_fig,
                        "link_confidence": "linked_by_path_and_seed_condition_needs_collaborator_confirmation",
                        "evidence_for_link": "SHAP cache path uses NoAttention/Seed_{seed}/{condition}.npz pattern",
                        "paper_grade_status": "candidate_paper_evidence_needs_config_confirmation",
                        "known_caveat": "Large cache not versioned; SHAP computation config not snapshotted.",
                        "required_confirmation": "Confirm cache was generated from the intended NoAttention checkpoint and dataset.",
                        "notes": "Can support attribution story after provenance confirmation.",
                    },
                    {
                        "provenance_id": f"P{pid + 2:03d}",
                        "condition": cond,
                        "seed": seed,
                        "stage": "shap_summary",
                        "asset_path": rel_if_exists(root, shap_fig),
                        "asset_type": "png_figure",
                        "upstream_asset": shap_cache,
                        "downstream_asset": "reports/paper_evidence/shap_seed_stability_summary.csv",
                        "link_confidence": "high_for_local_path_low_for_original_config",
                        "evidence_for_link": "summary tables and figures use condition/seed naming",
                        "paper_grade_status": "provisional_needs_human_confirmation",
                        "known_caveat": "Numeric SHAP summaries are local extraction over existing caches.",
                        "required_confirmation": "Confirm source SHAP caches are final collaborator outputs.",
                        "notes": "",
                    },
                ]
            )
            pid += 3

        for view in VIEWS:
            pc_dataset = f"PC_Datasets/Seed_49/{cond}.csv" if view == "channel" else f"PC_Datasets/Seed_49/{cond}_freq.csv"
            avg_pc_dataset = f"PC_Datasets/average/{cond}.csv" if view == "channel" else f"PC_Datasets/average/{cond}_freq.csv"
            dag = pcdag_figure(root, cond, view, "average")
            rows.extend(
                [
                    {
                        "provenance_id": f"P{pid:03d}",
                        "condition": cond,
                        "seed": "49",
                        "stage": "pc_dataset",
                        "asset_path": rel_if_exists(root, pc_dataset),
                        "asset_type": "csv_pc_input",
                        "upstream_asset": "CalculateShapValues/NoAttention/Seed_49/*.npz",
                        "downstream_asset": dag,
                        "link_confidence": "medium",
                        "evidence_for_link": "PC_Datasets naming and reports tie Seed_49 SHAP to PC-DAG export",
                        "paper_grade_status": "provisional_needs_human_confirmation",
                        "known_caveat": "Exact SHAP weighting and preprocessing config need author confirmation.",
                        "required_confirmation": "Confirm Seed_49 and/or average PC dataset is the intended paper setting.",
                        "notes": f"{view} view",
                    },
                    {
                        "provenance_id": f"P{pid + 1:03d}",
                        "condition": cond,
                        "seed": "average",
                        "stage": "pc_dataset",
                        "asset_path": rel_if_exists(root, avg_pc_dataset),
                        "asset_type": "csv_pc_input",
                        "upstream_asset": "multi-seed SHAP aggregation",
                        "downstream_asset": dag,
                        "link_confidence": "medium",
                        "evidence_for_link": "average PC_Datasets directory and average DAG figures exist locally",
                        "paper_grade_status": "candidate_setting_needs_lock",
                        "known_caveat": "Average setting may be better for figures, but final alpha/filtered policy is not author-confirmed.",
                        "required_confirmation": "Confirm average-vs-Seed_49 choice.",
                        "notes": f"{view} view",
                    },
                    {
                        "provenance_id": f"P{pid + 2:03d}",
                        "condition": cond,
                        "seed": "average",
                        "stage": "pcdag_figure",
                        "asset_path": dag,
                        "asset_type": "png_dag_figure",
                        "upstream_asset": avg_pc_dataset,
                        "downstream_asset": "reports/paper_evidence/pcdag_edges/pcdag_edges_long.csv",
                        "link_confidence": "high_for_local_extraction",
                        "evidence_for_link": "exported edge tables match condition/view graph assets",
                        "paper_grade_status": "provisional_needs_setting_lock",
                        "known_caveat": "PC-KCI alpha and filtered/original policy must be frozen.",
                        "required_confirmation": "Confirm alpha/filtering/physical-prior display policy.",
                        "notes": f"{view} view",
                    },
                ]
            )
            pid += 3

        rows.extend(
            [
                {
                    "provenance_id": f"P{pid:03d}",
                    "condition": cond,
                    "seed": "multi_seed",
                    "stage": "pcdag_edge_table",
                    "asset_path": "reports/paper_evidence/pcdag_edges/pcdag_edges_long.csv",
                    "asset_type": "csv_local_edge_export",
                    "upstream_asset": "DAG/PC_DAG",
                    "downstream_asset": "reports/paper_evidence/consistency/strict_feature_to_label_consistency_table.csv",
                    "link_confidence": "high_for_local_audit",
                    "evidence_for_link": "condition/view edge rows exported from existing DAG assets",
                    "paper_grade_status": "local_audit_table_supports_conservative_interpretation",
                    "known_caveat": "Direction interpretation should remain conservative.",
                    "required_confirmation": "Confirm edge interpretation policy with author.",
                    "notes": "",
                },
                {
                    "provenance_id": f"P{pid + 1:03d}",
                    "condition": cond,
                    "seed": "multi_seed",
                    "stage": "consistency_table",
                    "asset_path": "reports/paper_evidence/consistency/strict_feature_to_label_consistency_table.csv",
                    "asset_type": "csv_local_audit",
                    "upstream_asset": "SHAP summaries + PC-DAG edges",
                    "downstream_asset": "reports/paper_evidence/final_tables",
                    "link_confidence": "high_for_local_audit",
                    "evidence_for_link": "table joins SHAP top-k with strict feature_to_label label edges",
                    "paper_grade_status": "usable_for_claim_after_wording_control",
                    "known_caveat": "Supports audit/consistency, not mechanistic causality.",
                    "required_confirmation": "Confirm this conservative claim layer matches author intent.",
                    "notes": "",
                },
                {
                    "provenance_id": f"P{pid + 2:03d}",
                    "condition": cond,
                    "seed": "multi_seed",
                    "stage": "case_study",
                    "asset_path": "reports/deep_research_handoff/04_main_case_study_asset_pack.csv",
                    "asset_type": "csv_case_index",
                    "upstream_asset": "SHAP figures + PC-DAG figures + consistency tables",
                    "downstream_asset": "reports/stage2_evidence_lock/07_case_study_lock_candidates.csv",
                    "link_confidence": "high_for_local_case_selection",
                    "evidence_for_link": "case IDs are selected from existing SHAP/PC-DAG evidence tables",
                    "paper_grade_status": "provisional_needs_human_confirmation",
                    "known_caveat": "Case selection should be confirmed before final manuscript figures.",
                    "required_confirmation": "Confirm selected cases are scientifically appropriate.",
                    "notes": "",
                },
            ]
        )
        pid += 3

    write_csv(out / "04_checkpoint_shap_pcdag_provenance_map.csv", rows, fields)


def build_source_map(root: Path, out: Path, tables: dict[str, pd.DataFrame]) -> None:
    fields = [
        "paper_item_id",
        "paper_item_type",
        "proposed_title_or_role",
        "source_assets",
        "data_tables",
        "figure_assets",
        "condition_or_scope",
        "view",
        "current_status",
        "paper_use_status",
        "blocking_issue",
        "required_before_stage3",
        "required_before_submission",
        "owner",
        "notes",
    ]
    rows = [
        {
            "paper_item_id": "FIG1",
            "paper_item_type": "main_figure",
            "proposed_title_or_role": "Wavelet-CNN-SHAP-dual-view PC-DAG audit workflow",
            "source_assets": "docs/EXPERIMENTS_LOGIC_AND_SCI_STORYLINE.md; reports/deep_research_handoff/01_author_contribution_reconstruction.md",
            "data_tables": "reports/stage2_evidence_lock/04_checkpoint_shap_pcdag_provenance_map.csv",
            "figure_assets": "to_draw_by_author_or_ChatGPT_storyboard",
            "condition_or_scope": "all",
            "view": "pipeline",
            "current_status": "locked_ready",
            "paper_use_status": "main_text_candidate",
            "blocking_issue": "",
            "required_before_stage3": "none; can storyboard now",
            "required_before_submission": "align wording with 09_claim_wording_control.md",
            "owner": "ChatGPT + human researcher",
            "notes": "No algorithm rerun needed.",
        },
        {
            "paper_item_id": "FIG2",
            "paper_item_type": "main_figure",
            "proposed_title_or_role": "SHAP channel/frequency attribution across SEU conditions",
            "source_assets": "CalculateShapValues/NoAttention/TotalAnalysisGraphs/*.png",
            "data_tables": "reports/paper_evidence/shap_seed_stability_summary.csv; reports/paper_evidence/final_tables/table_shap_stability_main.csv",
            "figure_assets": "CalculateShapValues/NoAttention/TotalAnalysisGraphs/SHAP_All_Conditions_Channel.png; CalculateShapValues/NoAttention/TotalAnalysisGraphs/SHAP_All_Conditions_Frequency.png",
            "condition_or_scope": "four SEU conditions",
            "view": "channel/frequency",
            "current_status": "provisional_needs_human_confirmation",
            "paper_use_status": "main_text_candidate",
            "blocking_issue": "SHAP cache provenance/config snapshot not fully explicit.",
            "required_before_stage3": "can storyboard with provenance caveat",
            "required_before_submission": "confirm SHAP caches are final collaborator outputs",
            "owner": "human researcher",
            "notes": "Do not recompute SHAP in this sprint.",
        },
        {
            "paper_item_id": "FIG3",
            "paper_item_type": "main_figure",
            "proposed_title_or_role": "Dual-view PC-DAG examples",
            "source_assets": "DAG/PC_DAG/Seed_49/filtered/Significance_0.001/average/*.png",
            "data_tables": "reports/paper_evidence/pcdag_edges/pcdag_edge_label_summary.csv; reports/paper_evidence/pcdag_edges/pcdag_edges_long.csv",
            "figure_assets": "selected average filtered channel/frequency DAGs",
            "condition_or_scope": "four SEU conditions",
            "view": "channel/frequency",
            "current_status": "provisional_needs_human_confirmation",
            "paper_use_status": "main_text_candidate",
            "blocking_issue": "Final alpha, filtered/original, average/seed policy needs lock.",
            "required_before_stage3": "use candidate setting only",
            "required_before_submission": "author confirms PC-DAG setting policy",
            "owner": "human researcher",
            "notes": "Recommended wording treats graph as audit structure.",
        },
        {
            "paper_item_id": "FIG4",
            "paper_item_type": "main_figure",
            "proposed_title_or_role": "Attribution-graph consistency/conflict case study",
            "source_assets": "reports/deep_research_handoff/04_main_case_study_asset_pack.csv",
            "data_tables": "reports/paper_evidence/consistency/strict_feature_to_label_consistency_table.csv; reports/paper_evidence/consistency/adjacency_weak_consistency_table.csv",
            "figure_assets": "SHAP condition figures + selected PC-DAG panels",
            "condition_or_scope": "selected cases",
            "view": "channel/frequency",
            "current_status": "provisional_needs_human_confirmation",
            "paper_use_status": "main_text_candidate",
            "blocking_issue": "Case selection should be confirmed; shortcut wording must be conservative.",
            "required_before_stage3": "select 3-5 cases for storyboard",
            "required_before_submission": "author confirms case choices",
            "owner": "ChatGPT + human researcher",
            "notes": "Use strict/weak distinction.",
        },
        {
            "paper_item_id": "TBL1",
            "paper_item_type": "main_table",
            "proposed_title_or_role": "Baseline diagnostic performance status",
            "source_assets": "ModelTrain/NoAttention plots/checkpoints; collaborator files needed",
            "data_tables": "reports/stage2_evidence_lock/03_baseline_numeric_metrics_status.csv",
            "figure_assets": "ModelTrain/NoAttention/ConfusionAndF1/Seed_49/*.png",
            "condition_or_scope": "four SEU conditions",
            "view": "diagnosis",
            "current_status": "blocked_needs_collaborator_confirmation",
            "paper_use_status": "blocked_for_final_results",
            "blocking_issue": "Trusted original numeric baseline metrics missing locally.",
            "required_before_stage3": "can draft placeholder/status table",
            "required_before_submission": "must recover original numeric metrics or rerun a confirmed protocol",
            "owner": "human researcher/collaborator",
            "notes": "Local crop-width-512 metrics are not paper evidence.",
        },
        {
            "paper_item_id": "TBL2",
            "paper_item_type": "main_table",
            "proposed_title_or_role": "SHAP top-k stability",
            "source_assets": "CalculateShapValues/NoAttention",
            "data_tables": "reports/paper_evidence/final_tables/table_shap_stability_main.csv",
            "figure_assets": "CalculateShapValues/NoAttention/TotalAnalysisGraphs/*.png",
            "condition_or_scope": "four SEU conditions",
            "view": "channel/frequency",
            "current_status": "provisional_needs_human_confirmation",
            "paper_use_status": "main_text_candidate",
            "blocking_issue": "Need author confirmation of SHAP cache provenance.",
            "required_before_stage3": "none for storyboard",
            "required_before_submission": "cache provenance confirmation",
            "owner": "human researcher",
            "notes": "",
        },
        {
            "paper_item_id": "TBL3",
            "paper_item_type": "main_table",
            "proposed_title_or_role": "Strict and weak attribution-graph consistency audit",
            "source_assets": "reports/paper_evidence/pcdag_edges; reports/paper_evidence/consistency",
            "data_tables": "strict_feature_to_label_consistency_table.csv; adjacency_weak_consistency_table.csv; consistency_summary.csv",
            "figure_assets": "selected SHAP + PC-DAG panels",
            "condition_or_scope": "four SEU conditions",
            "view": "channel/frequency",
            "current_status": "locked_ready",
            "paper_use_status": "main_text_candidate_with_conservative_wording",
            "blocking_issue": "None for audit framing; setting confirmation still needed for final figure lock.",
            "required_before_stage3": "none",
            "required_before_submission": "confirm PC-DAG setting policy",
            "owner": "Codex/ChatGPT/human researcher",
            "notes": "Only feature_to_label is strict support.",
        },
        {
            "paper_item_id": "SUP1",
            "paper_item_type": "supplement",
            "proposed_title_or_role": "Confusion/F1 plots and training curves",
            "source_assets": "ModelTrain/NoAttention",
            "data_tables": "reports/stage2_evidence_lock/03_baseline_numeric_metrics_status.csv",
            "figure_assets": "ModelTrain/NoAttention/ConfusionAndF1; ModelTrain/NoAttention/SavedGraphs_6ch",
            "condition_or_scope": "SEU",
            "view": "diagnosis",
            "current_status": "supplement_only",
            "paper_use_status": "supplement_after_metric_confirmation",
            "blocking_issue": "plot-only locally",
            "required_before_stage3": "none",
            "required_before_submission": "numeric metric recovery",
            "owner": "human researcher",
            "notes": "",
        },
        {
            "paper_item_id": "SUP2",
            "paper_item_type": "supplement",
            "proposed_title_or_role": "Attention/no-attention comparison",
            "source_assets": "ModelTrain/AttentionAndNoAttention",
            "data_tables": "missing original numeric table",
            "figure_assets": "ModelTrain/AttentionAndNoAttention/SavedGraphs/*.png",
            "condition_or_scope": "SEU",
            "view": "attention support",
            "current_status": "supplement_only",
            "paper_use_status": "not_core",
            "blocking_issue": "not required for frozen mainline",
            "required_before_stage3": "none",
            "required_before_submission": "only if supplement retained",
            "owner": "ChatGPT decision",
            "notes": "Do not convert into main contribution.",
        },
        {
            "paper_item_id": "SUP3",
            "paper_item_type": "supplement_or_future",
            "proposed_title_or_role": "GNN/CWRU/cross-condition branches",
            "source_assets": "GNNCausal; CWRU; ModelTrain/NoAttention/CrossCondTest",
            "data_tables": "reports/reproducibility_audit/gnn_risk_report.md; full_split_overlap_report.csv",
            "figure_assets": "GNN/CWRU curves; Cross_condition_acc_comparison.png",
            "condition_or_scope": "support branches",
            "view": "optional",
            "current_status": "future_work_only",
            "paper_use_status": "do_not_use_as_main_claim",
            "blocking_issue": "GNN implementation risk; CWRU metric gap; cross split leakage.",
            "required_before_stage3": "none",
            "required_before_submission": "only if explicitly retained as supplement/limitation",
            "owner": "ChatGPT + human researcher",
            "notes": "Keep out of core contribution.",
        },
    ]
    write_csv(out / "05_final_table_figure_source_map.csv", rows, fields)


def build_pcdag_setting_candidates(root: Path, out: Path, tables: dict[str, pd.DataFrame]) -> None:
    fields = [
        "candidate_id",
        "condition",
        "view",
        "seed_or_average",
        "alpha_or_setting",
        "filtered_or_original",
        "physical_prior_or_background_knowledge",
        "figure_asset",
        "edge_table_asset",
        "label_edge_summary",
        "strict_edges_count",
        "weak_adjacency_count",
        "why_candidate",
        "risk_or_caveat",
        "recommended_use",
        "needs_human_confirmation",
    ]
    rows: list[dict] = []
    summary = tables["pcdag_summary"]
    strict = tables["strict"]
    weak = tables["weak"]
    cid = 1
    for _, srow in summary.iterrows():
        condition = srow.get("condition", "")
        view = srow.get("view", "")
        seed_or_avg = str(srow.get("seed_or_average", ""))
        strict_count = 0
        weak_count = 0
        if not strict.empty:
            mask = (strict["condition"] == condition) & (strict["view"] == view)
            strict_count = int(strict.loc[mask, "has_strict_feature_to_label_edge"].fillna(False).astype(bool).sum())
        if not weak.empty:
            mask = (weak["condition"] == condition) & (weak["view"] == view)
            weak_count = int(weak.loc[mask, "has_any_label_adjacency"].fillna(False).astype(bool).sum())
        rows.append(
            {
                "candidate_id": f"DAGSET{cid:02d}",
                "condition": condition,
                "view": view,
                "seed_or_average": seed_or_avg,
                "alpha_or_setting": f"alpha={srow.get('alpha', '')}; {srow.get('setting', '')}",
                "filtered_or_original": "filtered_candidate",
                "physical_prior_or_background_knowledge": "existing DAG assets include filtered/original/physical-prior variants; final display policy not locked",
                "figure_asset": pcdag_figure(root, condition, view, seed_or_avg),
                "edge_table_asset": "reports/paper_evidence/pcdag_edges/pcdag_edges_long.csv; reports/paper_evidence/pcdag_edges/pcdag_edge_label_summary.csv",
                "label_edge_summary": f"total_edges={srow.get('n_edges_total', '')}; label_edges={srow.get('n_label_related_edges', '')}; connected={srow.get('features_connected_to_label', '')}; feature_to_label={srow.get('features_directed_to_label', '')}; label_to_feature={srow.get('features_directed_from_label', '')}; undirected={srow.get('features_undirected_with_label', '')}",
                "strict_edges_count": strict_count,
                "weak_adjacency_count": weak_count,
                "why_candidate": "Existing exported edge summary and graph figure can support dual-view audit.",
                "risk_or_caveat": "Only feature_to_label is strict support; label_to_feature/undirected are weak adjacency; final alpha/filtering must be confirmed.",
                "recommended_use": "main_or_supplement_candidate_depending_on_case_selection",
                "needs_human_confirmation": "yes",
            }
        )
        cid += 1
    write_csv(out / "06_pcdag_setting_lock_candidates.csv", rows, fields)


def build_case_candidates(root: Path, out: Path, tables: dict[str, pd.DataFrame]) -> None:
    fields = [
        "case_id",
        "case_type",
        "condition",
        "view",
        "feature_name",
        "shap_evidence",
        "graph_evidence",
        "strict_or_weak",
        "source_tables",
        "candidate_figures",
        "claim_supported",
        "why_good_for_main_text",
        "risk_or_caveat",
        "recommended_status",
        "needs_human_confirmation",
    ]
    case_pack = tables["case_pack"]
    rows: list[dict] = []
    if not case_pack.empty:
        for _, row in case_pack.iterrows():
            condition = row.get("condition", "")
            view = row.get("view", "")
            candidate_figures = join_paths(
                [
                    row.get("shap_figure_candidates", "") or shap_figure(root, condition),
                    row.get("dag_figure_candidates", "") or pcdag_figure(root, condition, view, "average"),
                ]
            )
            rows.append(
                {
                    "case_id": row.get("case_id", ""),
                    "case_type": row.get("case_type", ""),
                    "condition": condition,
                    "view": view,
                    "feature_name": row.get("feature_name", ""),
                    "shap_evidence": row.get("shap_rank_or_importance_summary", ""),
                    "graph_evidence": row.get("graph_support_type", ""),
                    "strict_or_weak": row.get("strict_or_weak", ""),
                    "source_tables": row.get("related_table_rows", "")
                    + "; reports/paper_evidence/consistency/strict_feature_to_label_consistency_table.csv; reports/paper_evidence/consistency/adjacency_weak_consistency_table.csv",
                    "candidate_figures": candidate_figures,
                    "claim_supported": row.get("paper_claim_supported", ""),
                    "why_good_for_main_text": row.get("why_this_case_is_useful", ""),
                    "risk_or_caveat": row.get("risk_or_caveat", ""),
                    "recommended_status": row.get("recommended_use", ""),
                    "needs_human_confirmation": "yes",
                }
            )
    write_csv(out / "07_case_study_lock_candidates.csv", rows, fields)


def build_decision_register(out: Path, tables: dict[str, pd.DataFrame]) -> None:
    fields = [
        "decision_id",
        "evidence_or_claim",
        "current_decision",
        "rationale",
        "supporting_assets",
        "blocking_issue",
        "needed_confirmation",
        "owner",
        "deadline_relevance",
        "notes",
    ]
    decisions = [
        ("D01", "Mainline = SHAP attribution + dual-view PC-DAG audit", "locked", "Matches strongest existing assets and high-level strategy freeze.", "CalculateShapValues; PC_Datasets; DAG/PC_DAG; reports/paper_evidence/consistency", "", "Confirm no late pivot to performance/GNN paper.", "ChatGPT/human researcher", "Stage3 can start", ""),
        ("D02", "Do not make GNN a main contribution", "locked", "GNN branch has implementation-risk audit and is not needed for the frozen mainline.", "GNNCausal; reports/reproducibility_audit/gnn_risk_report.md", "", "None unless human researcher reopens branch.", "ChatGPT", "Prevents scope creep", ""),
        ("D03", "Do not use cross-condition as strong robustness claim", "locked", "Local audit found split overlap/exact duplicate risk.", "reports/reproducibility_audit/full_split_overlap_report.csv", "", "Only whether to mention as limitation/supplement.", "ChatGPT", "Manuscript claim control", ""),
        ("D04", "CWRU branch not core", "locked", "CWRU has curves/checkpoints but no paper-ready trusted metric table locally.", "CWRU/train_curves; CWRU/trained_models", "", "Only if collaborator supplies metrics and ChatGPT wants supplement.", "ChatGPT/human researcher", "Optional after main story", ""),
        ("D05", "Baseline numeric performance table", "blocked", "Trusted collaborator original numeric metrics are missing locally.", "ModelTrain/NoAttention plots/checkpoints", "Need accuracy/F1/report with seed/config/split provenance.", "Collaborator should provide original metrics or confirmed rerun protocol.", "human researcher/collaborator", "Blocks final Results section", "Local crop512 table is diagnostic-only."),
        ("D06", "Local crop-width-512 metrics", "discarded", "Human researcher explicitly stopped this route; results are provenance diagnostic only.", "reports/paper_evidence/baseline_metrics/baseline_metrics_paper_table.csv", "", "None.", "Codex", "Avoids false negative baseline story", "Do not cite as paper evidence."),
        ("D07", "SHAP attribution figures/tables", "provisional", "Strong existing local assets, multi-seed summaries present.", "CalculateShapValues/NoAttention; shap_seed_stability_summary.csv", "Need cache/checkpoint/config provenance confirmation.", "Author confirms caches are final intended outputs.", "human researcher", "Needed before submission", ""),
        ("D08", "Dual-view PC-DAG figures", "provisional", "Existing figures and exported edge tables support audit story.", "DAG/PC_DAG; pcdag_edges_long.csv; pcdag_edge_label_summary.csv", "Need alpha/filter/average/seed policy lock.", "Author confirms final PC-DAG setting.", "human researcher", "Needed before final figures", ""),
        ("D09", "Strict feature_to_label edge interpretation", "locked", "Conservative direction policy avoids overclaiming.", "strict_feature_to_label_consistency_table.csv", "", "Confirm with author but safe as wording baseline.", "Codex/ChatGPT", "Core claim wording", ""),
        ("D10", "label_to_feature and undirected edges", "locked", "Use as weak adjacency only, not strong causal evidence.", "adjacency_weak_consistency_table.csv; pcdag_edge_label_summary.csv", "", "None.", "Codex/ChatGPT", "Core claim wording", ""),
        ("D11", "Shortcut-risk language", "locked", "Evidence supports candidate risk flags, not confirmed shortcuts.", "consistency tables; case study pack", "", "Human researcher confirms case wording.", "ChatGPT/human researcher", "Manuscript safety", ""),
        ("D12", "Attribution-graph consistency table", "locked", "Local table directly joins SHAP and PC-DAG evidence.", "strict_feature_to_label_consistency_table.csv; adjacency_weak_consistency_table.csv; consistency_summary.csv", "", "Confirm table design fits final story.", "ChatGPT", "Stage3 storyboard", ""),
        ("D13", "Case study candidates", "provisional", "10 cases indexed from existing evidence; selected subset needed.", "reports/stage2_evidence_lock/07_case_study_lock_candidates.csv", "Need final choice and figure style.", "ChatGPT selects; human researcher confirms.", "ChatGPT/human researcher", "Stage3 figure plan", ""),
        ("D14", "Attention comparison", "supplement_only", "Existing assets may support discussion but not frozen main claim.", "ModelTrain/AttentionAndNoAttention", "Numeric metrics missing locally.", "Only retain if collaborator metrics are available.", "ChatGPT", "Optional", ""),
        ("D15", "Physical prior/filtered/original PC-DAG ablation", "provisional", "Assets exist, but role is not locked.", "DAG/PC_DAG/GearBox_PC_DAG_physical_prior.png; DAG/PC_DAG/*filtered*", "Need decide whether ablation is necessary or only method note.", "ChatGPT/human researcher decision.", "ChatGPT", "Before final method/result details", ""),
        ("D16", "Stage 3 can start before all confirmations", "locked", "Storyline, storyboard, and claim boundaries are sufficiently clear.", "reports/stage2_evidence_lock", "Final Results table waits for collaborator metrics and PC-DAG setting confirmation.", "Proceed with clear placeholders.", "ChatGPT", "Immediate next step", ""),
    ]
    rows = [
        {
            "decision_id": d[0],
            "evidence_or_claim": d[1],
            "current_decision": d[2],
            "rationale": d[3],
            "supporting_assets": d[4],
            "blocking_issue": d[5],
            "needed_confirmation": d[6],
            "owner": d[7],
            "deadline_relevance": d[8],
            "notes": d[9],
        }
        for d in decisions
    ]
    write_csv(out / "08_evidence_lock_decision_register.csv", rows, fields)


def build_markdown_files(out: Path, tables: dict[str, pd.DataFrame]) -> None:
    strict_counts = counts_by(tables["strict"], "strict_consistency_type")
    weak_counts = counts_by(tables["weak"], "adjacency_support_level")
    label_edge_counts = counts_by(tables["pcdag_edges"], "label_relation")

    write_text(
        out / "README.md",
        f"""
# Stage 2 Evidence Lock Package

This folder freezes local evidence sources before ChatGPT Stage 3 deep research and storyboard design. It is an evidence-control package, not manuscript prose.

## Reading Order

1. `01_stage2_scope_and_principles.md`
2. `08_evidence_lock_decision_register.csv`
3. `03_baseline_numeric_metrics_status.csv`
4. `04_checkpoint_shap_pcdag_provenance_map.csv`
5. `05_final_table_figure_source_map.csv`
6. `06_pcdag_setting_lock_candidates.csv`
7. `07_case_study_lock_candidates.csv`
8. `09_claim_wording_control.md`
9. `10_submission_readiness_after_stage2.md`
10. `02_author_result_request_packet.md`
11. `11_chatgpt_stage3_prompt.md`

## Current Evidence Snapshot

- Strict consistency counts: {strict_counts}
- Weak adjacency counts: {weak_counts}
- PC-DAG label-relation counts: {label_edge_counts}

## Non-Goals

This package did not train, evaluate checkpoints, recompute SHAP, rerun PC-KCI/GNN, build new datasets, or revise algorithms. The local `--crop-width 512` result is diagnostic-only and is not paper evidence.
""",
    )

    write_text(
        out / "01_stage2_scope_and_principles.md",
        """
# Stage 2 Scope And Principles

## Frozen Mainline

The mainline is: SHAP attribution plus dual-view PC-DAG attribution-graph consistency audit for trustworthy rotating-machinery diagnosis.

The paper should not be framed as a pure performance paper, a GNN paper, a cross-condition robustness paper, or a CWRU external-validation paper.

## Evidence Principles

- Use existing collaborator assets as the primary evidence base.
- Separate original result assets from later Codex audit/extraction tables.
- Treat SHAP/PC-DAG as a diagnostic-logic audit pipeline.
- Treat `feature_to_label` edges as strict graph support.
- Treat `label_to_feature` and undirected edges only as weak adjacency/context.
- Treat shortcut findings as candidate shortcut-risk or attribution-graph conflict, not confirmed shortcuts.
- Treat local `--crop-width 512` metrics as provenance diagnostics only.

## No-Go Actions In This Sprint

- No training.
- No checkpoint evaluation.
- No SHAP recomputation.
- No PC-KCI or GNN rerun.
- No new dataset construction.
- No CLI/config refactor.
- No manuscript prose generation.

## Status Vocabulary

- `locked_ready`: enough for Stage 3 storyboard under conservative wording.
- `provisional_needs_human_confirmation`: usable for planning, but author confirmation is required before submission.
- `blocked_needs_collaborator_confirmation`: cannot be used as final paper evidence until missing collaborator result files are recovered.
- `supplement_only`: do not use as a main claim.
- `limitation_only`: mention only as a caveat/risk.
- `future_work_only`: not part of current paper evidence.
- `do_not_use`: excluded from paper evidence.

## What Locked Means

Locked means the local evidence role and claim boundary are frozen for Stage 3 planning. It does not mean every numeric value is submission-ready. Baseline performance metrics and final PC-DAG display settings still require collaborator/human confirmation.
""",
    )

    write_text(
        out / "02_author_result_request_packet.md",
        """
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
""",
    )

    write_text(
        out / "09_claim_wording_control.md",
        """
# Claim Wording Control

## Can Write Strongly

- The project implements an attribution-graph audit workflow combining SHAP attribution and dual-view PC-DAG analysis.
- SHAP provides channel-level and wavelet-frequency-level attribution patterns for the trained diagnostic model.
- Existing PC-DAG assets provide channel-view and frequency-view graph structures for auditing diagnostic logic.
- The strict audit table uses only `feature_to_label` edges as directional graph support.
- The weak audit table separately records label adjacency from `label_to_feature` or undirected edges.
- The method can identify attribution-graph consistency, conflict, and candidate shortcut-risk patterns.

## Must Weaken

- Use "graph support", "diagnostic-logic audit", "attribution-graph consistency", or "candidate shortcut-risk" instead of confirmed causal mechanism.
- Use "PC-DAG-derived dependency structure" instead of ground-truth causal graph.
- Use "may indicate", "suggests", "is consistent with", and "flags for review" for shortcut/conflict cases.
- Use "cross-condition result is exploratory / limited by split overlap risk" if cross results are mentioned.
- Use "CWRU/GNN branch is exploratory or future work" unless confirmed metrics and implementation fixes are supplied.

## Cannot Write

- Do not claim the model discovers true physical causality.
- Do not claim `label_to_feature` edges prove causal influence from features to label.
- Do not claim undirected PC-DAG links are directional causal evidence.
- Do not claim shortcut features are confirmed shortcuts without intervention or leak-free robustness tests.
- Do not claim cross-condition robustness from the current cross split.
- Do not cite local `--crop-width 512` metrics as paper performance evidence.
- Do not frame GNN as a validated main contribution in the current paper.
- Do not present CWRU as a completed external validation unless collaborator numeric metrics are recovered.

## Recommended Main Claim Template

We propose a diagnostic-logic audit framework that couples SHAP attribution with dual-view PC-DAG structures to examine whether model-relevant channels and wavelet bands are consistent with label-related graph dependencies. The framework distinguishes strict feature-to-label graph support from weaker label adjacency, enabling conservative identification of attribution-graph agreement, conflict, and candidate shortcut-risk patterns.
""",
    )

    write_text(
        out / "10_submission_readiness_after_stage2.md",
        """
# Submission Readiness After Stage 2

## Locked

- Mainline: SHAP attribution + dual-view PC-DAG diagnostic-logic audit.
- GNN, cross-condition, CWRU, and attention are not core contributions.
- Strict causal wording policy: only `feature_to_label` is strict support.
- Local `--crop-width 512` artifact is diagnostic-only and excluded from paper evidence.
- Stage 3 can begin with storyboard/contribution/figure design.

## Needs Collaborator Confirmation

- Original NoAttention baseline numeric metrics.
- Original split/dataset/crop-width contract for baseline results.
- SHAP cache provenance and generation config.
- PC-DAG setting lock: seed vs average, alpha, filtered/original, physical prior role.
- Final case-study selection.

## What Stage 3 Can Start Now

- Contribution framing.
- Main figure storyboard.
- Claim hierarchy and conservative wording.
- Table skeletons for SHAP, PC-DAG, and consistency audit.
- Author request and follow-up experiment plan.

## What Must Wait

- Final Results performance table.
- Final diagnostic baseline comparison.
- Final PC-DAG figure settings.
- Any claim involving external validation, cross-condition robustness, or GNN improvement.

## P0 Blocking List

1. Trusted collaborator original baseline metrics are missing locally.
2. PC-DAG setting policy is not author-confirmed.
3. SHAP cache/checkpoint/config link needs confirmation.
4. Dataset/split/crop-width provenance must be clarified before quantitative performance claims.

## Two-Week Action Proposal

- Days 1-2: Send author result request packet and collect small confirmed files.
- Days 3-5: ChatGPT drafts Stage 3 storyboard and claim map using locked evidence boundaries.
- Days 6-8: Human researcher confirms baseline metric source and PC-DAG setting.
- Days 9-11: Codex updates source maps with confirmed files and prepares final tables.
- Days 12-14: ChatGPT finalizes figure/table plan and missing-experiment decision.

## Honest Conclusion

Stage 3 can start for storyline, contribution framing, and storyboard design. The final Results section cannot be frozen until collaborator baseline numeric metrics and PC-DAG setting confirmation are returned.
""",
    )

    write_text(
        out / "11_chatgpt_stage3_prompt.md",
        """
# Prompt For ChatGPT Stage 3

You are ChatGPT leading the scientific design of the gearbox project. Use the Stage 2 evidence-lock package as the local evidence boundary.

## Frozen Direction

Frame the paper around SHAP attribution plus dual-view PC-DAG attribution-graph consistency audit for trustworthy rotating-machinery diagnosis.

Do not frame the paper as:

- a pure model-performance paper;
- a GNN paper;
- a cross-condition robustness paper;
- a CWRU external-validation paper.

## Locked Evidence

- SHAP attribution assets and multi-seed summaries exist.
- Dual-view PC-DAG assets and exported edge summaries exist.
- Strict/weak consistency audit tables exist.
- Case-study candidates exist.
- Conservative claim wording is defined in `09_claim_wording_control.md`.

## Blocked Evidence

- Trusted original baseline numeric metrics are missing.
- PC-DAG final setting needs confirmation.
- SHAP cache/config provenance needs confirmation.
- Local `--crop-width 512` metrics are diagnostic-only and must not be used as paper evidence.

## Required Stage 3 Outputs

1. A contribution map with 2-3 defensible contributions.
2. A main-figure storyboard, including pipeline, SHAP, dual-view PC-DAG, and case-study panels.
3. A final table plan separating locked, provisional, and blocked tables.
4. A claim hierarchy using conservative causal language.
5. A minimal follow-up/confirmation plan for the collaborator.
6. A journal-positioning assessment that avoids overclaiming causality or robustness.
7. A supplement/future-work policy for attention, GNN, CWRU, and cross-condition.

## Wording Constraints

- Use "diagnostic-logic audit" and "attribution-graph consistency".
- Only `feature_to_label` edges can be called strict graph support.
- Treat `label_to_feature` and undirected edges as weak adjacency.
- Treat shortcut findings as candidate shortcut-risk, not confirmed shortcut mechanisms.
- Do not use local crop-width-512 metrics.
""",
    )


def build_all(root: Path, out: Path) -> None:
    tables = load_tables(root)
    out.mkdir(parents=True, exist_ok=True)
    build_baseline_metrics_status(root, out, tables)
    build_provenance_map(root, out, tables)
    build_source_map(root, out, tables)
    build_pcdag_setting_candidates(root, out, tables)
    build_case_candidates(root, out, tables)
    build_decision_register(out, tables)
    build_markdown_files(out, tables)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."), help="Project root")
    parser.add_argument("--out", type=Path, default=Path("reports/stage2_evidence_lock"), help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out = args.out if args.out.is_absolute() else root / args.out
    build_all(root, out)
    print(f"Wrote Stage 2 evidence-lock package to {out}")


if __name__ == "__main__":
    main()
