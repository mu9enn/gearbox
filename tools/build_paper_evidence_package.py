#!/usr/bin/env python3
"""Consolidate extracted evidence tables into a paper-facing status package."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="reports/paper_evidence")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out = (root / args.out).resolve()

    consistency = read_csv(out / "consistency" / "attribution_causality_consistency_table.csv")
    consistency_counts = Counter(row.get("consistency_type", "") for row in consistency)
    strict_consistency = read_csv(out / "consistency" / "strict_feature_to_label_consistency_table.csv")
    strict_counts = Counter(row.get("strict_consistency_type", "") for row in strict_consistency)
    adjacency_consistency = read_csv(out / "consistency" / "adjacency_weak_consistency_table.csv")
    adjacency_counts = Counter(row.get("adjacency_support_level", "") for row in adjacency_consistency)
    edge_summary = read_csv(out / "pcdag_edges" / "pcdag_edge_label_summary.csv")
    baseline_metrics = read_csv(out / "baseline_metrics" / "baseline_metrics_long.csv")
    baseline_status = Counter(row.get("status", "") for row in baseline_metrics)
    baseline_ok = baseline_status.get("ok", 0)
    baseline_crop_widths = sorted({row.get("crop_width", "") for row in baseline_metrics if row.get("status") == "ok"})
    baseline_human_confirmed = any(
        row.get("human_researcher_confirmed_crop_assumption", "").lower() == "true" for row in baseline_metrics
    )
    mismatch = read_csv(out / "baseline_metrics" / "checkpoint_dataset_mismatch_report.csv")
    shap_stability = read_csv(out / "shap_seed_stability_summary.csv")

    plan_rows = [
        {
            "paper_item": "Table 1",
            "type": "main_table",
            "source_assets": "ModelTrain/NoAttention/SavedGraphs_6ch; ModelTrain/NoAttention/ConfusionAndF1; collaborator numeric metrics if recovered",
            "what_it_shows": "CNN/no-attention diagnostic baseline from collaborator results",
            "paper_claim_supported": "Wavelet-CNN baseline can perform fault diagnosis",
            "current_status": "pending_trusted_numeric_metrics",
            "missing_piece": "Need collaborator-provided numeric metrics table or later formal rerun; local crop-width eval is diagnostic only",
            "recommended_next_action": "Recover collaborator metrics or plan formal rerun; do not use baseline_metrics_paper_table.csv as paper result",
        },
        {
            "paper_item": "Table 2",
            "type": "main_table",
            "source_assets": "reports/paper_evidence/shap_seed_stability_summary.csv",
            "what_it_shows": "SHAP channel/frequency rank stability across five seeds",
            "paper_claim_supported": "Model attention to channel/frequency features is quantifiable and seed-stable",
            "current_status": "available",
            "missing_piece": "Final formatting only",
            "recommended_next_action": "Select top features per condition/view for manuscript table",
        },
        {
            "paper_item": "Table 3",
            "type": "main_table",
            "source_assets": "reports/paper_evidence/consistency/strict_feature_to_label_consistency_table.csv; reports/paper_evidence/consistency/adjacency_weak_consistency_table.csv",
            "what_it_shows": "Strict feature-to-label support plus weak label-adjacency audit",
            "paper_claim_supported": "Conservative attribution-causality audit identifies strict support, unsupported high-SHAP features, and weak graph adjacency",
            "current_status": "available_conservative",
            "missing_piece": "Strict support is sparse; weak adjacency should not be described as causal direction",
            "recommended_next_action": "Use strict table for strong claims and adjacency table for weak graph-consistency claims",
        },
        {
            "paper_item": "Figure 1",
            "type": "main_figure",
            "source_assets": "code pipeline; PROJECT_AUDIT.md",
            "what_it_shows": "Framework overview from vibration to SHAP-PC-DAG audit",
            "paper_claim_supported": "Proposed diagnostic logic audit workflow",
            "current_status": "needs_drawing",
            "missing_piece": "No final schematic asset",
            "recommended_next_action": "Create clean framework diagram after table claims freeze",
        },
        {
            "paper_item": "Figure 2",
            "type": "main_figure",
            "source_assets": "CalculateShapValues/NoAttention/TotalAnalysisGraphs/*.png; shap_rank_summary.csv",
            "what_it_shows": "Channel/frequency SHAP attribution examples",
            "paper_claim_supported": "Model relies unevenly on channels and wavelet bands",
            "current_status": "available",
            "missing_piece": "Choose representative condition and unify style",
            "recommended_next_action": "Select 1-2 condition examples plus all-condition summary",
        },
        {
            "paper_item": "Figure 3",
            "type": "main_figure",
            "source_assets": "DAG/PC_DAG/Seed_49/**; reports/paper_evidence/pcdag_edges",
            "what_it_shows": "Channel-view and frequency-view PC-DAG examples",
            "paper_claim_supported": "Dual-view causal graph reveals label-related feature structure",
            "current_status": "available",
            "missing_piece": "Pick final seed/average/alpha setting",
            "recommended_next_action": "Use edge summary to select representative DAG figures",
        },
        {
            "paper_item": "Figure 4",
            "type": "main_figure",
            "source_assets": "consistency table; selected DAG figures; SHAP rank table",
            "what_it_shows": "Attribution-causality conflict case study",
            "paper_claim_supported": "High attribution without strict feature-to-label support suggests shortcut risk",
            "current_status": "candidate_available",
            "missing_piece": "Human selection of case study feature",
            "recommended_next_action": "Use high_shap_no_label_edge rows as candidates",
        },
        {
            "paper_item": "Supplement",
            "type": "supplement_table_figure",
            "source_assets": "ModelTrain/NoAttention/ConfusionAndF1; ModelTrain/AttentionAndNoAttention",
            "what_it_shows": "Confusion/F1 and attention comparison",
            "paper_claim_supported": "Diagnostic behavior details and auxiliary comparison",
            "current_status": "figure_only",
            "missing_piece": "classification_report CSV for confusion/F1",
            "recommended_next_action": "Export numeric reports without retraining if compatible data is located",
        },
        {
            "paper_item": "Limitation",
            "type": "limitation_note",
            "source_assets": "reports/reproducibility_audit/full_split_overlap_report.csv; CrossCondTest figure",
            "what_it_shows": "Old cross-condition split leakage",
            "paper_claim_supported": "Cross-condition results are not used as strong evidence",
            "current_status": "available",
            "missing_piece": "None for limitation wording",
            "recommended_next_action": "Keep cross-condition out of main claims",
        },
        {
            "paper_item": "Future",
            "type": "future_work_note",
            "source_assets": "GNNCausal; reports/reproducibility_audit/gnn_risk_report.md",
            "what_it_shows": "Causal GNN enhancement remains risky",
            "paper_claim_supported": "GNN is not part of current contribution",
            "current_status": "risk_documented",
            "missing_piece": "GNN classifier/scaler fix and rerun",
            "recommended_next_action": "Leave for later sprint",
        },
    ]
    write_csv(
        out / "paper_core_tables_figures_plan.csv",
        [
            "paper_item",
            "type",
            "source_assets",
            "what_it_shows",
            "paper_claim_supported",
            "current_status",
            "missing_piece",
            "recommended_next_action",
        ],
        plan_rows,
    )

    manifest_rows = [
        {
            "asset_path": "reports/paper_evidence/shap_seed_stability_summary.csv",
            "asset_type": "csv",
            "experiment_family": "shap_ranking",
            "paper_section_candidate": "Explanation",
            "evidence_strength": "A",
            "can_use_in_main": "True",
            "can_use_in_supplement": "True",
            "needs_reproduction": "False",
            "known_risk": "large SHAP source cache lacks config manifest",
            "recommended_action": "Use as Table 2 candidate",
            "notes": f"{len(shap_stability)} rows extracted across seeds",
        },
        {
            "asset_path": "reports/paper_evidence/pcdag_edges/pcdag_edges_long.csv",
            "asset_type": "csv",
            "experiment_family": "pcdag_edge_export",
            "paper_section_candidate": "Causal Audit",
            "evidence_strength": "B+",
            "can_use_in_main": "True",
            "can_use_in_supplement": "True",
            "needs_reproduction": "False",
            "known_risk": "label_to_feature/undirected_with_label are not strong causal-direction evidence",
            "recommended_action": "Use feature_to_label as strong evidence; use other label adjacencies only as weak graph evidence",
            "notes": f"{sum(int(row['n_edges_total']) for row in edge_summary) if edge_summary else 0} exported edges",
        },
        {
            "asset_path": "reports/paper_evidence/consistency/strict_feature_to_label_consistency_table.csv",
            "asset_type": "csv",
            "experiment_family": "attribution_causality_consistency",
            "paper_section_candidate": "Causal Audit",
            "evidence_strength": "A- conservative",
            "can_use_in_main": "True",
            "can_use_in_supplement": "True",
            "needs_reproduction": "False",
            "known_risk": "Strict feature_to_label evidence is sparse",
            "recommended_action": "Use as conservative Table 3 strong-evidence layer",
            "notes": f"strict_counts={dict(strict_counts)}; adjacency_counts={dict(adjacency_counts)}; legacy_counts={dict(consistency_counts)}",
        },
        {
            "asset_path": "reports/paper_evidence/baseline_metrics/baseline_metrics_long.csv",
            "asset_type": "csv",
            "experiment_family": "local_crop_width_diagnostic",
            "paper_section_candidate": "Reproducibility Gap",
            "evidence_strength": "D diagnostic only",
            "can_use_in_main": "False",
            "can_use_in_supplement": "False",
            "needs_reproduction": "False" if baseline_ok and baseline_human_confirmed else "True",
            "known_risk": "local crop-width-512 eval is near random and reflects current code/data/checkpoint provenance gap; not collaborator historical result",
            "recommended_action": "Do not use as paper evidence; recover collaborator metrics or formally rerun later",
            "notes": f"baseline_status={dict(baseline_status)}; diagnostic_only=True; mismatch_rows={len(mismatch)}",
        },
        {
            "asset_path": "DAG/PC_DAG/Seed_49",
            "asset_type": "figures",
            "experiment_family": "dual_view_pcdag_figures",
            "paper_section_candidate": "Causal Audit",
            "evidence_strength": "A-",
            "can_use_in_main": "True",
            "can_use_in_supplement": "True",
            "needs_reproduction": "False",
            "known_risk": "many settings; final alpha/seed choice must be frozen",
            "recommended_action": "Select representative figures using edge summaries",
            "notes": "DAG figure assets already exist",
        },
        {
            "asset_path": "ModelTrain/NoAttention/CrossCondTest",
            "asset_type": "figure",
            "experiment_family": "cross_condition",
            "paper_section_candidate": "Limitation",
            "evidence_strength": "C",
            "can_use_in_main": "False",
            "can_use_in_supplement": "True",
            "needs_reproduction": "True",
            "known_risk": "train/cross_test exact duplicate in current wavelet_dataset",
            "recommended_action": "Use only as leakage limitation/risk note",
            "notes": "Not strong robustness evidence",
        },
    ]
    write_csv(
        out / "paper_asset_manifest.csv",
        [
            "asset_path",
            "asset_type",
            "experiment_family",
            "paper_section_candidate",
            "evidence_strength",
            "can_use_in_main",
            "can_use_in_supplement",
            "needs_reproduction",
            "known_risk",
            "recommended_action",
            "notes",
        ],
        manifest_rows,
    )

    status_lines = [
        "# Paper Evidence Status",
        "",
        "## Main Evidence Available",
        "",
        f"- SHAP seed-stability ranking is available: {len(shap_stability)} rows.",
        f"- PC-DAG edge export is available: {sum(int(row['n_edges_total']) for row in edge_summary) if edge_summary else 0} edges, {sum(int(row['n_label_related_edges']) for row in edge_summary) if edge_summary else 0} label-related edges.",
        f"- Strict feature-to-label consistency table is available: {len(strict_consistency)} rows with counts {dict(strict_counts)}.",
        f"- Weak adjacency consistency table is available: {len(adjacency_consistency)} rows with support counts {dict(adjacency_counts)}.",
        f"- Local crop-width baseline diagnostic rows are available: {baseline_ok} ok rows, crop widths {baseline_crop_widths}; these are diagnostic-only and not paper-candidate metrics.",
        "",
        "## Main Evidence Still Missing",
        "",
        "- Trusted collaborator baseline numeric metrics are still missing or not yet indexed; local crop-width metrics must not be used as paper evidence.",
        "- `label_to_feature` and `undirected_with_label` edges are excluded from strong causal-direction evidence.",
        "",
        "## Supplement Candidates",
        "",
        "- Existing confusion/F1 PNGs and attention/no-attention curves can support supplement after numeric export.",
        "- Existing CWRU/GNN assets remain optional and should not enter the main contribution.",
        "",
        "## Risk / Do Not Use As Strong Evidence",
        "",
        "- Old cross-condition results remain risk/reference only due to train/cross_test exact duplicates.",
        "- Local `--crop-width 512` baseline metrics remain a reproducibility/provenance diagnostic only.",
        "- GNN causal enhancement remains future work because of the documented `classifier(x)` and scaler risks.",
        "",
        "## Next Minimal Engineering Task",
        "",
        "- Recover collaborator baseline numeric metrics or plan a later formal rerun under a frozen protocol.",
        "- Freeze whether manuscript uses strict causal-direction language or weaker causal-adjacency / graph-consistency language.",
    ]
    (out / "paper_evidence_status.md").write_text("\n".join(status_lines) + "\n", encoding="utf-8")
    print(f"Wrote paper evidence package to {out.relative_to(root)}")
    print(f"consistency_counts={dict(consistency_counts)}")
    print(f"strict_counts={dict(strict_counts)}")
    print(f"adjacency_counts={dict(adjacency_counts)}")
    print(f"baseline_status={dict(baseline_status)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
