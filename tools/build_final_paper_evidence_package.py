#!/usr/bin/env python3
"""Build final paper-facing evidence tables and figure/case selections."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


NOTE_512 = (
    "old checkpoint/SHAP used 512-width input; current wavelet data are 1024-width; "
    "human researcher confirmed using first 512 points via [..., :512] to recover historical evaluation protocol"
)


def rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        fieldnames = seen
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def boolish(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def pick_existing(root: Path, candidates: list[str]) -> str:
    for candidate in candidates:
        if (root / candidate).exists():
            return candidate
    return candidates[0] if candidates else "to_select"


def dag_path(condition: str, view: str, root: Path, filtered: bool = True) -> str:
    suffix = "ch" if view == "channel" else "freq"
    base = "filtered" if filtered else "original"
    candidates = [
        f"DAG/PC_DAG/Seed_49/{base}/Significance_0.001/average/{condition}_{suffix}.png",
        f"DAG/PC_DAG/Seed_49/{base}/Significance_0.001/{condition}_{suffix}.png",
        f"DAG/PC_DAG/Seed_49/{base}/Significance_0.05/average/{condition}_{suffix}.png",
        f"DAG/PC_DAG/Seed_49/{base}/Significance_0.05/{condition}_{suffix}.png",
    ]
    return pick_existing(root, candidates)


def shap_path(condition: str, root: Path) -> str:
    return pick_existing(
        root,
        [
            f"CalculateShapValues/NoAttention/TotalAnalysisGraphs/SHAP_{condition}.png",
            "CalculateShapValues/NoAttention/TotalAnalysisGraphs/SHAP_All_Conditions_Channel.png",
            "CalculateShapValues/NoAttention/TotalAnalysisGraphs/SHAP_All_Conditions_Frequency.png",
        ],
    )


def build_baseline_summary(root: Path, evidence: Path) -> list[dict]:
    paper = read_df(evidence / "baseline_metrics" / "baseline_metrics_paper_table.csv")
    if paper.empty:
        return []
    rows = []
    for _, row in paper.iterrows():
        rows.append(
            {
                "condition": row["condition"],
                "accuracy": f"{float(row['accuracy_mean']):.4f} +/- {float(row['accuracy_std']):.4f}",
                "macro_f1": f"{float(row['macro_f1_mean']):.4f} +/- {float(row['macro_f1_std']):.4f}",
                "weighted_f1": f"{float(row['weighted_f1_mean']):.4f} +/- {float(row['weighted_f1_std']):.4f}",
                "n_seeds": int(row["n_seeds"]),
                "n_samples": int(row["n_samples"]),
                "paper_use_status": row["paper_use_status"],
                "caveat": NOTE_512,
            }
        )
    return rows


def build_final_tables(root: Path, evidence: Path) -> dict[str, pd.DataFrame]:
    final_tables = evidence / "final_tables"
    final_tables.mkdir(parents=True, exist_ok=True)

    shap = read_df(evidence / "shap_seed_stability_summary.csv")
    strict = read_df(evidence / "consistency" / "strict_feature_to_label_consistency_table.csv")
    adjacency = read_df(evidence / "consistency" / "adjacency_weak_consistency_table.csv")

    if not shap.empty:
        shap_top = (
            shap.sort_values(["condition", "view", "mean_rank", "mean_importance_across_seeds"], ascending=[True, True, True, False])
            .groupby(["condition", "view"], as_index=False, group_keys=False)
            .head(3)
            .copy()
        )
        shap_top["paper_interpretation"] = shap_top.apply(
            lambda r: f"top SHAP {r['view']} feature candidate; use as attribution ranking evidence", axis=1
        )
        shap_top = shap_top[
            [
                "condition",
                "view",
                "feature_name",
                "mean_importance_across_seeds",
                "std_importance_across_seeds",
                "mean_rank",
                "std_rank",
                "topk_frequency",
                "n_seeds",
                "paper_interpretation",
            ]
        ]
        shap_top.to_csv(final_tables / "table_shap_stability_main.csv", index=False)

    if not strict.empty:
        strict_main = strict[strict["has_strict_feature_to_label_edge"].map(boolish)].copy()
        strict_main["paper_interpretation"] = strict_main["is_topk_shap"].map(boolish).map(
            {
                True: "high-attribution feature with strict feature-to-label PC-DAG support",
                False: "low-attribution feature with strict feature-to-label PC-DAG support; underused graph-supported candidate",
            }
        )
        strict_main = strict_main[
            [
                "condition",
                "view",
                "feature_name",
                "is_topk_shap",
                "has_strict_feature_to_label_edge",
                "strict_consistency_type",
                "strict_evidence_strength",
                "paper_interpretation",
            ]
        ]
        strict_main.to_csv(final_tables / "table_strict_consistency_main.csv", index=False)

    if not adjacency.empty:
        adjacency_main = adjacency.copy()
        adjacency_main["paper_interpretation"] = adjacency_main.apply(
            lambda r: (
                "high-attribution feature with label-related graph adjacency"
                if boolish(r["is_topk_shap"]) and boolish(r["has_any_label_adjacency"])
                else "high-attribution feature without label adjacency; shortcut-risk candidate"
                if boolish(r["is_topk_shap"])
                else "low-attribution feature with label adjacency; underused graph-supported candidate"
                if boolish(r["has_any_label_adjacency"])
                else "low-attribution feature without label adjacency"
            ),
            axis=1,
        )
        adjacency_main = adjacency_main[
            [
                "condition",
                "view",
                "feature_name",
                "is_topk_shap",
                "has_any_label_adjacency",
                "has_strict_feature_to_label_edge",
                "adjacency_support_level",
                "adjacency_consistency_type",
                "paper_interpretation",
            ]
        ]
        adjacency_main.to_csv(final_tables / "table_weak_adjacency_audit_main.csv", index=False)

    shortcut_rows = []
    if not strict.empty:
        for _, row in strict[strict["is_topk_shap"].map(boolish) & ~strict["has_strict_feature_to_label_edge"].map(boolish)].iterrows():
            shortcut_rows.append(
                {
                    "candidate_type": "strict_shortcut_risk_candidate",
                    "condition": row["condition"],
                    "view": row["view"],
                    "feature_name": row["feature_name"],
                    "shap_status": "high_shap",
                    "graph_support_status": "no_strict_feature_to_label_support",
                    "mean_rank": row.get("mean_rank", ""),
                    "mean_importance_across_seeds": row.get("mean_importance_across_seeds", ""),
                    "paper_interpretation": "shortcut-risk candidate; do not claim confirmed shortcut without intervention",
                }
            )
    if not adjacency.empty:
        no_adj = adjacency[adjacency["is_topk_shap"].map(boolish) & ~adjacency["has_any_label_adjacency"].map(boolish)]
        for _, row in no_adj.iterrows():
            shortcut_rows.append(
                {
                    "candidate_type": "weak_shortcut_risk_candidate",
                    "condition": row["condition"],
                    "view": row["view"],
                    "feature_name": row["feature_name"],
                    "shap_status": "high_shap",
                    "graph_support_status": "no_label_adjacency",
                    "mean_rank": "",
                    "mean_importance_across_seeds": "",
                    "paper_interpretation": "stronger shortcut-risk candidate because both strict direction and weak label adjacency are absent",
                }
            )
    write_csv(
        final_tables / "table_shortcut_risk_candidates.csv",
        shortcut_rows,
        [
            "candidate_type",
            "condition",
            "view",
            "feature_name",
            "shap_status",
            "graph_support_status",
            "mean_rank",
            "mean_importance_across_seeds",
            "paper_interpretation",
        ],
    )

    underused_rows = []
    if not strict.empty:
        low_strict = strict[~strict["is_topk_shap"].map(boolish) & strict["has_strict_feature_to_label_edge"].map(boolish)]
        for _, row in low_strict.iterrows():
            underused_rows.append(
                {
                    "candidate_type": "strict_underused_graph_supported_candidate",
                    "condition": row["condition"],
                    "view": row["view"],
                    "feature_name": row["feature_name"],
                    "shap_status": "low_shap",
                    "graph_support_status": "strict_feature_to_label_support",
                    "paper_interpretation": "underused graph-supported feature candidate",
                }
            )
    if not adjacency.empty:
        low_adj = adjacency[~adjacency["is_topk_shap"].map(boolish) & adjacency["has_any_label_adjacency"].map(boolish)]
        for _, row in low_adj.iterrows():
            underused_rows.append(
                {
                    "candidate_type": "weak_underused_graph_supported_candidate",
                    "condition": row["condition"],
                    "view": row["view"],
                    "feature_name": row["feature_name"],
                    "shap_status": "low_shap",
                    "graph_support_status": row["adjacency_support_level"],
                    "paper_interpretation": "underused graph-supported feature candidate under weak adjacency policy",
                }
            )
    write_csv(
        final_tables / "table_underused_causal_candidates.csv",
        underused_rows,
        [
            "candidate_type",
            "condition",
            "view",
            "feature_name",
            "shap_status",
            "graph_support_status",
            "paper_interpretation",
        ],
    )

    return {"shap": shap, "strict": strict, "adjacency": adjacency}


def build_figures_and_cases(root: Path, evidence: Path) -> tuple[list[dict], list[dict]]:
    final_figures = evidence / "final_figures"
    final_figures.mkdir(parents=True, exist_ok=True)
    strict = read_df(evidence / "final_tables" / "table_strict_consistency_main.csv")
    weak = read_df(evidence / "final_tables" / "table_weak_adjacency_audit_main.csv")
    shortcuts = read_df(evidence / "final_tables" / "table_shortcut_risk_candidates.csv")
    underused = read_df(evidence / "final_tables" / "table_underused_causal_candidates.csv")

    strict_case = strict[strict["is_topk_shap"].map(boolish)].head(1) if not strict.empty else pd.DataFrame()
    weak_case = weak[
        weak["is_topk_shap"].map(boolish)
        & weak["has_any_label_adjacency"].map(boolish)
        & ~weak["has_strict_feature_to_label_edge"].map(boolish)
    ].head(1) if not weak.empty else pd.DataFrame()
    shortcut_case = shortcuts[shortcuts["candidate_type"].eq("weak_shortcut_risk_candidate")].head(1) if not shortcuts.empty else pd.DataFrame()
    underused_case = underused[underused["candidate_type"].str.startswith("strict", na=False)].head(1) if not underused.empty else pd.DataFrame()

    case_rows = []
    for case_id, case_type, df, claim in [
        ("case_1", "strict_consistent_case", strict_case, "High SHAP feature has strict feature-to-label PC-DAG support."),
        ("case_2", "weak_adjacency_consistent_case", weak_case, "High SHAP feature has label-related graph adjacency but no directional claim."),
        ("case_3", "shortcut_risk_candidate", shortcut_case, "High SHAP feature lacks label adjacency and is a shortcut-risk candidate."),
        ("case_4", "underused_graph_supported_candidate", underused_case, "Low SHAP feature has graph support and may be underused by the model."),
    ]:
        if df.empty:
            case_rows.append(
                {
                    "case_id": case_id,
                    "case_type": case_type,
                    "condition": "not_available",
                    "view": "not_available",
                    "feature_name": "not_available",
                    "shap_rank_or_status": "not_available",
                    "graph_support_status": "not_available",
                    "recommended_claim": "No candidate found in current tables.",
                    "required_assets": "",
                    "notes": "Requires author review or additional evidence.",
                }
            )
            continue
        row = df.iloc[0]
        condition = row["condition"]
        view = row["view"]
        feature = row["feature_name"]
        support = row.get("adjacency_support_level", row.get("graph_support_status", row.get("strict_consistency_type", "")))
        shap_status = row.get("mean_rank", row.get("shap_status", "selected"))
        if pd.isna(shap_status) or str(shap_status).strip() == "":
            shap_status = row.get("shap_status", "selected")
        assets = f"{shap_path(condition, root)}; {dag_path(condition, view, root)}"
        case_rows.append(
            {
                "case_id": case_id,
                "case_type": case_type,
                "condition": condition,
                "view": view,
                "feature_name": feature,
                "shap_rank_or_status": shap_status,
                "graph_support_status": support,
                "recommended_claim": claim,
                "required_assets": assets,
                "notes": "Use conservative language; do not overclaim causal direction unless strict support is present.",
            }
        )

    write_csv(
        final_figures / "selected_case_studies.csv",
        case_rows,
        [
            "case_id",
            "case_type",
            "condition",
            "view",
            "feature_name",
            "shap_rank_or_status",
            "graph_support_status",
            "recommended_claim",
            "required_assets",
            "notes",
        ],
    )

    figures = [
        {
            "figure_id": "Fig1",
            "paper_role": "Framework overview",
            "candidate_path": "to_draw_by_author",
            "condition": "all",
            "view": "pipeline",
            "what_it_shows": "raw vibration -> wavelet tensor -> CNN baseline -> SHAP -> SHAP-weighted PC-DAG -> graph-consistency audit",
            "why_selected": "No polished schematic exists; this is the main workflow figure.",
            "risk_or_caveat": "Must be drawn manually from frozen evidence chain.",
            "status": "to_draw_by_author",
        },
        {
            "figure_id": "Fig2",
            "paper_role": "SHAP attribution example",
            "candidate_path": pick_existing(root, ["CalculateShapValues/NoAttention/TotalAnalysisGraphs/SHAP_All_Conditions_Frequency.png"]),
            "condition": "all",
            "view": "frequency",
            "what_it_shows": "frequency-band SHAP ranking across conditions",
            "why_selected": "Frequency view contains the clearest strict/weak consistency cases.",
            "risk_or_caveat": "Style may need unification for publication.",
            "status": "existing_asset_candidate",
        },
        {
            "figure_id": "Fig3a",
            "paper_role": "channel-view PC-DAG example",
            "candidate_path": dag_path("Bearing_20_0", "channel", root),
            "condition": "Bearing_20_0",
            "view": "channel",
            "what_it_shows": "channel-view label-related graph structure",
            "why_selected": "Contains strict underused channel support candidates.",
            "risk_or_caveat": "Only feature_to_label is strict; other label adjacency is weak.",
            "status": "existing_asset_candidate",
        },
        {
            "figure_id": "Fig3b",
            "paper_role": "frequency-view PC-DAG example",
            "candidate_path": dag_path("Bearing_30_2", "frequency", root),
            "condition": "Bearing_30_2",
            "view": "frequency",
            "what_it_shows": "frequency-view graph structure with strict high-SHAP support",
            "why_selected": "Matches the strongest strict consistent case cD2.",
            "risk_or_caveat": "Use conservative causal-direction language.",
            "status": "existing_asset_candidate",
        },
        {
            "figure_id": "Fig4",
            "paper_role": "conflict/shortcut-risk case",
            "candidate_path": shap_path("Gear_20_0", root) + "; " + dag_path("Gear_20_0", "frequency", root),
            "condition": "Gear_20_0",
            "view": "frequency",
            "what_it_shows": "high-SHAP feature lacking label adjacency",
            "why_selected": "Gear_20_0/cD4 is a weak shortcut-risk candidate.",
            "risk_or_caveat": "Shortcut-risk candidate only; not confirmed shortcut.",
            "status": "existing_asset_candidate",
        },
        {
            "figure_id": "Supp1",
            "paper_role": "baseline curve/confusion supplement",
            "candidate_path": pick_existing(
                root,
                [
                    "ModelTrain/NoAttention/ConfusionAndF1/Seed_49/ConfusionMatrix/Bearing_20_0.png",
                    "ModelTrain/NoAttention/SavedGraphs_6ch/Seed_49/Bearing_20_0.png",
                ],
            ),
            "condition": "Bearing_20_0",
            "view": "baseline",
            "what_it_shows": "existing baseline confusion or training curve asset",
            "why_selected": "Supplementary diagnostic behavior illustration.",
            "risk_or_caveat": f"Numeric table uses 512 crop assumption: {NOTE_512}",
            "status": "supplement_existing_asset_candidate",
        },
    ]
    write_csv(
        final_figures / "selected_main_figures.csv",
        figures,
        ["figure_id", "paper_role", "candidate_path", "condition", "view", "what_it_shows", "why_selected", "risk_or_caveat", "status"],
    )

    notes = [
        "# Figure Selection Notes",
        "",
        "- Existing figures were selected without redrawing or rerunning SHAP/PC-DAG.",
        "- Fig1 should be drawn by the author because no polished framework overview exists.",
        "- PC-DAG figures should be described under the conservative edge policy: feature_to_label is strict support; label_to_feature and undirected edges are weak adjacency only.",
        "- Shortcut-risk cases are candidates, not confirmed shortcuts.",
        f"- Baseline supplement inherits the historical-width caveat: {NOTE_512}.",
    ]
    (final_figures / "figure_selection_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    return figures, case_rows


def build_manifest_and_package(root: Path, evidence: Path, figures: list[dict], cases: list[dict], tables: dict[str, pd.DataFrame]) -> None:
    baseline_rows = build_baseline_summary(root, evidence)
    final_manifest_rows = [
        {
            "asset_path": "reports/paper_evidence/baseline_metrics/baseline_metrics_paper_table.csv",
            "asset_type": "csv",
            "paper_role": "main Table 1 candidate",
            "evidence_strength": "B with historical-width caveat",
            "paper_use_status": "main_candidate_with_width_caveat",
            "known_caveat": NOTE_512,
            "recommended_use": "Report no-attention CNN baseline metrics on test_set only; do not infer cross-condition robustness.",
        },
        {
            "asset_path": "reports/paper_evidence/final_tables/table_shap_stability_main.csv",
            "asset_type": "csv",
            "paper_role": "main SHAP attribution table",
            "evidence_strength": "A-",
            "paper_use_status": "main_candidate",
            "known_caveat": "SHAP cache lacks full config snapshot, but five seeds are available.",
            "recommended_use": "Use for top channel/frequency attribution ranking.",
        },
        {
            "asset_path": "reports/paper_evidence/final_tables/table_strict_consistency_main.csv",
            "asset_type": "csv",
            "paper_role": "strict graph support table",
            "evidence_strength": "A conservative but sparse",
            "paper_use_status": "main_candidate",
            "known_caveat": "Only feature_to_label edges are strict support; sparse evidence.",
            "recommended_use": "Use for strict directional support cases only.",
        },
        {
            "asset_path": "reports/paper_evidence/final_tables/table_weak_adjacency_audit_main.csv",
            "asset_type": "csv",
            "paper_role": "weak graph-consistency audit table",
            "evidence_strength": "B+",
            "paper_use_status": "main_candidate_with_conservative_language",
            "known_caveat": "label_to_feature and undirected_with_label are weak adjacency, not causal direction.",
            "recommended_use": "Use for attribution-graph consistency/conflict audit.",
        },
        {
            "asset_path": "reports/paper_evidence/final_tables/table_shortcut_risk_candidates.csv",
            "asset_type": "csv",
            "paper_role": "shortcut-risk candidates",
            "evidence_strength": "B",
            "paper_use_status": "main_or_supplement_candidate",
            "known_caveat": "Candidates only; no intervention confirms shortcut.",
            "recommended_use": "Use for high-attribution but graph-unsupported risk screening.",
        },
        {
            "asset_path": "reports/paper_evidence/final_figures/selected_main_figures.csv",
            "asset_type": "csv",
            "paper_role": "figure plan",
            "evidence_strength": "B",
            "paper_use_status": "planning_asset",
            "known_caveat": "Some existing figures require publication styling.",
            "recommended_use": "Use to drive final figure drawing and selection.",
        },
    ]
    write_csv(
        evidence / "final_evidence_manifest.csv",
        final_manifest_rows,
        ["asset_path", "asset_type", "paper_role", "evidence_strength", "paper_use_status", "known_caveat", "recommended_use"],
    )

    checklist = [
        {
            "item": "NoAttention baseline metrics on test_set",
            "status": "complete_with_caveat" if baseline_rows else "missing",
            "blocking_level": "P1",
            "evidence_path": "reports/paper_evidence/baseline_metrics/baseline_metrics_paper_table.csv",
            "next_action": "Use with historical-width caveat; do not use cross_test as main evidence.",
            "owner": "Codex/ChatGPT",
        },
        {
            "item": "SHAP stability main table",
            "status": "complete",
            "blocking_level": "non_blocking",
            "evidence_path": "reports/paper_evidence/final_tables/table_shap_stability_main.csv",
            "next_action": "ChatGPT selects final narrative features.",
            "owner": "ChatGPT",
        },
        {
            "item": "Strict feature_to_label consistency",
            "status": "complete_sparse",
            "blocking_level": "non_blocking",
            "evidence_path": "reports/paper_evidence/final_tables/table_strict_consistency_main.csv",
            "next_action": "Use strict direction only for listed cases.",
            "owner": "ChatGPT",
        },
        {
            "item": "Weak adjacency graph-consistency audit",
            "status": "complete",
            "blocking_level": "non_blocking",
            "evidence_path": "reports/paper_evidence/final_tables/table_weak_adjacency_audit_main.csv",
            "next_action": "Use conservative graph-consistency wording.",
            "owner": "ChatGPT",
        },
        {
            "item": "Framework overview figure",
            "status": "to_draw",
            "blocking_level": "P1",
            "evidence_path": "reports/paper_evidence/final_figures/selected_main_figures.csv",
            "next_action": "Author draws clean pipeline diagram.",
            "owner": "human researcher",
        },
        {
            "item": "Cross-condition evidence",
            "status": "limitation_only",
            "blocking_level": "non_blocking",
            "evidence_path": "reports/reproducibility_audit/full_split_overlap_report.csv",
            "next_action": "Do not use as strong robustness evidence.",
            "owner": "ChatGPT",
        },
        {
            "item": "GNN contribution",
            "status": "future_work_only",
            "blocking_level": "non_blocking",
            "evidence_path": "reports/reproducibility_audit/gnn_risk_report.md",
            "next_action": "Keep outside main contribution.",
            "owner": "ChatGPT",
        },
    ]
    write_csv(
        evidence / "final_paper_readiness_checklist.csv",
        checklist,
        ["item", "status", "blocking_level", "evidence_path", "next_action", "owner"],
    )

    strict_df = read_df(evidence / "final_tables" / "table_strict_consistency_main.csv")
    weak_df = read_df(evidence / "final_tables" / "table_weak_adjacency_audit_main.csv")
    shortcut_df = read_df(evidence / "final_tables" / "table_shortcut_risk_candidates.csv")
    underused_df = read_df(evidence / "final_tables" / "table_underused_causal_candidates.csv")

    baseline_text = "\n".join(
        f"- {r['condition']}: accuracy {r['accuracy']}; macro-F1 {r['macro_f1']}; weighted-F1 {r['weighted_f1']}; n={r['n_samples']}; seeds={r['n_seeds']}."
        for r in baseline_rows
    )
    package = [
        "# Final Paper Evidence Package",
        "",
        "## Completed Structured Evidence",
        "",
        "- Baseline metrics were recovered for existing NoAttention checkpoints on `test_set` with `--crop-width 512`.",
        f"- Historical-width caveat: {NOTE_512}.",
        f"- SHAP stability table rows: {len(tables.get('shap', []))}.",
        f"- Strict feature-to-label support rows: {len(strict_df)}.",
        f"- Weak adjacency audit rows: {len(weak_df)}.",
        f"- Shortcut-risk candidate rows: {len(shortcut_df)}.",
        f"- Underused graph-supported candidate rows: {len(underused_df)}.",
        "",
        "## Baseline Summary",
        "",
        baseline_text if baseline_text else "- Baseline summary not available.",
        "",
        "## Main Table Candidates",
        "",
        "- `reports/paper_evidence/baseline_metrics/baseline_metrics_paper_table.csv`",
        "- `reports/paper_evidence/final_tables/table_shap_stability_main.csv`",
        "- `reports/paper_evidence/final_tables/table_strict_consistency_main.csv`",
        "- `reports/paper_evidence/final_tables/table_weak_adjacency_audit_main.csv`",
        "- `reports/paper_evidence/final_tables/table_shortcut_risk_candidates.csv`",
        "- `reports/paper_evidence/final_tables/table_underused_causal_candidates.csv`",
        "",
        "## Main Figure Candidates",
        "",
        "- `reports/paper_evidence/final_figures/selected_main_figures.csv`",
        "- `reports/paper_evidence/final_figures/selected_case_studies.csv`",
        "",
        "## Strong Claims Currently Supported",
        "",
        "- Existing NoAttention CNN checkpoints can be evaluated on the historical 512-width input protocol, with the caveat recorded.",
        "- SHAP provides seed-stability rankings for channel and frequency features.",
        "- Strict feature_to_label PC-DAG support exists for a small set of features.",
        "- Weak label adjacency can support an attribution-graph consistency/conflict audit.",
        "",
        "## Claims That Must Be Weakened",
        "",
        "- Do not describe label_to_feature or undirected_with_label edges as strict causal direction.",
        "- Do not present shortcut-risk candidates as confirmed shortcuts.",
        "- Do not describe baseline metrics without the 512-width historical data caveat.",
        "",
        "## Results Not For Strong Main Claims",
        "",
        "- Old cross-condition results are limitation/risk only due to documented split overlap.",
        "- GNN causal enhancement remains future work rather than a main contribution.",
        "- CWRU/attention experiments remain optional support unless separately reproduced.",
        "",
        "## Recommended First Assets For ChatGPT",
        "",
        "- Start from `final_evidence_manifest.csv` and `final_paper_readiness_checklist.csv`.",
        "- Use baseline paper table for Table 1 with caveat.",
        "- Use SHAP and strict/weak consistency tables for the main graph-consistency audit story.",
        "- Use selected case studies to choose the main narrative examples.",
        "",
        "## Remaining Minimal Points",
        "",
        "- Author should draw the framework overview figure.",
        "- Existing SHAP/DAG figures likely need publication styling.",
        "- Manuscript methods must state the 512-width historical input caveat.",
    ]
    (evidence / "final_evidence_package.md").write_text("\n".join(package) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="reports/paper_evidence")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    evidence = (root / args.out).resolve()
    tables = build_final_tables(root, evidence)
    figures, cases = build_figures_and_cases(root, evidence)
    build_manifest_and_package(root, evidence, figures, cases, tables)
    print(f"Wrote final paper evidence package to {rel(evidence, root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
