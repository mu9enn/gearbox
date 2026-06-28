#!/usr/bin/env python3
"""Extract paper-evidence tables from existing local assets.

This script is read-only with respect to experiment assets. It does not train,
evaluate checkpoints, recompute SHAP, or rerun PC-KCI.
"""

from __future__ import annotations

import argparse
import csv
import re
import zipfile
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np


CONDITIONS = ("Bearing_20_0", "Bearing_30_2", "Gear_20_0", "Gear_30_2")
CHANNEL_NAMES = ("ch2", "ch3", "ch4", "ch6", "ch7", "ch8")
FREQ_NAMES = ("cA5", "cD5", "cD4", "cD3", "cD2")
BASELINE_METRIC_FIELDS = [
    "condition",
    "seed",
    "model_family",
    "experiment_family",
    "split",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "precision_macro",
    "recall_macro",
    "loss",
    "source_path",
    "extraction_method",
    "status",
    "notes",
]


def relpath(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def infer_condition(path: Path) -> str:
    text = path.as_posix()
    for condition in CONDITIONS:
        if condition in text:
            return condition
    return "unknown"


def infer_seed(path: Path) -> str:
    match = re.search(r"Seed_(\d+)", path.as_posix())
    return match.group(1) if match else "unknown"


def infer_model_family(path: Path) -> str:
    text = path.as_posix()
    name = path.name
    if "AttentionAndNoAttention" in text:
        if "_Attention" in name:
            return "attention"
        if "_NoAttention" in name:
            return "no_attention"
        return "attention_and_no_attention"
    if "NoAttention" in text:
        return "no_attention"
    return "unknown"


def infer_experiment_family(path: Path) -> str:
    text = path.as_posix()
    if "SavedGraphs_6ch" in text or "SavedModels_6ch" in text:
        return "baseline_6ch"
    if "SavedGraphs_4ch" in text or "SavedModels_4ch" in text:
        return "channel_ablation_4ch"
    if "SavedGraphs_freq" in text or "SavedModels_freq" in text or "SHAP_freq" in text:
        return "frequency_ablation"
    if "BaseGraphs" in text or "BaseModels" in text or "BaseModelTrain" in text:
        return "channel_frequency_ablation"
    if "ConfusionAndF1" in text:
        return "confusion_f1"
    if "CrossCondTest" in text:
        return "cross_condition_risk_reference"
    if "AttentionAndNoAttention" in text:
        return "attention_comparison"
    return "unknown"


def asset_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "figure"
    if suffix == ".pth":
        return "checkpoint"
    if suffix == ".npz":
        return "numeric_npz"
    if suffix in {".csv", ".json", ".log"}:
        return "structured_metric_candidate"
    return suffix.lstrip(".") or "unknown"


def baseline_inventory(root: Path) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    metrics_rows: list[dict] = []
    scan_roots = [
        root / "ModelTrain" / "NoAttention",
        root / "ModelTrain" / "AttentionAndNoAttention",
    ]
    suffixes = {".png", ".pth", ".npz", ".csv", ".json", ".log"}
    for scan_root in scan_roots:
        if not scan_root.exists():
            continue
        for path in sorted(p for p in scan_root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes):
            rel = relpath(path, root)
            kind = asset_type(path)
            experiment = infer_experiment_family(path)
            contains_metrics = kind == "structured_metric_candidate"
            contains_curve = "SavedGraphs" in rel or "BaseGraphs" in rel or "train" in path.stem.lower()
            contains_confusion = "ConfusionMatrix" in rel or "/F1/" in rel
            contains_checkpoint = path.suffix.lower() == ".pth"
            paper_role = "main_candidate" if experiment == "baseline_6ch" else "supplement_candidate"
            notes = []
            if experiment == "cross_condition_risk_reference":
                paper_role = "risk_reference_only"
                notes.append("risk_reference_only_due_to_overlap")
            if path.suffix.lower() == ".png" and (contains_curve or contains_confusion):
                notes.append("figure_only_needs_numeric_export")
            if contains_checkpoint:
                notes.append("checkpoint_available_but_not_evaluated_this_round")
            rows.append(
                {
                    "relative_path": rel,
                    "asset_type": kind,
                    "condition": infer_condition(path),
                    "seed": infer_seed(path),
                    "model_family": infer_model_family(path),
                    "experiment_family": experiment,
                    "contains_metrics": str(contains_metrics),
                    "contains_curve": str(contains_curve),
                    "contains_confusion": str(contains_confusion),
                    "contains_checkpoint": str(contains_checkpoint),
                    "paper_role_candidate": paper_role,
                    "notes": "; ".join(notes),
                }
            )
            if contains_metrics:
                status = "structured_metrics_available_unparsed"
                notes_text = "existing structured file found; parser not implemented for unknown schema"
            elif path.suffix.lower() == ".png":
                status = "figure_only_needs_numeric_export"
                notes_text = "no OCR or image-value inference performed"
            elif contains_checkpoint:
                status = "checkpoint_needs_lightweight_eval_export"
                notes_text = "checkpoint exists but this sprint does not run inference"
            else:
                status = "not_metric_source"
                notes_text = "asset is not a direct metric table"
            metrics_rows.append(
                {
                    "condition": infer_condition(path),
                    "seed": infer_seed(path),
                    "model_family": infer_model_family(path),
                    "experiment_family": experiment,
                    "split": "unknown",
                    "accuracy": "",
                    "macro_f1": "",
                    "weighted_f1": "",
                    "precision_macro": "",
                    "recall_macro": "",
                    "loss": "",
                    "source_path": rel,
                    "extraction_method": "inventory_only_no_inference",
                    "status": status,
                    "notes": notes_text,
                }
            )
    return rows, metrics_rows


def npz_keys_and_shapes(path: Path) -> dict[str, tuple[tuple[int, ...], str]]:
    info = {}
    with zipfile.ZipFile(path) as zf:
        for member in zf.infolist():
            if not member.filename.endswith(".npy"):
                continue
            key = Path(member.filename).stem
            with zf.open(member) as fp:
                version = np.lib.format.read_magic(fp)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(fp)
                else:
                    shape, _, dtype = np.lib.format.read_array_header_2_0(fp)
            info[key] = (tuple(shape), str(dtype))
    return info


def condition_from_shap_file(path: Path) -> str:
    stem = path.stem.replace("_ShapValues", "")
    return stem if stem in CONDITIONS else infer_condition(path)


def aggregate_shap(path: Path) -> tuple[list[dict], dict]:
    condition = condition_from_shap_file(path)
    seed = infer_seed(path)
    key_info = npz_keys_and_shapes(path)
    if "shap_values" not in key_info:
        return [], {"status": "missing_shap_values", "keys": sorted(key_info)}
    try:
        with np.load(path, allow_pickle=False) as data:
            shap_values = data["shap_values"]
            channel_vals = np.mean(np.abs(shap_values), axis=(0, 2, 3, 4))
            freq_vals = np.mean(np.abs(shap_values[:, :, 0:-1, :, :]), axis=(0, 1, 3, 4))
    except Exception as exc:  # noqa: BLE001
        return [], {"status": "load_error", "error": repr(exc), "keys": sorted(key_info)}
    rows: list[dict] = []
    for view, names, values in (
        ("channel", CHANNEL_NAMES, channel_vals),
        ("frequency", FREQ_NAMES, freq_vals),
    ):
        total = float(values.sum())
        normalized = values / total if total else values
        order = np.argsort(-normalized)
        ranks = {int(idx): rank + 1 for rank, idx in enumerate(order)}
        for idx, (name, raw_value, norm_value) in enumerate(zip(names, values, normalized)):
            rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "view": view,
                    "feature_name": name,
                    "feature_index": idx,
                    "mean_abs_shap": f"{float(raw_value):.12g}",
                    "normalized_importance": f"{float(norm_value):.12g}",
                    "rank": ranks[idx],
                    "source_npz": "",
                    "aggregation_method": "mean_abs_over_samples_channels_or_freq_time_class; freq_uses_shap_values[:,:,0:-1,:,:]",
                    "status": "ok",
                    "notes": f"keys={sorted(key_info)}; shape={key_info['shap_values'][0]}; dtype={key_info['shap_values'][1]}",
                }
            )
    return rows, {"status": "ok", "keys": sorted(key_info), "shape": key_info["shap_values"][0]}


def shap_tables(root: Path) -> tuple[list[dict], list[dict], list[str]]:
    rows: list[dict] = []
    notes: list[str] = []
    for path in sorted((root / "CalculateShapValues" / "NoAttention").glob("Seed_*/*.npz")):
        shap_rows, meta = aggregate_shap(path)
        if meta.get("status") != "ok":
            notes.append(f"{relpath(path, root)}: {meta}")
        for row in shap_rows:
            row["source_npz"] = relpath(path, root)
        rows.extend(shap_rows)

    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["view"], row["feature_name"])].append(row)

    stability_rows: list[dict] = []
    for (condition, view, feature_name), items in sorted(grouped.items()):
        importances = np.array([float(item["normalized_importance"]) for item in items])
        ranks = np.array([float(item["rank"]) for item in items])
        topk_frequency = float(np.mean(ranks <= 3))
        sources = "|".join(sorted({item["source_npz"] for item in items}))
        stability_rows.append(
            {
                "condition": condition,
                "view": view,
                "feature_name": feature_name,
                "mean_importance_across_seeds": f"{float(importances.mean()):.12g}",
                "std_importance_across_seeds": f"{float(importances.std(ddof=0)):.12g}",
                "mean_rank": f"{float(ranks.mean()):.12g}",
                "std_rank": f"{float(ranks.std(ddof=0)):.12g}",
                "topk_frequency": f"{topk_frequency:.6f}",
                "n_seeds": len(items),
                "source_files": sources,
                "status": "ok" if len(items) >= 3 else "partial_seed_coverage",
                "notes": "topk_frequency uses rank<=3; frequency names follow PCDataset/WaveletDAG five-band order cA5,cD5,cD4,cD3,cD2",
            }
        )
    return rows, stability_rows, notes


def parse_dag_asset(path: Path) -> dict:
    rel = path.as_posix()
    view = "channel" if path.stem.endswith("_ch") else "frequency" if path.stem.endswith("_freq") else "unknown"
    setting_match = re.search(r"Significance_([^/]+)", rel)
    setting = setting_match.group(1) if setting_match else "unknown"
    original_or_filtered = "filtered" if "/filtered/" in rel else "original" if "/original/" in rel else "unknown"
    seed = infer_seed(path)
    return {
        "view": view,
        "alpha_or_setting": setting,
        "original_or_filtered": original_or_filtered,
        "seed": seed,
    }


def pcdag_inventory(root: Path) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    for base in [root / "PC_Datasets", root / "DAG"]:
        if not base.exists():
            continue
        for path in sorted(p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".png", ".pdf", ".npz"}):
            rel = relpath(path, root)
            dag_meta = parse_dag_asset(path)
            is_edge_table = False
            notes = []
            if path.suffix.lower() == ".csv" and path.parent.name in {"Seed_49", "average"}:
                asset_kind = "pc_dataset_csv"
                notes.append("PC input dataset, not learned edge table")
            elif path.suffix.lower() in {".png", ".pdf"} and "PC_DAG" in rel:
                asset_kind = "dag_figure"
                notes.append("DAG figure available; edge table not persisted")
            elif path.suffix.lower() == ".npz":
                asset_kind = "numeric_intermediate"
            else:
                asset_kind = asset_type(path)
            rows.append(
                {
                    "relative_path": rel,
                    "asset_type": asset_kind,
                    "condition": infer_condition(path),
                    "seed": dag_meta["seed"],
                    "view": dag_meta["view"],
                    "alpha_or_setting": dag_meta["alpha_or_setting"],
                    "original_or_filtered": dag_meta["original_or_filtered"],
                    "contains_edge_table": str(is_edge_table),
                    "contains_dag_figure": str(asset_kind == "dag_figure"),
                    "paper_role_candidate": "main_candidate" if asset_kind in {"dag_figure", "pc_dataset_csv"} else "reference",
                    "notes": "; ".join(notes),
                }
            )
    edge_rows = [
        {
            "condition": "",
            "seed": "",
            "view": "",
            "setting": "",
            "source_node": "",
            "target_node": "",
            "edge_direction": "",
            "edge_weight_or_score": "",
            "is_label_related": "",
            "is_physical_prior_allowed": "",
            "source_path": "",
            "extraction_method": "not_available",
            "status": "needs_pcdag_rerun_for_edge_csv",
            "notes": "Existing DAG figures are available, but edge tables are not persisted; future PC-DAG generation should export edge CSV.",
        }
    ]
    return rows, edge_rows


def consistency_candidates(shap_stability_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for row in shap_stability_rows:
        shap_rank = float(row["mean_rank"])
        is_topk = shap_rank <= 3
        rows.append(
            {
                "condition": row["condition"],
                "seed_or_average": "across_existing_seeds",
                "view": row["view"],
                "feature_name": row["feature_name"],
                "shap_rank": row["mean_rank"],
                "shap_importance": row["mean_importance_across_seeds"],
                "is_topk_shap": str(is_topk),
                "has_label_related_causal_edge": "unknown",
                "edge_direction_to_label": "unknown",
                "consistency_type": "unknown_missing_edge_table",
                "evidence_status": "needs_pcdag_edge_export",
                "shap_source": row["source_files"],
                "dag_source": "DAG/PC_DAG/Seed_49",
                "notes": "SHAP side is extracted; causal edge side is unknown because PC-DAG edge CSV is not persisted.",
            }
        )
    return rows


def paper_asset_manifest(root: Path) -> list[dict]:
    entries = [
        ("ModelTrain/NoAttention/SavedGraphs_6ch", "figures", "baseline_6ch", "Diagnostic Baseline", "B", True, True, True, "figure_only_no_metrics_csv", "export numeric metrics table"),
        ("ModelTrain/NoAttention/ConfusionAndF1", "figures", "confusion_f1", "Supplement", "B", False, True, True, "figure_only", "export classification_report.csv"),
        ("CalculateShapValues/NoAttention/Seed_*", "large_npz", "shap_no_attention", "Explanation", "A", False, False, False, "large_cache_no_manifest", "use extracted rank summaries"),
        ("CalculateShapValues/NoAttention/TotalAnalysisGraphs", "figures", "shap_summary", "Explanation", "A", True, True, False, "figure_only_has_extracted_csv", "candidate main figures"),
        ("PC_Datasets/Seed_49", "csv", "shap_to_pc_dataset", "Causal Audit", "B", True, True, True, "input_table_not_edge_table", "confirm data contract before final rerun"),
        ("PC_Datasets/average", "csv", "shap_to_pc_dataset_average", "Causal Audit", "B", True, True, False, "input_table_not_edge_table", "use as candidate if provenance accepted"),
        ("DAG/PC_DAG/Seed_49", "figures", "dual_view_pcdag", "Causal Audit", "A", True, True, True, "edge_table_missing", "export edge CSV next"),
        ("reports/paper_evidence/attribution_causality_consistency_candidates.csv", "csv", "consistency_candidates", "Causal Audit", "B", True, True, True, "causal_edge_unknown", "fill edge columns after edge export"),
        ("ModelTrain/NoAttention/CrossCondTest", "figure", "cross_condition", "Risk/Limitation", "C", False, True, True, "train_cross_test_overlap", "do not use as strong evidence"),
        ("GNNCausal", "pt/png/pth", "gnn_causal", "Future Work", "C", False, True, True, "classifier_x_and_scaler_risk", "quarantine until fixed"),
        ("CWRU", "png/pth", "external_validation", "Supplement/Future", "C", False, True, True, "metrics_not_unified", "optional later"),
    ]
    rows = []
    for asset_path, asset_kind, family, section, strength, main, supp, repro, risk, action in entries:
        exists = (root / asset_path.replace("*", "")).exists() if "*" not in asset_path else bool(list(root.glob(asset_path)))
        rows.append(
            {
                "asset_path": asset_path,
                "asset_type": asset_kind,
                "experiment_family": family,
                "paper_section_candidate": section,
                "evidence_strength": strength,
                "can_use_in_main": str(main and exists),
                "can_use_in_supplement": str(supp and exists),
                "needs_reproduction": str(repro),
                "known_risk": risk,
                "recommended_action": action,
                "notes": "exists" if exists else "missing_or_pattern_unresolved",
            }
        )
    return rows


def write_notes(path: Path, counts: dict[str, int], shap_notes: list[str]) -> None:
    lines = [
        "# Paper Evidence Extraction Notes",
        "",
        "## Extracted",
        "",
        f"- Baseline asset inventory rows: {counts['baseline_inventory']}.",
        f"- Baseline metric rows: {counts['baseline_metrics']}; numeric fields are mostly unavailable because existing assets are figures/checkpoints, not metrics CSV.",
        f"- SHAP rank rows: {counts['shap_rank']}; seed stability rows: {counts['shap_stability']}.",
        f"- PC-DAG asset inventory rows: {counts['pcdag_inventory']}.",
        f"- Consistency candidate rows: {counts['consistency_candidates']}.",
        "",
        "## Missing Or Limited",
        "",
        "- Existing baseline curves/confusion/F1 are mostly PNG-only; no OCR or image-value inference was performed.",
        "- Existing DAG figures are available, but learned edge tables are not persisted.",
        "- Consistency candidates therefore contain reliable SHAP-side ranks but unknown causal-edge columns.",
        "- Cross-condition assets are marked risk/reference only because current cross split has train/cross_test exact duplicates.",
        "- GNN assets are marked risk/reference only because GNNCausalSEU currently feeds `x` rather than propagated `out` into the classifier.",
        "",
        "## SHAP Notes",
        "",
        "- Aggregation follows existing NoAttention scripts: mean absolute SHAP over samples, non-target axes, time, and class outputs.",
        "- Frequency extraction uses `shap_values[:,:,0:-1,:,:]` and names the five bands as `cA5,cD5,cD4,cD3,cD2`, matching `PCDataset(wavelet).py` and `DAG/WaveletDAG.py`.",
    ]
    if shap_notes:
        lines.append("- SHAP extraction warnings:")
        lines.extend(f"  - {note}" for note in shap_notes)
    lines.extend(
        [
            "",
            "## Next Minimal Engineering Step",
            "",
            "- Add metric export to evaluation/confusion scripts without retraining.",
            "- Add edge CSV export to PC-DAG generation next time PC-KCI is run.",
            "- Join exported PC-DAG edges with `shap_seed_stability_summary.csv` to finalize the consistency/conflict audit table.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="reports/paper_evidence")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out = (root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    baseline_rows, metric_rows = baseline_inventory(root)
    shap_rows, shap_stability_rows, shap_notes = shap_tables(root)
    pcdag_rows, edge_rows = pcdag_inventory(root)
    consistency_rows = consistency_candidates(shap_stability_rows)
    manifest_rows = paper_asset_manifest(root)

    write_csv(out / "baseline_metric_inventory.csv", [
        "relative_path", "asset_type", "condition", "seed", "model_family", "experiment_family",
        "contains_metrics", "contains_curve", "contains_confusion", "contains_checkpoint",
        "paper_role_candidate", "notes",
    ], baseline_rows)
    write_csv(out / "baseline_metrics_from_existing_assets.csv", BASELINE_METRIC_FIELDS, metric_rows)
    write_csv(out / "shap_rank_summary.csv", [
        "condition", "seed", "view", "feature_name", "feature_index", "mean_abs_shap",
        "normalized_importance", "rank", "source_npz", "aggregation_method", "status", "notes",
    ], shap_rows)
    write_csv(out / "shap_seed_stability_summary.csv", [
        "condition", "view", "feature_name", "mean_importance_across_seeds",
        "std_importance_across_seeds", "mean_rank", "std_rank", "topk_frequency",
        "n_seeds", "source_files", "status", "notes",
    ], shap_stability_rows)
    write_csv(out / "pcdag_asset_inventory.csv", [
        "relative_path", "asset_type", "condition", "seed", "view", "alpha_or_setting",
        "original_or_filtered", "contains_edge_table", "contains_dag_figure",
        "paper_role_candidate", "notes",
    ], pcdag_rows)
    write_csv(out / "pcdag_edge_summary.csv", [
        "condition", "seed", "view", "setting", "source_node", "target_node",
        "edge_direction", "edge_weight_or_score", "is_label_related",
        "is_physical_prior_allowed", "source_path", "extraction_method", "status", "notes",
    ], edge_rows)
    write_csv(out / "attribution_causality_consistency_candidates.csv", [
        "condition", "seed_or_average", "view", "feature_name", "shap_rank",
        "shap_importance", "is_topk_shap", "has_label_related_causal_edge",
        "edge_direction_to_label", "consistency_type", "evidence_status",
        "shap_source", "dag_source", "notes",
    ], consistency_rows)
    write_csv(out / "paper_asset_manifest.csv", [
        "asset_path", "asset_type", "experiment_family", "paper_section_candidate",
        "evidence_strength", "can_use_in_main", "can_use_in_supplement",
        "needs_reproduction", "known_risk", "recommended_action", "notes",
    ], manifest_rows)

    counts = {
        "baseline_inventory": len(baseline_rows),
        "baseline_metrics": len(metric_rows),
        "shap_rank": len(shap_rows),
        "shap_stability": len(shap_stability_rows),
        "pcdag_inventory": len(pcdag_rows),
        "consistency_candidates": len(consistency_rows),
    }
    write_notes(out / "extraction_notes.md", counts, shap_notes)
    print(f"Wrote paper evidence tables to {out.relative_to(root)}")
    for key, value in counts.items():
        print(f"{key}: {value}")
    print(f"pcdag_edge_summary: {len(edge_rows)}")
    print(f"paper_asset_manifest: {len(manifest_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
