#!/usr/bin/env python3
"""Build attribution-causality consistency tables from SHAP and PC-DAG edges."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def relpath(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def edge_index(edge_rows: list[dict]) -> dict[tuple[str, str, str], list[dict]]:
    index: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in edge_rows:
        if row.get("is_label_related") != "True":
            continue
        feature = row["source_node"] if row["target_node"] == "label" else row["target_node"]
        if feature == "label":
            continue
        index[(row["condition"], row["view"], feature)].append(row)
    return index


def classify(is_topk: bool, label_edges: list[dict]) -> tuple[str, str, str, str]:
    if not label_edges:
        if is_topk:
            return "False", "none", "none", "high_shap_no_label_edge_potential_shortcut"
        return "False", "none", "none", "low_shap_no_label_edge_uninformative"
    relation_types = sorted({row["label_relation_type"] for row in label_edges})
    direction = "|".join(sorted({row["edge_direction_interpreted"] for row in label_edges}))
    if any(rel == "feature_to_label" for rel in relation_types):
        if is_topk:
            return "True", "|".join(relation_types), direction, "high_shap_label_edge_consistent"
        return "True", "|".join(relation_types), direction, "low_shap_label_edge_underused_causal_feature"
    if any(rel in {"label_to_feature", "undirected_with_label"} for rel in relation_types):
        return "True", "|".join(relation_types), direction, "unknown_edge_direction"
    return "True", "|".join(relation_types), direction, "unknown_edge_direction"


def classify_strict(is_topk: bool, label_edges: list[dict]) -> tuple[bool, str, str]:
    strict_edges = [row for row in label_edges if row["label_relation_type"] == "feature_to_label"]
    if is_topk and strict_edges:
        return True, "high_shap_strict_causal_support", "strong"
    if is_topk and not strict_edges:
        return False, "high_shap_without_strict_causal_support_potential_shortcut", "moderate"
    if (not is_topk) and strict_edges:
        return True, "low_shap_with_strict_causal_support_underused_feature", "strong"
    return False, "low_shap_without_strict_causal_support", "moderate"


def classify_adjacency(is_topk: bool, label_edges: list[dict]) -> tuple[bool, bool, bool, str, str]:
    has_any = bool(label_edges)
    has_strict = any(row["label_relation_type"] == "feature_to_label" for row in label_edges)
    has_weak_only = has_any and not has_strict
    if has_strict:
        level = "strict_directed_support"
    elif has_weak_only:
        level = "weak_label_adjacency_only"
    else:
        level = "no_label_adjacency"

    if is_topk and has_any:
        ctype = "high_shap_with_label_adjacency"
    elif is_topk and not has_any:
        ctype = "high_shap_without_label_adjacency_potential_shortcut"
    elif (not is_topk) and has_any:
        ctype = "low_shap_with_label_adjacency_underused_or_conflicting"
    else:
        ctype = "low_shap_without_label_adjacency"
    return has_any, has_strict, has_weak_only, level, ctype


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shap", default="reports/paper_evidence/shap_seed_stability_summary.csv")
    parser.add_argument("--edges", default="reports/paper_evidence/pcdag_edges/pcdag_edges_long.csv")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out = (root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out).resolve()
    shap_path = (root / args.shap).resolve()
    edge_path = (root / args.edges).resolve()
    shap_rows = read_csv(shap_path)
    edge_rows = read_csv(edge_path) if edge_path.exists() else []
    edges = edge_index(edge_rows)

    table_rows: list[dict] = []
    strict_rows: list[dict] = []
    adjacency_rows: list[dict] = []
    for shap in shap_rows:
        mean_rank = float(shap["mean_rank"])
        is_topk = mean_rank <= args.topk
        key = (shap["condition"], shap["view"], shap["feature_name"])
        label_edges = edges.get(key, [])
        has_edge, relation_type, direction, consistency = classify(is_topk, label_edges)
        evidence_strength = "strong" if label_edges and consistency in {
            "high_shap_label_edge_consistent",
            "low_shap_label_edge_underused_causal_feature",
        } else "moderate" if consistency in {
            "high_shap_no_label_edge_potential_shortcut",
            "low_shap_no_label_edge_uninformative",
        } else "weak"
        edge_source = "|".join(sorted({row["source_pc_csv"] for row in label_edges})) if label_edges else relpath(edge_path, root) if edge_path.exists() else ""
        table_rows.append(
            {
                "condition": shap["condition"],
                "view": shap["view"],
                "feature_name": shap["feature_name"],
                "mean_importance_across_seeds": shap["mean_importance_across_seeds"],
                "std_importance_across_seeds": shap["std_importance_across_seeds"],
                "mean_rank": shap["mean_rank"],
                "std_rank": shap["std_rank"],
                "topk_frequency": shap["topk_frequency"],
                "is_topk_shap": str(is_topk),
                "has_label_related_causal_edge": has_edge,
                "label_relation_type": relation_type,
                "edge_direction_to_label": direction,
                "consistency_type": consistency,
                "evidence_strength": evidence_strength,
                "shap_source": shap["source_files"],
                "edge_source": edge_source,
                "notes": f"topk_threshold={args.topk}; label_edges={len(label_edges)}",
            }
        )
        has_strict, strict_type, strict_strength = classify_strict(is_topk, label_edges)
        has_any_adj, has_strict_adj, has_weak_only, support_level, adjacency_type = classify_adjacency(is_topk, label_edges)
        strict_sources = "|".join(sorted({row["source_pc_csv"] for row in label_edges if row["label_relation_type"] == "feature_to_label"}))
        adjacency_sources = "|".join(sorted({row["source_pc_csv"] for row in label_edges}))
        strict_rows.append(
            {
                "condition": shap["condition"],
                "view": shap["view"],
                "feature_name": shap["feature_name"],
                "mean_importance_across_seeds": shap["mean_importance_across_seeds"],
                "mean_rank": shap["mean_rank"],
                "std_rank": shap["std_rank"],
                "topk_frequency": shap["topk_frequency"],
                "is_topk_shap": str(is_topk),
                "has_strict_feature_to_label_edge": str(has_strict),
                "strict_consistency_type": strict_type,
                "strict_evidence_strength": strict_strength,
                "edge_source": strict_sources,
                "notes": f"strict policy uses only feature_to_label edges; topk_threshold={args.topk}; label_edges={len(label_edges)}",
            }
        )
        adjacency_rows.append(
            {
                "condition": shap["condition"],
                "view": shap["view"],
                "feature_name": shap["feature_name"],
                "is_topk_shap": str(is_topk),
                "has_any_label_adjacency": str(has_any_adj),
                "has_strict_feature_to_label_edge": str(has_strict_adj),
                "has_only_weak_or_conflicting_label_adjacency": str(has_weak_only),
                "adjacency_support_level": support_level,
                "adjacency_consistency_type": adjacency_type,
                "notes": f"weak adjacency includes label_to_feature/undirected_with_label without causal-direction claim; sources={adjacency_sources}",
            }
        )

    write_csv(
        out / "attribution_causality_consistency_table.csv",
        [
            "condition",
            "view",
            "feature_name",
            "mean_importance_across_seeds",
            "std_importance_across_seeds",
            "mean_rank",
            "std_rank",
            "topk_frequency",
            "is_topk_shap",
            "has_label_related_causal_edge",
            "label_relation_type",
            "edge_direction_to_label",
            "consistency_type",
            "evidence_strength",
            "shap_source",
            "edge_source",
            "notes",
        ],
        table_rows,
    )
    write_csv(
        out / "strict_feature_to_label_consistency_table.csv",
        [
            "condition",
            "view",
            "feature_name",
            "mean_importance_across_seeds",
            "mean_rank",
            "std_rank",
            "topk_frequency",
            "is_topk_shap",
            "has_strict_feature_to_label_edge",
            "strict_consistency_type",
            "strict_evidence_strength",
            "edge_source",
            "notes",
        ],
        strict_rows,
    )
    write_csv(
        out / "adjacency_weak_consistency_table.csv",
        [
            "condition",
            "view",
            "feature_name",
            "is_topk_shap",
            "has_any_label_adjacency",
            "has_strict_feature_to_label_edge",
            "has_only_weak_or_conflicting_label_adjacency",
            "adjacency_support_level",
            "adjacency_consistency_type",
            "notes",
        ],
        adjacency_rows,
    )

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in table_rows:
        grouped[(row["condition"], row["view"])].append(row)
    summary_rows = []
    for (condition, view), items in sorted(grouped.items()):
        counts = Counter(row["consistency_type"] for row in items)
        summary_rows.append(
            {
                "condition": condition,
                "view": view,
                "topk": args.topk,
                "n_features": len(items),
                "n_high_shap_label_edge_consistent": counts["high_shap_label_edge_consistent"],
                "n_high_shap_no_label_edge_potential_shortcut": counts["high_shap_no_label_edge_potential_shortcut"],
                "n_low_shap_label_edge_underused_causal_feature": counts["low_shap_label_edge_underused_causal_feature"],
                "n_low_shap_no_label_edge_uninformative": counts["low_shap_no_label_edge_uninformative"],
                "n_unknown_edge_direction": counts["unknown_edge_direction"],
                "n_missing_edge_evidence": counts["missing_edge_evidence"],
                "topk_features": "|".join(row["feature_name"] for row in items if row["is_topk_shap"] == "True"),
                "label_edge_features": "|".join(row["feature_name"] for row in items if row["has_label_related_causal_edge"] == "True"),
                "status": "ok" if edge_rows else "missing_edge_evidence",
                "notes": "Counts are based on SHAP stability table joined with exported PC-DAG edges.",
            }
        )
    write_csv(
        out / "consistency_summary.csv",
        [
            "condition",
            "view",
            "topk",
            "n_features",
            "n_high_shap_label_edge_consistent",
            "n_high_shap_no_label_edge_potential_shortcut",
            "n_low_shap_label_edge_underused_causal_feature",
            "n_low_shap_no_label_edge_uninformative",
            "n_unknown_edge_direction",
            "n_missing_edge_evidence",
            "topk_features",
            "label_edge_features",
            "status",
            "notes",
        ],
        summary_rows,
    )
    notes = [
        "# Consistency Notes",
        "",
        f"- SHAP source: `{relpath(shap_path, root)}`.",
        f"- Edge source: `{relpath(edge_path, root) if edge_path.exists() else args.edges}`.",
        f"- High-SHAP threshold: mean rank <= {args.topk}.",
        f"- Rows: {len(table_rows)}; summaries: {len(summary_rows)}.",
        f"- Consistency counts: {dict(Counter(row['consistency_type'] for row in table_rows))}.",
        "- `feature_to_label` edges are treated as attribution-causality consistency when the feature is high-SHAP.",
        "- Features with high SHAP but no label-related PC-DAG edge are marked as potential shortcut candidates.",
    ]
    (out / "consistency_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(f"Wrote {len(table_rows)} consistency rows to {relpath(out, root)}")
    print(dict(Counter(row["consistency_type"] for row in table_rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
