#!/usr/bin/env python3
"""Summarize conservative PC-DAG edge policies for paper evidence."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def policy_row(policy: str, rows: list[dict], support_predicate, strength: str, recommendation: str) -> dict:
    high = [row for row in rows if row["is_topk_shap"] == "True"]
    low = [row for row in rows if row["is_topk_shap"] != "True"]
    high_support = [row for row in high if support_predicate(row)]
    low_support = [row for row in low if support_predicate(row)]
    return {
        "policy": policy,
        "n_features_total": len(rows),
        "n_high_shap_features": len(high),
        "n_high_shap_with_support": len(high_support),
        "n_high_shap_without_support": len(high) - len(high_support),
        "n_low_shap_with_support": len(low_support),
        "n_low_shap_without_support": len(low) - len(low_support),
        "interpretation_strength": strength,
        "paper_use_recommendation": recommendation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", required=True)
    parser.add_argument("--edges", default="reports/paper_evidence/pcdag_edges/pcdag_edges_long.csv")
    parser.add_argument("--strict", default="reports/paper_evidence/consistency/strict_feature_to_label_consistency_table.csv")
    parser.add_argument("--adjacency", default="reports/paper_evidence/consistency/adjacency_weak_consistency_table.csv")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out = (root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out).resolve()
    edge_path = (root / args.edges).resolve()
    strict_path = (root / args.strict).resolve()
    adjacency_path = (root / args.adjacency).resolve()
    edges = read_csv(edge_path)
    strict_rows = read_csv(strict_path)
    adjacency_rows = read_csv(adjacency_path)

    edge_counts = Counter(row["label_relation_type"] for row in edges)
    label_edges = [row for row in edges if row["is_label_related"] == "True"]
    bad_direction = [row for row in label_edges if row["label_relation_type"] == "label_to_feature"]
    undirected = [row for row in label_edges if row["label_relation_type"] == "undirected_with_label"]
    strict = [row for row in label_edges if row["label_relation_type"] == "feature_to_label"]

    summary_rows = [
        policy_row(
            "strict_feature_to_label_only",
            strict_rows,
            lambda row: row["has_strict_feature_to_label_edge"] == "True",
            "strong_but_sparse",
            "Use only for conservative causal-direction support; enough for case-study evidence, not broad directional claim.",
        ),
        policy_row(
            "weak_label_adjacency",
            adjacency_rows,
            lambda row: row["has_any_label_adjacency"] == "True",
            "weak_direction_neutral",
            "Use for graph-consistency / label-related adjacency audit; do not claim feature causes label unless strict support exists.",
        ),
    ]
    write_csv(
        out / "edge_policy_summary.csv",
        [
            "policy",
            "n_features_total",
            "n_high_shap_features",
            "n_high_shap_with_support",
            "n_high_shap_without_support",
            "n_low_shap_with_support",
            "n_low_shap_without_support",
            "interpretation_strength",
            "paper_use_recommendation",
        ],
        summary_rows,
    )

    notes = [
        "# Edge Policy Notes",
        "",
        f"- Exported PC-DAG edges: {len(edges)}.",
        f"- Label-related edges: {len(label_edges)}.",
        f"- Edge relation counts: {dict(edge_counts)}.",
        f"- Strong `feature_to_label` edges retained for strict causal-support evidence: {len(strict)}.",
        f"- `label_to_feature` edges discarded from strong causal support: {len(bad_direction)}.",
        f"- `undirected_with_label` edges retained only as weak adjacency: {len(undirected)}.",
        "",
        "## Policy",
        "",
        "- Strict table: only `feature_to_label` supports directional attribution-causality consistency.",
        "- Adjacency table: any label-related edge can support a weaker graph-adjacency statement, without directional causal language.",
        "- `label_to_feature` is not used as causal support because label is a diagnosis/fault class and should not be interpreted as causing vibration features.",
        "",
        "## Recommendation",
        "",
        "- Strong wording should be limited to strict `feature_to_label` cases.",
        "- If strict cases are too sparse for a broad claim, use `attribution-causal-adjacency audit` or `graph-consistency audit` language.",
    ]
    (out / "edge_policy_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")

    sanity = [
        "# PC-DAG Direction Sanity Report",
        "",
        "## Facts Checked",
        "",
        "- `Common/PCCausal.py` and `tools/export_pcdag_edges.py` both build background knowledge intended to forbid `label -> feature`.",
        "- The PC input column order is mapped back from causallearn's `X1...Xn` nodes to feature names plus `label`.",
        "- Edge endpoints are interpreted conservatively: `TAIL/ARROW` means source -> target; `ARROW/TAIL` means target -> source; other endpoint combinations are treated as undirected/partially oriented.",
        "- The exported edge table is based on the original PC graph, not the later visualization-only filtered graph.",
        "",
        "## Direction Finding",
        "",
        f"- `feature_to_label`: {edge_counts.get('feature_to_label', 0)}.",
        f"- `label_to_feature`: {edge_counts.get('label_to_feature', 0)}.",
        f"- `undirected_with_label`: {edge_counts.get('undirected_with_label', 0)}.",
        "",
        "## Interpretation",
        "",
        "- The most likely cause of `label_to_feature` rows is that the original causallearn PC graph still contains orientations that should not be used as physical causal direction evidence.",
        "- The visualization code's filtered graph keeps only feature -> label edges for the displayed filtered DAG, but that filtered graph was not previously persisted as an edge CSV.",
        "- Endpoint reversal is possible in any custom exporter, but the observed `TAIL/ARROW` string representation matches causallearn's printed `X --> Y` convention in smoke tests.",
        "- Therefore `feature_to_label` can be used as strong evidence; `undirected_with_label` can be weak adjacency; `label_to_feature` should be discarded from strong evidence.",
    ]
    (out.parent / "pcdag_edges" / "pcdag_direction_sanity_report.md").write_text("\n".join(sanity) + "\n", encoding="utf-8")
    print(f"Wrote edge policy summary to {out}")
    print(dict(edge_counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
