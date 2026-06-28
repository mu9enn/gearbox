#!/usr/bin/env python3
"""Export PC-DAG edges from existing PC_Datasets CSV files.

This reruns the existing lightweight WaveletDAG PC-KCI procedure on persisted
PC input CSVs only to export edge tables. It does not train models or recompute
SHAP values.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


CHANNEL_COLS = ["ch2", "ch3", "ch4", "ch6", "ch7", "ch8"]
FREQ_COLS = ["cD2", "cD3", "cD4", "cD5", "cA5"]
LABEL_COL = "label"


def relpath(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def git_commit(root: Path) -> str:
    import subprocess

    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def discover_pc_csvs(root: Path, pc_root: Path, include_seeds: set[str], include_average: bool) -> list[tuple[Path, str]]:
    items: list[tuple[Path, str]] = []
    for seed in sorted(include_seeds):
        base = pc_root / f"Seed_{seed}"
        if base.exists():
            items.extend((path, seed) for path in sorted(base.glob("*.csv")))
    if include_average:
        base = pc_root / "average"
        if base.exists():
            items.extend((path, "average") for path in sorted(base.glob("*.csv")))
    return [(path, seed) for path, seed in items if path.is_file()]


def clean_dataset(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df[cols + [LABEL_COL]].copy()
    for col in cols + [LABEL_COL]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)
        out[col] = out[col].fillna(out[col].median())
    return out


def build_background_knowledge(feature_cols: list[str]):
    from causallearn.graph.GraphNode import GraphNode
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

    # Mirrors Common/PCCausal.py: label must not point to feature nodes. The
    # graph object later returns X1..Xn, so exported names are mapped by order.
    nodes = {col: GraphNode(col) for col in feature_cols + [LABEL_COL]}
    bk = BackgroundKnowledge()
    for feature in feature_cols:
        bk.add_forbidden_by_node(nodes[LABEL_COL], nodes[feature])
    return bk


def run_pc(data: pd.DataFrame, feature_cols: list[str], alpha: float, max_cond_vars: int):
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import CIT

    values = data[feature_cols + [LABEL_COL]].values
    ci_test = CIT(data=values, method="kci")
    bk = build_background_knowledge(feature_cols)
    return pc(
        values,
        ci_test=ci_test,
        background_knowledge=bk,
        significance_level=alpha,
        max_cond_vars=max_cond_vars,
        stable=True,
        show_progress=False,
    )


def endpoint_name(endpoint) -> str:
    return str(endpoint).split(".")[-1]


def interpret_edge(node1: str, node2: str, ep1: str, ep2: str) -> tuple[str, str, bool]:
    if ep1 == "TAIL" and ep2 == "ARROW":
        return node1, node2, True
    if ep1 == "ARROW" and ep2 == "TAIL":
        return node2, node1, True
    return node1, node2, False


def label_relation(source: str, target: str, node1: str, node2: str, is_directed: bool) -> tuple[bool, str, str]:
    if LABEL_COL not in {node1, node2}:
        return False, "not_label_related", "none"
    if is_directed and target == LABEL_COL:
        return True, "feature_to_label", f"{source}->label"
    if is_directed and source == LABEL_COL:
        return True, "label_to_feature", f"label->{target}"
    return True, "undirected_with_label", "undirected_or_partially_oriented"


def export_edges_for_graph(
    graph,
    columns: list[str],
    condition: str,
    seed_or_average: str,
    view: str,
    alpha: float,
    source_pc_csv: str,
) -> list[dict]:
    rows: list[dict] = []
    graph_nodes = graph.G.get_nodes()
    name_map = {node.get_name(): columns[idx] for idx, node in enumerate(graph_nodes)}
    for edge in graph.G.get_graph_edges():
        raw_node1 = edge.get_node1().get_name()
        raw_node2 = edge.get_node2().get_name()
        node1 = name_map.get(raw_node1, raw_node1)
        node2 = name_map.get(raw_node2, raw_node2)
        ep1 = endpoint_name(edge.get_endpoint1())
        ep2 = endpoint_name(edge.get_endpoint2())
        source, target, is_directed = interpret_edge(node1, node2, ep1, ep2)
        is_label_related, relation_type, direction_to_label = label_relation(source, target, node1, node2, is_directed)
        physical_allowed = not (source == LABEL_COL and target != LABEL_COL)
        rows.append(
            {
                "condition": condition,
                "seed_or_average": seed_or_average,
                "view": view,
                "setting": "existing_wavelet_dag",
                "alpha": alpha,
                "independence_test": "kci",
                "source_node": source,
                "target_node": target,
                "edge_endpoint_source": ep1 if source == node1 else ep2,
                "edge_endpoint_target": ep2 if target == node2 else ep1,
                "edge_direction_interpreted": direction_to_label if is_label_related else (f"{source}->{target}" if is_directed else "undirected_or_partially_oriented"),
                "is_directed": str(is_directed),
                "is_label_related": str(is_label_related),
                "label_relation_type": relation_type,
                "is_physical_prior_allowed": str(physical_allowed),
                "source_pc_csv": source_pc_csv,
                "graph_source": "causallearn.pc",
                "export_method": "rerun_existing_pcdag_on_persisted_pc_csv",
                "status": "ok",
                "notes": f"raw_edge={raw_node1} {ep1} / {raw_node2} {ep2}; mapped_by_input_column_order",
            }
        )
    return rows


def edge_summary(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, str, str], list[dict]] = {}
    for row in rows:
        key = (row["condition"], row["seed_or_average"], row["view"], row["setting"], str(row["alpha"]))
        grouped.setdefault(key, []).append(row)
    summary = []
    for (condition, seed, view, setting, alpha), items in sorted(grouped.items()):
        label_items = [row for row in items if row["is_label_related"] == "True"]
        to_label = sorted({row["source_node"] for row in label_items if row["label_relation_type"] == "feature_to_label"})
        from_label = sorted({row["target_node"] for row in label_items if row["label_relation_type"] == "label_to_feature"})
        undirected = sorted(
            {
                row["source_node"] if row["target_node"] == LABEL_COL else row["target_node"]
                for row in label_items
                if row["label_relation_type"] == "undirected_with_label"
            }
        )
        connected = sorted(set(to_label + from_label + undirected))
        summary.append(
            {
                "condition": condition,
                "seed_or_average": seed,
                "view": view,
                "setting": setting,
                "alpha": alpha,
                "n_edges_total": len(items),
                "n_label_related_edges": len(label_items),
                "features_connected_to_label": "|".join(connected),
                "features_directed_to_label": "|".join(to_label),
                "features_directed_from_label": "|".join(from_label),
                "features_undirected_with_label": "|".join(undirected),
                "status": "ok",
                "notes": "edge directions interpreted from causallearn endpoints TAIL/ARROW",
            }
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--pc-datasets", default="PC_Datasets")
    parser.add_argument("--out", required=True)
    parser.add_argument("--include-seeds", nargs="*", default=["49"])
    parser.add_argument("--include-average", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    pc_root = (root / args.pc_datasets).resolve()
    out = (root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(root),
        "script": relpath(Path(__file__), root),
        "pc_datasets": relpath(pc_root, root),
        "include_seeds": args.include_seeds,
        "include_average": args.include_average,
        "alpha": args.alpha,
        "independence_test": "kci",
        "algorithm": "causallearn.search.ConstraintBased.PC.pc",
        "status": "started",
        "runs": [],
    }

    all_rows: list[dict] = []
    files = discover_pc_csvs(root, pc_root, set(args.include_seeds), args.include_average)
    for csv_path, seed_or_average in files:
        condition = csv_path.stem
        source_rel = relpath(csv_path, root)
        try:
            df = pd.read_csv(csv_path)
            for view, cols in [("channel", CHANNEL_COLS), ("frequency", FREQ_COLS)]:
                missing = [col for col in cols + [LABEL_COL] if col not in df.columns]
                if missing:
                    raise ValueError(f"{source_rel} missing columns for {view}: {missing}")
                data = clean_dataset(df, cols)
                max_cond_vars = len(data.columns) - 2
                graph = run_pc(data, cols, alpha=args.alpha, max_cond_vars=max_cond_vars)
                columns = cols + [LABEL_COL]
                rows = export_edges_for_graph(graph, columns, condition, seed_or_average, view, args.alpha, source_rel)
                all_rows.extend(rows)
                manifest["runs"].append(
                    {
                        "source_pc_csv": source_rel,
                        "condition": condition,
                        "seed_or_average": seed_or_average,
                        "view": view,
                        "n_rows": int(len(data)),
                        "n_edges": len(rows),
                        "status": "ok",
                    }
                )
        except Exception as exc:  # noqa: BLE001
            manifest["runs"].append(
                {
                    "source_pc_csv": source_rel,
                    "condition": condition,
                    "seed_or_average": seed_or_average,
                    "view": "unknown",
                    "status": "error",
                    "error": repr(exc),
                }
            )

    long_fields = [
        "condition",
        "seed_or_average",
        "view",
        "setting",
        "alpha",
        "independence_test",
        "source_node",
        "target_node",
        "edge_endpoint_source",
        "edge_endpoint_target",
        "edge_direction_interpreted",
        "is_directed",
        "is_label_related",
        "label_relation_type",
        "is_physical_prior_allowed",
        "source_pc_csv",
        "graph_source",
        "export_method",
        "status",
        "notes",
    ]
    write_csv(out / "pcdag_edges_long.csv", long_fields, all_rows)
    summary_rows = edge_summary(all_rows)
    write_csv(
        out / "pcdag_edge_label_summary.csv",
        [
            "condition",
            "seed_or_average",
            "view",
            "setting",
            "alpha",
            "n_edges_total",
            "n_label_related_edges",
            "features_connected_to_label",
            "features_directed_to_label",
            "features_directed_from_label",
            "features_undirected_with_label",
            "status",
            "notes",
        ],
        summary_rows,
    )
    manifest["status"] = "ok" if all(run.get("status") == "ok" for run in manifest["runs"]) else "partial_or_failed"
    manifest["n_edges_total"] = len(all_rows)
    manifest["n_label_related_edges"] = sum(row["is_label_related"] == "True" for row in all_rows)
    (out / "pcdag_run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(all_rows)} edges to {relpath(out / 'pcdag_edges_long.csv', root)}")
    print(f"Label-related edges: {manifest['n_label_related_edges']}")
    print(f"Runs: {len(manifest['runs'])}; status={manifest['status']}")
    return 0 if manifest["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
