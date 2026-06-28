#!/usr/bin/env python3
"""Lightweight reproducibility/data-contract audit for the gearbox project.

This tool reads metadata and small arrays only. It does not train models, run
SHAP, run PC-KCI, or execute any experiment script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import re
import subprocess
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np


NPZ_SCAN_PATTERNS = (
    "wavelet_dataset/*.npz",
    "DAG/**/*.npz",
    "CWRU/**/*.npz",
)

KEY_USAGE_SCRIPT_PATTERNS = (
    "Common/*.py",
    "ModelTrain/**/*.py",
    "CalculateShapValues/**/*.py",
    "GNNCausal/**/*.py",
    "CWRU/**/*.py",
)

ASSET_ROOTS = (
    "DataSets",
    "wavelet_dataset",
    "CalculateShapValues",
    "ModelTrain",
    "DAG",
    "PC_Datasets",
    "GNNCausal",
    "CWRU",
)

ASSET_EXTENSIONS = {
    ".npz",
    ".npy",
    ".csv",
    ".json",
    ".jsonl",
    ".pth",
    ".pt",
    ".png",
    ".log",
    ".md",
    ".mat",
}

SPLIT_DATA_KEYS = {
    "train": "train_set",
    "valid": "valid_set",
    "test": "test_set",
    "cross": "cross_set",
    "cross_finetune": "cross_finetune_set",
    "cross_test": "cross_test_set",
}

SPLIT_LABEL_KEYS = {
    "train": "train_label",
    "valid": "valid_label",
    "test": "test_label",
    "cross": "cross_label",
    "cross_finetune": "cross_finetune_label",
    "cross_test": "cross_test_label",
}

INTERESTING_KEY_NAMES = {
    *SPLIT_DATA_KEYS.values(),
    *SPLIT_LABEL_KEYS.values(),
    "all_data",
    "all_labels",
    "shap_values",
    "label",
    "labels",
    "data",
}


@dataclass
class ArrayMeta:
    key: str
    shape: tuple[int, ...]
    dtype: str
    error: str = ""


def relpath(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def npz_header_metadata(path: Path) -> list[ArrayMeta]:
    """Read npy headers from an npz zip without materializing full arrays."""
    rows: list[ArrayMeta] = []
    try:
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                if not info.filename.endswith(".npy"):
                    continue
                key = Path(info.filename).stem
                try:
                    with zf.open(info) as fp:
                        version = np.lib.format.read_magic(fp)
                        if version == (1, 0):
                            shape, _, dtype = np.lib.format.read_array_header_1_0(fp)
                        elif version == (2, 0):
                            shape, _, dtype = np.lib.format.read_array_header_2_0(fp)
                        else:
                            shape, _, dtype = np.lib.format.read_array_header_2_0(fp)
                    rows.append(ArrayMeta(key=key, shape=tuple(shape), dtype=str(dtype)))
                except Exception as exc:  # noqa: BLE001 - report and continue
                    rows.append(ArrayMeta(key=key, shape=(), dtype="", error=repr(exc)))
    except Exception as exc:  # noqa: BLE001 - report and continue
        rows.append(ArrayMeta(key="__npz_open_error__", shape=(), dtype="", error=repr(exc)))
    return rows


def safe_load_array(npz_path: Path, key: str):
    with np.load(npz_path, allow_pickle=False) as data:
        return data[key]


def format_label_distribution(values: np.ndarray, max_unique: int = 40) -> str:
    flat = np.asarray(values).reshape(-1)
    counter = Counter(flat.tolist())
    parts = []
    for label, count in sorted(counter.items(), key=lambda item: str(item[0]))[:max_unique]:
        parts.append(f"{label}:{count}")
    if len(counter) > max_unique:
        parts.append(f"...(+{len(counter) - max_unique} labels)")
    return "|".join(parts)


def discover_npz_files(root: Path) -> list[Path]:
    found: set[Path] = set()
    for pattern in NPZ_SCAN_PATTERNS:
        found.update(root.glob(pattern))
    return sorted(path for path in found if path.is_file())


def discover_all_local_npz_files(root: Path) -> list[Path]:
    found: list[Path] = []
    for asset_root in ASSET_ROOTS:
        base = root / asset_root
        if base.exists():
            found.extend(base.rglob("*.npz"))
    return sorted(path for path in found if path.is_file())


def audit_npz_contracts(root: Path, out: Path, max_nan_elements: int) -> dict[str, list[str]]:
    rows: list[dict] = []
    path_to_keys: dict[str, list[str]] = {}
    for npz_path in discover_npz_files(root):
        rel = relpath(npz_path, root)
        metas = npz_header_metadata(npz_path)
        keys = [meta.key for meta in metas]
        path_to_keys[rel] = keys
        for meta in metas:
            nan_count = ""
            label_distribution = ""
            if meta.error:
                nan_count = f"skipped: {meta.error}"
            else:
                elements = math.prod(meta.shape) if meta.shape else 0
                is_numeric = np.dtype(meta.dtype).kind in {"b", "i", "u", "f", "c"}
                if is_numeric and elements <= max_nan_elements:
                    try:
                        arr = safe_load_array(npz_path, meta.key)
                        if np.issubdtype(arr.dtype, np.number):
                            nan_count = str(int(np.isnan(arr).sum())) if np.issubdtype(arr.dtype, np.floating) else "0"
                    except Exception as exc:  # noqa: BLE001
                        nan_count = f"skipped: {repr(exc)}"
                elif is_numeric:
                    nan_count = f"skipped: elements_gt_{max_nan_elements}"
                else:
                    nan_count = "skipped: non_numeric"

                if "label" in meta.key.lower() and elements <= max_nan_elements:
                    try:
                        label_distribution = format_label_distribution(safe_load_array(npz_path, meta.key))
                    except Exception as exc:  # noqa: BLE001
                        label_distribution = f"skipped: {repr(exc)}"
                elif "label" in meta.key.lower():
                    label_distribution = f"skipped: elements_gt_{max_nan_elements}"

            rows.append(
                {
                    "relative_path": rel,
                    "file_size_mb": f"{file_size_mb(npz_path):.3f}",
                    "keys": "|".join(keys),
                    "array_name": meta.key,
                    "shape": "x".join(str(part) for part in meta.shape),
                    "dtype": meta.dtype,
                    "nan_count_if_numeric": nan_count,
                    "label_distribution_if_label": label_distribution,
                }
            )

    write_csv(
        out / "npz_contracts.csv",
        [
            "relative_path",
            "file_size_mb",
            "keys",
            "array_name",
            "shape",
            "dtype",
            "nan_count_if_numeric",
            "label_distribution_if_label",
        ],
        rows,
    )
    return path_to_keys


def deterministic_indices(n_samples: int, limit: int) -> np.ndarray:
    if limit <= 0 or n_samples <= limit:
        return np.arange(n_samples)
    return np.unique(np.linspace(0, n_samples - 1, num=limit, dtype=np.int64))


def sample_hashes(arr: np.ndarray, limit: int) -> tuple[set[str], int, str]:
    n_samples = int(arr.shape[0]) if arr.ndim else 0
    indices = deterministic_indices(n_samples, limit)
    hashes: set[str] = set()
    for idx in indices:
        sample = np.ascontiguousarray(arr[idx])
        digest = hashlib.sha1()
        digest.update(str(sample.shape).encode("utf-8"))
        digest.update(str(sample.dtype).encode("utf-8"))
        digest.update(sample.view(np.uint8))
        hashes.add(digest.hexdigest())
    method = "full_sha1" if len(indices) == n_samples else f"sampled_deterministic_sha1_{len(indices)}"
    return hashes, len(indices), method


def audit_split_overlap(root: Path, out: Path, max_samples_per_split: int, auto_sample_large_mb: float) -> list[dict]:
    rows: list[dict] = []
    for npz_path in sorted((root / "wavelet_dataset").glob("*.npz")):
        rel = relpath(npz_path, root)
        try:
            with np.load(npz_path, allow_pickle=False) as data:
                available = set(data.files)
                split_hashes: dict[str, set[str]] = {}
                split_ns: dict[str, int] = {}
                split_sample_ns: dict[str, int] = {}
                split_methods: dict[str, str] = {}
                notes_by_split: dict[str, str] = {}

                auto_limit = max_samples_per_split
                if auto_limit <= 0 and file_size_mb(npz_path) > auto_sample_large_mb:
                    auto_limit = 512

                for split, key in SPLIT_DATA_KEYS.items():
                    if key not in available:
                        notes_by_split[split] = f"missing data key {key}; available={sorted(available)}"
                        continue
                    arr = data[key]
                    split_ns[split] = int(arr.shape[0])
                    limit = auto_limit
                    hashes, sampled_n, method = sample_hashes(arr, limit)
                    split_hashes[split] = hashes
                    split_sample_ns[split] = sampled_n
                    split_methods[split] = method
                    label_key = SPLIT_LABEL_KEYS.get(split)
                    if label_key and label_key not in available:
                        notes_by_split[split] = f"missing label key {label_key}"

                preferred_pairs = [
                    ("train", "cross_finetune"),
                    ("train", "test"),
                    ("train", "valid"),
                    ("valid", "test"),
                    ("cross_finetune", "cross_test"),
                ]
                observed_pairs = list(combinations(sorted(split_hashes), 2))
                all_pairs = []
                for pair in preferred_pairs + observed_pairs:
                    if pair[0] in split_hashes and pair[1] in split_hashes and pair not in all_pairs:
                        all_pairs.append(pair)

                if not split_hashes:
                    rows.append(
                        {
                            "npz_path": rel,
                            "split_a": "",
                            "split_b": "",
                            "n_a": "",
                            "n_b": "",
                            "exact_duplicate_count": "",
                            "duplicate_rate_a": "",
                            "duplicate_rate_b": "",
                            "method": "not_checked",
                            "notes": "; ".join(notes_by_split.values()) or f"no expected split keys; available={sorted(available)}",
                        }
                    )
                    continue

                for split_a, split_b in all_pairs:
                    intersection = split_hashes[split_a] & split_hashes[split_b]
                    denom_a = max(split_sample_ns[split_a], 1)
                    denom_b = max(split_sample_ns[split_b], 1)
                    method = f"{split_methods[split_a]} vs {split_methods[split_b]}"
                    notes = []
                    if "sampled" in method:
                        notes.append("duplicate rates are over sampled hashes")
                    for split in (split_a, split_b):
                        if split in notes_by_split:
                            notes.append(notes_by_split[split])
                    rows.append(
                        {
                            "npz_path": rel,
                            "split_a": split_a,
                            "split_b": split_b,
                            "n_a": split_ns[split_a],
                            "n_b": split_ns[split_b],
                            "exact_duplicate_count": len(intersection),
                            "duplicate_rate_a": f"{len(intersection) / denom_a:.6f}",
                            "duplicate_rate_b": f"{len(intersection) / denom_b:.6f}",
                            "method": method,
                            "notes": "; ".join(notes),
                        }
                    )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "npz_path": rel,
                    "split_a": "",
                    "split_b": "",
                    "n_a": "",
                    "n_b": "",
                    "exact_duplicate_count": "",
                    "duplicate_rate_a": "",
                    "duplicate_rate_b": "",
                    "method": "error",
                    "notes": repr(exc),
                }
            )

    write_csv(
        out / "split_overlap_report.csv",
        [
            "npz_path",
            "split_a",
            "split_b",
            "n_a",
            "n_b",
            "exact_duplicate_count",
            "duplicate_rate_a",
            "duplicate_rate_b",
            "method",
            "notes",
        ],
        rows,
    )
    return rows


def collect_npz_key_index(root: Path) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    for npz_path in discover_all_local_npz_files(root):
        rel = relpath(npz_path, root)
        for meta in npz_header_metadata(npz_path):
            if meta.error:
                continue
            index[meta.key].append(rel)
    return index


def discover_scripts(root: Path) -> list[Path]:
    found: set[Path] = set()
    for pattern in KEY_USAGE_SCRIPT_PATTERNS:
        found.update(root.glob(pattern))
    return sorted(path for path in found if path.is_file())


def audit_npz_key_usage(root: Path, out: Path, key_index: dict[str, list[str]]) -> list[dict]:
    bracket_key_re = re.compile(r"\[\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]\s*\]")
    rows: list[dict] = []
    for script in discover_scripts(root):
        try:
            text = script.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "script": relpath(script, root),
                    "expected_key": "",
                    "context_line": "",
                    "actual_available_in_related_npz": "",
                    "status": "read_error",
                    "notes": repr(exc),
                }
            )
            continue
        if "np.load" not in text:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for match in bracket_key_re.finditer(line):
                key = match.group(1)
                if key not in INTERESTING_KEY_NAMES and key not in key_index:
                    continue
                available_paths = key_index.get(key, [])
                status = "found_in_local_npz" if available_paths else "missing_from_scanned_npz"
                if available_paths:
                    preview = "|".join(available_paths[:6])
                    if len(available_paths) > 6:
                        preview += f"|...(+{len(available_paths) - 6})"
                else:
                    preview = ""
                notes = ""
                if key in {"all_data", "all_labels"} and not available_paths:
                    notes = "potential contract drift: current wavelet exports use split keys rather than all_data/all_labels"
                rows.append(
                    {
                        "script": relpath(script, root),
                        "expected_key": key,
                        "context_line": f"{lineno}: {line.strip()}",
                        "actual_available_in_related_npz": preview,
                        "status": status,
                        "notes": notes,
                    }
                )

    write_csv(
        out / "npz_key_usage_report.csv",
        [
            "script",
            "expected_key",
            "context_line",
            "actual_available_in_related_npz",
            "status",
            "notes",
        ],
        rows,
    )
    return rows


def git_file_set(root: Path, args: list[str]) -> set[str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        return set()
    if result.returncode not in {0, 1}:
        return set()
    return {item.decode("utf-8", errors="replace") for item in result.stdout.split(b"\0") if item}


def asset_category(rel: str) -> str:
    first = rel.split("/", 1)[0]
    return first


def likely_role(path: Path, rel: str) -> str:
    lower = rel.lower()
    suffix = path.suffix.lower()
    if "shap" in lower and suffix in {".npz", ".npy"}:
        return "shap_cache"
    if suffix in {".pth", ".pt", ".ckpt"}:
        return "checkpoint_or_edge_tensor"
    if suffix == ".png":
        return "figure_or_training_curve"
    if suffix in {".csv", ".json", ".jsonl", ".log"}:
        return "metric_or_log_candidate"
    if rel.startswith("DataSets/"):
        return "raw_or_external_dataset"
    if rel.startswith("wavelet_dataset/"):
        return "preprocessed_wavelet_dataset"
    if rel.startswith("DAG/"):
        return "causal_dag_artifact"
    if rel.startswith("PC_Datasets/"):
        return "pc_dag_input_table"
    if suffix in {".npz", ".npy", ".mat"}:
        return "numeric_dataset_or_intermediate"
    return "documentation_or_other"


def recommendation_for(path: Path, rel: str, tracked: bool, ignored: bool) -> str:
    size = file_size_mb(path)
    role = likely_role(path, rel)
    suffix = path.suffix.lower()
    if size > 50 or suffix in {".npz", ".npy", ".pth", ".pt", ".mat", ".ckpt"}:
        return "do_not_version_large_asset" if ignored else "needs_ignore_before_git_add"
    if role in {"metric_or_log_candidate", "pc_dag_input_table"}:
        return "candidate_metric_source"
    if role == "figure_or_training_curve":
        return "candidate_paper_figure" if tracked else "convert_to_manifest"
    if tracked:
        return "keep_local_reference"
    return "unknown"


def audit_local_assets(root: Path, out: Path) -> list[dict]:
    rows: list[dict] = []
    tracked_files = git_file_set(root, ["ls-files", "-z"])
    ignored_files = git_file_set(root, ["ls-files", "--ignored", "--others", "--exclude-standard", "-z"])
    for asset_root in ASSET_ROOTS:
        base = root / asset_root
        if not base.exists():
            continue
        for path in sorted(p for p in base.rglob("*") if p.is_file()):
            if path.suffix.lower() not in ASSET_EXTENSIONS:
                continue
            rel = relpath(path, root)
            tracked = rel in tracked_files
            ignored = rel in ignored_files
            rows.append(
                {
                    "relative_path": rel,
                    "file_type": path.suffix.lower(),
                    "file_size_mb": f"{file_size_mb(path):.3f}",
                    "modified_time": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                    "category": asset_category(rel),
                    "likely_role": likely_role(path, rel),
                    "tracked_by_git": str(tracked),
                    "ignored_by_git": str(ignored),
                    "recommendation": recommendation_for(path, rel, tracked, ignored),
                }
            )

    write_csv(
        out / "local_research_assets.csv",
        [
            "relative_path",
            "file_type",
            "file_size_mb",
            "modified_time",
            "category",
            "likely_role",
            "tracked_by_git",
            "ignored_by_git",
            "recommendation",
        ],
        rows,
    )
    return rows


def extract_function_block(text: str, signature: str) -> str:
    start = text.find(signature)
    if start < 0:
        return ""
    lines = text[start:].splitlines()
    block = []
    for idx, line in enumerate(lines):
        if idx > 0 and line.startswith("class ") and not line.startswith("    "):
            break
        block.append(line)
    return "\n".join(block)


def line_hits(root: Path, path: Path, needle_re: str) -> list[str]:
    pattern = re.compile(needle_re)
    hits = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if pattern.search(line):
            hits.append(f"{relpath(path, root)}:{lineno}: {line.strip()}")
    return hits


def audit_gnn_risk(root: Path, out: Path) -> dict[str, str]:
    network_path = root / "Common" / "NetWorkFrame.py"
    causal_gnn_path = root / "GNNCausal" / "CausalGNN.py"
    network_text = network_path.read_text(encoding="utf-8", errors="replace")
    seu_block = extract_function_block(network_text, "class GNNCausalSEU")
    computes_out = bool(re.search(r"\bout\s*=", seu_block)) and "out = out + x" in seu_block
    classifier_input = "unknown"
    if re.search(r"self\.classifier\(\s*x\s*\)", seu_block):
        classifier_input = "x"
    if re.search(r"self\.classifier\(\s*out\s*\)", seu_block):
        classifier_input = "out"

    fit_transform_hits = line_hits(root, causal_gnn_path, r"\.fit_transform\(")
    transform_hits = line_hits(root, causal_gnn_path, r"\.transform\(")
    scaler_consistency = (
        "inconsistent_or_split_refit"
        if len(fit_transform_hits) > 1
        else "single_train_fit_observed"
    )

    lines = [
        "# GNN Risk Report",
        "",
        "## GNNCausalSEU.forward",
        "",
        f"- Computes causal propagation variables: {'yes' if computes_out else 'no'}",
        f"- Final classifier input detected: `{classifier_input}`",
    ]
    if classifier_input == "x":
        lines.append(
            "- Risk: current SEU GNN forward pass does not support a strong "
            "`causal-enhanced classification` claim, because the classifier is fed `x` rather than propagated `out`."
        )
    elif classifier_input == "out":
        lines.append("- Observation: classifier appears to use propagated `out`.")
    else:
        lines.append("- Observation: classifier input could not be determined by static regex.")

    lines.extend(
        [
            "",
            "## GNNCausal/CausalGNN.py scaler usage",
            "",
            f"- Scaler consistency assessment: `{scaler_consistency}`",
            "- `fit_transform` calls:",
            *[f"  - `{hit}`" for hit in fit_transform_hits],
            "- `transform` calls:",
            *[f"  - `{hit}`" for hit in transform_hits],
        ]
    )
    if len(fit_transform_hits) > 1:
        lines.append(
            "- Risk: finetune/cross branch appears to refit a scaler instead of using the train-fitted scaler consistently."
        )

    out_path = out / "gnn_risk_report.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "computes_out": str(computes_out),
        "classifier_input": classifier_input,
        "scaler_consistency": scaler_consistency,
        "fit_transform_count": str(len(fit_transform_hits)),
    }


def summarize_risks(
    out: Path,
    overlap_rows: list[dict],
    key_usage_rows: list[dict],
    asset_rows: list[dict],
    gnn_facts: dict[str, str],
) -> None:
    p0 = []
    p1 = []
    p2 = []

    overlap_hits = [
        row
        for row in overlap_rows
        if str(row.get("exact_duplicate_count", "")).isdigit() and int(row["exact_duplicate_count"]) > 0
    ]
    train_cross_finetune_hits = [
        row
        for row in overlap_hits
        if {row["split_a"], row["split_b"]} == {"train", "cross_finetune"}
    ]
    train_eval_hits = [
        row
        for row in overlap_hits
        if "train" in {row["split_a"], row["split_b"]}
        and ({row["split_a"], row["split_b"]} & {"test", "cross_test", "valid", "cross_finetune"})
    ]
    if train_cross_finetune_hits:
        p0.append(
            "Detected exact sample overlap between `train` and `cross_finetune` in at least one wavelet dataset; "
            "current cross-condition finetune evidence is not paper-grade."
        )
    if train_eval_hits:
        examples = "; ".join(
            f"{row['npz_path']} {row['split_a']}~{row['split_b']} dup={row['exact_duplicate_count']}"
            for row in train_eval_hits[:4]
        )
        p0.append(
            "Detected sampled exact overlap between `train` and an evaluation/cross split; "
            f"examples: {examples}. Treat affected results as non-paper-grade until strict split hashes are regenerated."
        )
    else:
        p1.append(
            "No sampled exact `train`/`cross_finetune` duplicate was detected, but large files may have been sampled; "
            "strict split regeneration is still recommended before paper experiments."
        )

    missing_keys = [row for row in key_usage_rows if row["status"] == "missing_from_scanned_npz"]
    if any(row["expected_key"] in {"all_data", "all_labels"} for row in missing_keys):
        p0.append(
            "`Common/PCDataset(wavelet).py`-style `all_data/all_labels` expectations are not found in local scanned npz files; "
            "SHAP-to-PC data contract needs confirmation."
        )

    if gnn_facts.get("classifier_input") == "x":
        p0.append(
            "`GNNCausalSEU.forward` computes propagated `out` but feeds `x` into the classifier; do not use SEU GNN as core causal-enhancement evidence."
        )
    if gnn_facts.get("scaler_consistency") == "inconsistent_or_split_refit":
        p0.append(
            "`GNNCausal/CausalGNN.py` contains multiple `fit_transform` calls, including finetune refit; scaler provenance is inconsistent."
        )

    large_local = [
        row
        for row in asset_rows
        if float(row["file_size_mb"]) > 50 and row["tracked_by_git"] == "False"
    ]
    if large_local:
        p1.append(
            f"{len(large_local)} large local research assets are not versioned; keep them out of Git but record hash/config/manifest before using as evidence."
        )

    if missing_keys:
        p1.append(
            f"{len(missing_keys)} npz key usages were not matched to scanned local npz keys; inspect `npz_key_usage_report.csv` before reruns."
        )

    p2.append("Reports are metadata only; they do not replace config snapshots, metrics JSON/CSV, checkpoint manifests, or git-hash capture.")
    p2.append("Asset manifest uses file metadata and simple role heuristics; paper-grade provenance still needs explicit experiment manifests.")

    lines = ["# Reproducibility Audit Risk Summary", ""]
    for title, items in (("P0", p0), ("P1", p1), ("P2", p2)):
        lines.append(f"## {title}")
        lines.append("")
        if items:
            lines.extend(f"- {item}" for item in items)
        else:
            lines.append("- No issue detected by this lightweight audit.")
        lines.append("")

    lines.extend(
        [
            "## Suggested Next Minimal Engineering Tasks",
            "",
            "- Add a strict no-overlap split generator and persist split indices/hash summaries.",
            "- Add a small experiment manifest writer for seed, config, metrics, checkpoint, git hash, and source data hash.",
            "- Fix or quarantine SEU GNN causal propagation before any causal-enhancement claim.",
            "- Normalize SHAP-to-PC input/output contracts before rerunning attribution-causality experiments.",
        ]
    )
    (out / "risk_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument("--out", default="reports/reproducibility_audit", help="Output report directory")
    parser.add_argument(
        "--max-nan-elements",
        type=int,
        default=1_000_000,
        help="Maximum numeric array elements to materialize for NaN/label checks",
    )
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=0,
        help="0 means full split unless the npz exceeds --auto-sample-large-mb",
    )
    parser.add_argument(
        "--auto-sample-large-mb",
        type=float,
        default=1024.0,
        help="Auto-sample this many samples per split for wavelet npz files above this size",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out = (root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    audit_npz_contracts(root, out, args.max_nan_elements)
    overlap_rows = audit_split_overlap(root, out, args.max_samples_per_split, args.auto_sample_large_mb)
    key_index = collect_npz_key_index(root)
    key_usage_rows = audit_npz_key_usage(root, out, key_index)
    asset_rows = audit_local_assets(root, out)
    gnn_facts = audit_gnn_risk(root, out)
    summarize_risks(out, overlap_rows, key_usage_rows, asset_rows, gnn_facts)

    print(f"Wrote reproducibility audit reports to {out.relative_to(root)}")
    print(f"NPZ key index: {len(key_index)} unique keys")
    print(f"Split overlap rows: {len(overlap_rows)}")
    print(f"Asset rows: {len(asset_rows)}")
    print(f"GNN classifier input: {gnn_facts.get('classifier_input')}")
    print(f"Scaler assessment: {gnn_facts.get('scaler_consistency')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
