#!/usr/bin/env python3
"""Validate gearbox wavelet npz dataset contracts."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from npz_stream_utils import (  # noqa: E402
    file_size_mb,
    hash_sample_bytes,
    iter_npy_sample_bytes,
    label_key_for,
    load_label_distribution,
    npz_members,
    relpath,
    split_keys,
    write_csv,
)


REQUIRED_SPLITS = ("train_set", "valid_set", "test_set", "cross_test_set")
OPTIONAL_SPLITS = ("cross_finetune_set", "cross_set")
EXPECTED_TAIL_SHAPE = (6, 6, 1024)


def inspect_split(npz_path: Path, split_key: str, members: dict, algorithm: str) -> dict:
    member = members[split_key]
    hashes: set[str] = set()
    counter: Counter[str] = Counter()
    nan_count = 0
    checked_nan = True
    for sample_member, sample_bytes in iter_npy_sample_bytes(npz_path, split_key):
        digest = hash_sample_bytes(sample_member, sample_bytes, algorithm=algorithm)
        counter[digest] += 1
        hashes.add(digest)
        if np.issubdtype(sample_member.dtype, np.floating):
            sample = np.frombuffer(sample_bytes, dtype=sample_member.dtype)
            nan_count += int(np.isnan(sample).sum())
        elif np.issubdtype(sample_member.dtype, np.number):
            nan_count += 0
        else:
            checked_nan = False
    internal_duplicates = sum(count - 1 for count in counter.values() if count > 1)
    return {
        "hashes": hashes,
        "nan_count": nan_count if checked_nan else "",
        "internal_duplicate_count": internal_duplicates,
        "shape_ok": tuple(member.shape[1:]) == EXPECTED_TAIL_SHAPE,
        "dtype": str(member.dtype),
        "n_samples": member.n_samples,
    }


def validate_file(npz_path: Path, root: Path, algorithm: str) -> list[dict]:
    rows: list[dict] = []
    rel = relpath(npz_path, root)
    try:
        members = npz_members(npz_path)
    except Exception as exc:  # noqa: BLE001
        return [
            {
                "npz_path": rel,
                "check_type": "file",
                "split": "",
                "split_b": "",
                "status": "fail",
                "n_samples": "",
                "shape": "",
                "dtype": "",
                "label_key": "",
                "label_count": "",
                "label_distribution": "",
                "nan_count": "",
                "internal_duplicate_count": "",
                "exact_duplicate_count": "",
                "notes": repr(exc),
            }
        ]

    observed_splits = split_keys(members)
    missing_required = [key for key in REQUIRED_SPLITS if key not in members]
    rows.append(
        {
            "npz_path": rel,
            "check_type": "file",
            "split": "",
            "split_b": "",
            "status": "pass" if not missing_required else "fail",
            "n_samples": "",
            "shape": "",
            "dtype": "",
            "label_key": "",
            "label_count": "",
            "label_distribution": "",
            "nan_count": "",
            "internal_duplicate_count": "",
            "exact_duplicate_count": "",
            "notes": f"keys={sorted(members)}; missing_required={missing_required}",
        }
    )

    split_results: dict[str, dict] = {}
    for split_key in observed_splits:
        label_key = label_key_for(split_key)
        member = members[split_key]
        label_member = members.get(label_key)
        label_count = label_member.n_samples if label_member else ""
        split_status = "pass"
        notes = []
        if split_key in REQUIRED_SPLITS + OPTIONAL_SPLITS and tuple(member.shape[1:]) != EXPECTED_TAIL_SHAPE:
            split_status = "fail"
            notes.append(f"expected_tail_shape={EXPECTED_TAIL_SHAPE}")
        if not label_member:
            split_status = "fail"
            notes.append(f"missing_label_key={label_key}")
        elif label_member.n_samples != member.n_samples:
            split_status = "fail"
            notes.append(f"label_count_mismatch={label_member.n_samples}")
        try:
            result = inspect_split(npz_path, split_key, members, algorithm=algorithm)
            split_results[split_key] = result
            if result["nan_count"] not in {"", 0}:
                split_status = "fail"
                notes.append(f"nan_count={result['nan_count']}")
            if result["internal_duplicate_count"]:
                split_status = "warn"
                notes.append(f"internal_duplicate_count={result['internal_duplicate_count']}")
        except Exception as exc:  # noqa: BLE001
            split_status = "fail"
            result = {
                "nan_count": "",
                "internal_duplicate_count": "",
                "n_samples": member.n_samples,
                "shape_ok": False,
                "dtype": str(member.dtype),
            }
            notes.append(repr(exc))

        rows.append(
            {
                "npz_path": rel,
                "check_type": "split",
                "split": split_key,
                "split_b": "",
                "status": split_status,
                "n_samples": member.n_samples,
                "shape": "x".join(str(part) for part in member.shape),
                "dtype": str(member.dtype),
                "label_key": label_key,
                "label_count": label_count,
                "label_distribution": load_label_distribution(npz_path, label_key) if label_member else "",
                "nan_count": result["nan_count"],
                "internal_duplicate_count": result["internal_duplicate_count"],
                "exact_duplicate_count": "",
                "notes": "; ".join(notes),
            }
        )

    for split_a, split_b in combinations(observed_splits, 2):
        if split_a not in split_results or split_b not in split_results:
            continue
        duplicates = split_results[split_a]["hashes"] & split_results[split_b]["hashes"]
        n_a = split_results[split_a]["n_samples"]
        n_b = split_results[split_b]["n_samples"]
        rows.append(
            {
                "npz_path": rel,
                "check_type": "between_split_duplicate",
                "split": split_a,
                "split_b": split_b,
                "status": "fail" if duplicates else "pass",
                "n_samples": f"{n_a}|{n_b}",
                "shape": "",
                "dtype": "",
                "label_key": "",
                "label_count": "",
                "label_distribution": "",
                "nan_count": "",
                "internal_duplicate_count": "",
                "exact_duplicate_count": len(duplicates),
                "notes": f"duplicate_rate_a={len(duplicates) / max(n_a, 1):.8f}; duplicate_rate_b={len(duplicates) / max(n_b, 1):.8f}",
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, help="Directory containing dataset npz files")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--root", default=".", help="Repository root for relative paths")
    parser.add_argument("--algorithm", default="sha1", choices=["sha1", "sha256"], help="Hash algorithm")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    dataset_dir = (root / args.dataset_dir).resolve() if not Path(args.dataset_dir).is_absolute() else Path(args.dataset_dir)
    files = sorted(dataset_dir.glob("*.npz")) if dataset_dir.exists() else []
    rows: list[dict] = []
    if not files:
        rows.append(
            {
                "npz_path": str(dataset_dir),
                "check_type": "dataset_dir",
                "split": "",
                "split_b": "",
                "status": "fail",
                "n_samples": "",
                "shape": "",
                "dtype": "",
                "label_key": "",
                "label_count": "",
                "label_distribution": "",
                "nan_count": "",
                "internal_duplicate_count": "",
                "exact_duplicate_count": "",
                "notes": "no_npz_files_found",
            }
        )
    for npz_path in files:
        print(f"Validating {relpath(npz_path, root)} ({file_size_mb(npz_path):.1f} MB)", flush=True)
        rows.extend(validate_file(npz_path, root, args.algorithm))

    out = (root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out).resolve()
    write_csv(
        out,
        [
            "npz_path",
            "check_type",
            "split",
            "split_b",
            "status",
            "n_samples",
            "shape",
            "dtype",
            "label_key",
            "label_count",
            "label_distribution",
            "nan_count",
            "internal_duplicate_count",
            "exact_duplicate_count",
            "notes",
        ],
        rows,
    )
    failures = [row for row in rows if row["status"] == "fail"]
    warnings = [row for row in rows if row["status"] == "warn"]
    print(f"Wrote {len(rows)} rows to {relpath(out, root)}")
    print(f"Validation failures: {len(failures)}; warnings: {len(warnings)}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
