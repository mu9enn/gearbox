#!/usr/bin/env python3
"""Full exact-sample overlap check for wavelet dataset npz splits."""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from npz_stream_utils import file_size_mb, hash_split, npz_members, relpath, split_keys, write_csv


def check_file(npz_path: Path, root: Path, algorithm: str) -> list[dict]:
    rows: list[dict] = []
    try:
        members = npz_members(npz_path)
        keys = split_keys(members)
    except Exception as exc:  # noqa: BLE001
        return [
            {
                "npz_path": relpath(npz_path, root),
                "split_a": "",
                "split_b": "",
                "n_a": "",
                "n_b": "",
                "exact_duplicate_count": "",
                "duplicate_rate_a": "",
                "duplicate_rate_b": "",
                "method": f"streaming_{algorithm}",
                "status": "error",
                "notes": repr(exc),
            }
        ]

    split_hashes: dict[str, set[str]] = {}
    split_stats: dict[str, dict[str, int]] = {}
    split_errors: dict[str, str] = {}
    for key in keys:
        try:
            hashes, stats = hash_split(npz_path, key, algorithm=algorithm)
            split_hashes[key] = hashes
            split_stats[key] = stats
        except Exception as exc:  # noqa: BLE001
            split_errors[key] = repr(exc)

    if len(split_hashes) < 2:
        rows.append(
            {
                "npz_path": relpath(npz_path, root),
                "split_a": "",
                "split_b": "",
                "n_a": "",
                "n_b": "",
                "exact_duplicate_count": "",
                "duplicate_rate_a": "",
                "duplicate_rate_b": "",
                "method": f"streaming_{algorithm}",
                "status": "error",
                "notes": f"fewer_than_two_splits; keys={keys}; errors={split_errors}",
            }
        )
        return rows

    for split_a, split_b in combinations(keys, 2):
        if split_a not in split_hashes or split_b not in split_hashes:
            rows.append(
                {
                    "npz_path": relpath(npz_path, root),
                    "split_a": split_a,
                    "split_b": split_b,
                    "n_a": members.get(split_a).n_samples if split_a in members else "",
                    "n_b": members.get(split_b).n_samples if split_b in members else "",
                    "exact_duplicate_count": "",
                    "duplicate_rate_a": "",
                    "duplicate_rate_b": "",
                    "method": f"streaming_{algorithm}",
                    "status": "error",
                    "notes": f"{split_a}:{split_errors.get(split_a, '')}; {split_b}:{split_errors.get(split_b, '')}",
                }
            )
            continue
        duplicates = split_hashes[split_a] & split_hashes[split_b]
        n_a = split_stats[split_a]["n_samples"]
        n_b = split_stats[split_b]["n_samples"]
        notes = []
        if split_stats[split_a]["internal_duplicate_count"]:
            notes.append(f"{split_a}_internal_duplicates={split_stats[split_a]['internal_duplicate_count']}")
        if split_stats[split_b]["internal_duplicate_count"]:
            notes.append(f"{split_b}_internal_duplicates={split_stats[split_b]['internal_duplicate_count']}")
        rows.append(
            {
                "npz_path": relpath(npz_path, root),
                "split_a": split_a,
                "split_b": split_b,
                "n_a": n_a,
                "n_b": n_b,
                "exact_duplicate_count": len(duplicates),
                "duplicate_rate_a": f"{len(duplicates) / max(n_a, 1):.8f}",
                "duplicate_rate_b": f"{len(duplicates) / max(n_b, 1):.8f}",
                "method": f"streaming_{algorithm}",
                "status": "ok",
                "notes": "; ".join(notes),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument("--dataset-dir", default="wavelet_dataset", help="Directory containing dataset npz files")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--file", action="append", help="Optional specific npz file(s) to check")
    parser.add_argument("--algorithm", default="sha1", choices=["sha1", "sha256"], help="Hash algorithm")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    dataset_dir = (root / args.dataset_dir).resolve()
    if args.file:
        files = [(root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve() for value in args.file]
    else:
        files = sorted(dataset_dir.glob("*.npz"))

    rows: list[dict] = []
    for npz_path in files:
        print(f"Checking {relpath(npz_path, root)} ({file_size_mb(npz_path):.1f} MB)", flush=True)
        rows.extend(check_file(npz_path, root, args.algorithm))

    out = (root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out).resolve()
    write_csv(
        out,
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
            "status",
            "notes",
        ],
        rows,
    )
    duplicate_rows = [row for row in rows if str(row["exact_duplicate_count"]).isdigit() and int(row["exact_duplicate_count"]) > 0]
    print(f"Wrote {len(rows)} rows to {relpath(out, root)}")
    print(f"Rows with exact duplicates: {len(duplicate_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
