#!/usr/bin/env python3
"""Build or plan leak-free SEU wavelet datasets without overwriting history."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


CONDITION_SPECS = {
    "Bearing_20_0": {
        "subset": "bearing_set",
        "rpm": 1200,
        "labels": ["ball_20_0", "comb_20_0", "health_20_0", "inner_20_0", "outer_20_0"],
    },
    "Bearing_30_2": {
        "subset": "bearing_set",
        "rpm": 1800,
        "labels": ["ball_30_2", "comb_30_2", "health_30_2", "inner_30_2", "outer_30_2"],
    },
    "Gear_20_0": {
        "subset": "gear_set",
        "rpm": 1200,
        "labels": ["chipped_20_0", "Health_20_0", "miss_20_0", "root_20_0", "surface_20_0"],
    },
    "Gear_30_2": {
        "subset": "gear_set",
        "rpm": 1800,
        "labels": ["chipped_30_2", "Health_30_2", "miss_30_2", "root_30_2", "surface_30_2"],
    },
}

SPLIT_RATIOS = {
    "train": 0.70,
    "valid": 0.10,
    "test": 0.10,
    "cross_finetune": 0.05,
    "cross_test": 0.05,
}

OUTPUT_KEYS = [
    "train_set",
    "train_label",
    "valid_set",
    "valid_label",
    "test_set",
    "test_label",
    "cross_finetune_set",
    "cross_finetune_label",
    "cross_test_set",
    "cross_test_label",
    "label_mapping",
]


def git_commit(root: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_data(file_path: Path, skip_rows: int = 16, use_cols: int = 8) -> np.ndarray:
    import pandas as pd

    data = pd.read_csv(
        file_path,
        skiprows=skip_rows,
        sep=r"\s+|,",
        engine="python",
        usecols=range(use_cols),
    )
    return data.values


def angle_resample(vib_signal: np.ndarray, fs: int = 5120, rpm: int | None = None, angle_fs: int = 256) -> np.ndarray:
    from scipy.interpolate import interp1d

    t = np.arange(len(vib_signal)) / fs
    del t
    freq_hz = np.full(shape=len(vib_signal), fill_value=rpm, dtype=np.float64) / 60.0
    angle = np.cumsum(freq_hz) / fs * 360
    angle_new = np.linspace(angle[0], angle[-1], int(len(angle) * angle_fs / fs * freq_hz[0]))
    f_interp = interp1d(angle, vib_signal, kind="cubic", fill_value="extrapolate")
    return f_interp(angle_new)


def window_data(signal: np.ndarray, segment_length: int, step: int) -> np.ndarray:
    windows = []
    for start in range(0, len(signal) - segment_length + 1, step):
        windows.append(signal[start : start + segment_length])
    return np.asarray(windows)


def wavelet_function(data: np.ndarray, wavelet_type: str, layers: int) -> np.ndarray:
    import pywt

    wave_coef = []
    for single_data in data:
        single_ch = []
        for i in range(6):
            wt_coef = pywt.swt(data=single_data[:, i], wavelet=wavelet_type, level=layers)
            coef = []
            for idx, (u, v) in enumerate(wt_coef):
                if idx == 0:
                    coef.append(u)
                    coef.append(v)
                else:
                    coef.append(v)
            single_ch.append(coef)
        wave_coef.append(single_ch)
    return np.asarray(wave_coef)


def raw_split_bounds(n_rows: int, guard_rows: int) -> dict[str, tuple[int, int]]:
    usable = n_rows - guard_rows * (len(SPLIT_RATIOS) - 1)
    if usable <= 0:
        raise ValueError(f"source sequence too short for guard_rows={guard_rows}: n_rows={n_rows}")
    lengths = {name: int(usable * ratio) for name, ratio in SPLIT_RATIOS.items()}
    lengths["cross_test"] = usable - sum(lengths[name] for name in lengths if name != "cross_test")
    bounds: dict[str, tuple[int, int]] = {}
    cursor = 0
    for idx, name in enumerate(SPLIT_RATIOS):
        start = cursor
        end = start + lengths[name]
        bounds[name] = (start, end)
        cursor = end + (guard_rows if idx < len(SPLIT_RATIOS) - 1 else 0)
    return bounds


def process_raw_split(raw_signal: np.ndarray, rpm: int, segment_length: int, step: int, wavelet_type: str, layers: int) -> np.ndarray:
    selected = raw_signal[:, [1, 2, 3, 5, 6, 7]]
    angle_channels = [angle_resample(selected[:, idx], rpm=rpm) for idx in range(6)]
    angle_signal = np.asarray(angle_channels).T
    windows = window_data(angle_signal, segment_length=segment_length, step=step)
    return wavelet_function(windows, wavelet_type=wavelet_type, layers=layers)


def source_files(source: Path) -> list[Path]:
    return sorted(source.glob("*_set/*.csv"))


def source_metadata(root: Path, source: Path, hash_files: bool) -> dict:
    metadata = {}
    for path in source_files(source):
        key = path.resolve().relative_to(root.resolve()).as_posix()
        item = {"size_bytes": path.stat().st_size}
        item["sha256"] = sha256_file(path) if hash_files else None
        metadata[key] = item
    return metadata


def planned_split_summary(source: Path, segment_length: int, guard_rows: int) -> dict:
    summary = {}
    for condition, spec in CONDITION_SPECS.items():
        condition_summary = {}
        for label in spec["labels"]:
            path = source / spec["subset"] / f"{label}.csv"
            if not path.exists():
                condition_summary[label] = {"status": "missing", "path": str(path)}
                continue
            # Count rows cheaply while accounting for the 16 skipped header rows.
            with path.open("rb") as f:
                total_lines = sum(1 for _ in f)
            data_rows = max(total_lines - 16, 0)
            bounds = raw_split_bounds(data_rows, guard_rows=guard_rows)
            condition_summary[label] = {
                "status": "planned",
                "source_rows_after_skip": data_rows,
                "raw_split_bounds": bounds,
                "minimum_window_length": segment_length,
                "guard_rows_between_splits": guard_rows,
            }
        summary[condition] = condition_summary
    return summary


def build_condition(
    source: Path,
    out_dir: Path,
    condition: str,
    spec: dict,
    segment_length: int,
    step: int,
    guard_rows: int,
    wavelet_type: str,
    layers: int,
) -> dict:
    label_to_id = {label: idx for idx, label in enumerate(spec["labels"])}
    split_arrays = {name: [] for name in SPLIT_RATIOS}
    split_labels = {name: [] for name in SPLIT_RATIOS}

    for label in spec["labels"]:
        path = source / spec["subset"] / f"{label}.csv"
        raw = load_data(path)
        bounds = raw_split_bounds(len(raw), guard_rows=guard_rows)
        for split_name, (start, end) in bounds.items():
            coef = process_raw_split(
                raw[start:end],
                rpm=spec["rpm"],
                segment_length=segment_length,
                step=step,
                wavelet_type=wavelet_type,
                layers=layers,
            )
            split_arrays[split_name].append(coef)
            split_labels[split_name].extend([label] * coef.shape[0])

    output = {}
    for split_name in SPLIT_RATIOS:
        output[f"{split_name}_set"] = np.reshape(split_arrays[split_name], (-1, 6, 6, segment_length))
        output[f"{split_name}_label"] = np.asarray([label_to_id[label] for label in split_labels[split_name]], dtype=np.int64)
    output["label_mapping"] = np.asarray(list(zip(spec["labels"], range(len(spec["labels"])))), dtype=str)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{condition}.npz"
    np.savez(out_path, **output)
    return {
        "path": out_path.as_posix(),
        "keys": sorted(output.keys()),
        "shapes": {key: list(value.shape) for key, value in output.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument("--source", required=True, help="SEU source directory")
    parser.add_argument("--out", required=True, help="Output directory for leak-free npz files")
    parser.add_argument("--manifest", required=True, help="Manifest JSON path")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--segment-length", type=int, default=1024)
    parser.add_argument("--step", type=int, default=1024, help="Non-overlapping window step for leak-free datasets")
    parser.add_argument("--guard-rows", type=int, default=5120, help="Raw-row guard gap between split blocks")
    parser.add_argument("--wavelet-type", default="db4")
    parser.add_argument("--wavelet-layers", type=int, default=5)
    parser.add_argument("--hash-source-files", action="store_true", help="Compute sha256 for source CSV files")
    parser.add_argument("--dry-run", action="store_true", help="Write manifest only; do not create npz files")
    args = parser.parse_args()

    np.random.seed(args.seed)
    root = Path(args.root).resolve()
    source = (root / args.source).resolve() if not Path(args.source).is_absolute() else Path(args.source).resolve()
    out_dir = (root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out).resolve()
    manifest_path = (root / args.manifest).resolve() if not Path(args.manifest).is_absolute() else Path(args.manifest).resolve()

    planned_outputs = {condition: (out_dir / f"{condition}.npz").resolve().relative_to(root).as_posix() for condition in CONDITION_SPECS}
    manifest = {
        "git_commit": git_commit(root),
        "script": Path(__file__).resolve().relative_to(root).as_posix(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "source": source.relative_to(root).as_posix() if source.is_relative_to(root) else source.as_posix(),
        "source_files": [path.relative_to(root).as_posix() for path in source_files(source)],
        "source_file_sizes": {
            path.relative_to(root).as_posix(): path.stat().st_size for path in source_files(source)
        },
        "source_file_hashes_if_feasible": source_metadata(root, source, args.hash_source_files),
        "seed": args.seed,
        "split_policy": {
            "name": "raw_contiguous_blocks_before_resample_window_wavelet",
            "ratios": SPLIT_RATIOS,
            "guard_rows_between_splits": args.guard_rows,
            "window_step": args.step,
            "no_cross_test_reuse_of_train_valid_test_rows": True,
            "cross_finetune_policy": "dedicated raw block disjoint from train/valid/test/cross_test",
        },
        "wavelet_type": args.wavelet_type,
        "wavelet_layers": args.wavelet_layers,
        "segment_length": args.segment_length,
        "resample_policy": {
            "method": "angle_resample",
            "fs": 5120,
            "angle_fs": 256,
            "rpm_by_condition": {condition: spec["rpm"] for condition, spec in CONDITION_SPECS.items()},
        },
        "normalization_policy": "none_in_dataset_builder",
        "output_files": planned_outputs,
        "output_keys": OUTPUT_KEYS,
        "output_shapes": {},
        "split_hash_summary": "not_computed_in_builder; run tools/validate_npz_contract.py after actual build",
        "planned_raw_splits": planned_split_summary(source, args.segment_length, args.guard_rows),
        "known_limitations": [
            "Each fault-condition CSV appears to be one continuous sequence; this builder prevents exact row reuse via raw contiguous split blocks but cannot create independent source runs where none exist.",
            "Dry-run does not generate npz files or compute split hashes.",
            "Full source sha256 is only computed when --hash-source-files is provided.",
        ],
    }

    if not args.dry_run:
        output_files = {}
        for condition, spec in CONDITION_SPECS.items():
            print(f"Building {condition}", flush=True)
            result = build_condition(
                source=source,
                out_dir=out_dir,
                condition=condition,
                spec=spec,
                segment_length=args.segment_length,
                step=args.step,
                guard_rows=args.guard_rows,
                wavelet_type=args.wavelet_type,
                layers=args.wavelet_layers,
            )
            result["path"] = str(Path(result["path"]).resolve().relative_to(root))
            output_files[condition] = result
        manifest["output_shapes"] = {condition: result["shapes"] for condition, result in output_files.items()}
        manifest["output_files"] = {condition: result["path"] for condition, result in output_files.items()}

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote manifest to {manifest_path.relative_to(root)}")
    if args.dry_run:
        print("Dry run only; no npz files were generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
