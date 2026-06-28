#!/usr/bin/env python3
"""Diagnose checkpoint/data width mismatch for no-attention baseline models."""

from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch


CONDITIONS = ("Bearing_20_0", "Bearing_30_2", "Gear_20_0", "Gear_30_2")


def relpath(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def npz_header(path: Path) -> dict[str, tuple[tuple[int, ...], str]]:
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


def checkpoint_expected_width(path: Path) -> tuple[int | None, str]:
    state = torch.load(path, map_location="cpu")
    weight = state.get("feature_extractor.5.weight")
    conv = state.get("feature_extractor.0.weight")
    if weight is None or conv is None:
        return None, "missing expected no-attention keys"
    in_features = int(weight.shape[1])
    out_channels = int(conv.shape[0])
    denom = out_channels * 6
    if in_features % denom:
        return None, f"in_features {in_features} not divisible by {denom}"
    return (in_features // denom) * 4, f"conv_out={out_channels}; linear_in={in_features}; inferred_width={(in_features // denom) * 4}"


def condition_from_checkpoint(path: Path) -> str:
    return path.stem if path.stem in CONDITIONS else "unknown"


def seed_from_checkpoint(path: Path) -> str:
    for part in path.parts:
        if part.startswith("Seed_"):
            return part.split("_", 1)[1]
    return "unknown"


def file_hits(root: Path, patterns: list[str], regex: str) -> list[str]:
    hits = []
    compiled = re.compile(regex)
    for pattern in patterns:
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception:
                continue
            for lineno, line in enumerate(lines, start=1):
                if compiled.search(line):
                    hits.append(f"{relpath(path, root)}:{lineno}: {line.strip()}")
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    dataset_dir = (root / args.dataset_dir).resolve()
    model_dir = (root / args.model_dir).resolve()
    out = (root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out).resolve()

    rows = []
    for ckpt in sorted(model_dir.glob("Seed_*/*.pth")):
        condition = condition_from_checkpoint(ckpt)
        seed = seed_from_checkpoint(ckpt)
        data_path = dataset_dir / f"{condition}.npz"
        expected_width, ckpt_note = checkpoint_expected_width(ckpt)
        dataset_width = ""
        dataset_shape = ""
        dataset_status = "missing"
        if data_path.exists():
            header = npz_header(data_path)
            if "test_set" in header:
                shape = header["test_set"][0]
                dataset_shape = "x".join(str(x) for x in shape)
                dataset_width = shape[-1]
                dataset_status = "ok"
        rows.append(
            {
                "checkpoint_path": relpath(ckpt, root),
                "condition": condition,
                "seed": seed,
                "checkpoint_expected_width": expected_width or "",
                "checkpoint_note": ckpt_note,
                "dataset_path": relpath(data_path, root) if data_path.exists() else data_path.as_posix(),
                "dataset_test_shape": dataset_shape,
                "dataset_width": dataset_width,
                "dataset_status": dataset_status,
                "width_match": str(bool(expected_width and dataset_width == expected_width)),
                "safe_crop_512_supported_by_code": "False",
                "notes": "Checkpoint is architecture-compatible with width 512; current dataset width is 1024.",
            }
        )

    shap_rows = []
    for shap in sorted((root / "CalculateShapValues" / "NoAttention").glob("Seed_*/*.npz")):
        header = npz_header(shap)
        if "shap_values" not in header:
            continue
        shape = header["shap_values"][0]
        shap_rows.append(
            {
                "source": relpath(shap, root),
                "condition": shap.stem,
                "seed": seed_from_checkpoint(shap),
                "shap_shape": "x".join(str(x) for x in shape),
                "shap_width": shape[3] if len(shape) > 3 else "",
            }
        )

    script_hits = {
        "explicit_512_or_crop_hits": file_hits(
            root,
            [
                "ModelTrain/NoAttention/**/*.py",
                "CalculateShapValues/NoAttention/**/*.py",
                "Common/NetWorkFrame.py",
                "Common/ModelTrainAndVisiable.py",
            ],
            r"(:512|0:512|512)",
        ),
        "legacy_key_hits": file_hits(
            root,
            ["ModelTrain/NoAttention/model_train_6ch/*.py", "CalculateShapValues/NoAttention/*.py"],
            r"(train_data|test_data|class_names|bearing_20_0\.npz|gear_20_0\.npz)",
        ),
    }

    write_csv(
        out / "checkpoint_dataset_mismatch_report.csv",
        [
            "checkpoint_path",
            "condition",
            "seed",
            "checkpoint_expected_width",
            "checkpoint_note",
            "dataset_path",
            "dataset_test_shape",
            "dataset_width",
            "dataset_status",
            "width_match",
            "safe_crop_512_supported_by_code",
            "notes",
        ],
        rows,
    )

    write_csv(out / "shap_width_inventory.csv", ["source", "condition", "seed", "shap_shape", "shap_width"], shap_rows)

    width_counts = Counter(str(row["checkpoint_expected_width"]) for row in rows)
    dataset_width_counts = Counter(str(row["dataset_width"]) for row in rows)
    shap_width_counts = Counter(str(row["shap_width"]) for row in shap_rows)
    notes = [
        "# Checkpoint Dataset Mismatch Notes",
        "",
        "## Findings",
        "",
        f"- Checkpoints inspected: {len(rows)}.",
        f"- Checkpoint expected widths: {dict(width_counts)}.",
        f"- Current dataset test widths: {dict(dataset_width_counts)}.",
        f"- SHAP cache widths: {dict(shap_width_counts)}.",
        "- Current `wavelet_dataset/*.npz` files contain `train_set/valid_set/test_set` arrays with width 1024.",
        "- Existing NoAttention SHAP caches have shape `(500, 6, 6, 512, 5)`, so they were produced from width-512 model inputs.",
        "- Legacy training/SHAP scripts reference lowercase dataset names and keys such as `train_data/test_data/class_names`, which are not present in the current uppercase `wavelet_dataset/*.npz` contract.",
        "",
        "## Safety Judgment",
        "",
        "- It is technically possible to run the saved checkpoints on `current_split[..., :512]` because the checkpoint architecture expects width 512.",
        "- However, this repository does not currently contain the original lowercase 512-width `.npz` files, nor explicit training-code evidence that the old data was exactly the first 512 points of the current 1024-point arrays.",
        "- Therefore `--crop-width 512` should be treated as a compatibility/smoke evaluation, not as paper-grade baseline metrics, unless human researcher confirms the old preprocessing contract.",
        "",
        "## Script Evidence",
        "",
        "### Explicit 512 / crop-related hits",
        *[f"- `{hit}`" for hit in script_hits["explicit_512_or_crop_hits"][:80]],
        "",
        "### Legacy dataset key/path hits",
        *[f"- `{hit}`" for hit in script_hits["legacy_key_hits"][:80]],
    ]
    (out / "checkpoint_dataset_mismatch_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(f"Wrote mismatch report to {out}")
    print(f"checkpoint_widths={dict(width_counts)} dataset_widths={dict(dataset_width_counts)} shap_widths={dict(shap_width_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
