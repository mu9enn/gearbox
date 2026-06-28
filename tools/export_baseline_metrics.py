#!/usr/bin/env python3
"""Export lightweight baseline metrics from existing checkpoints when compatible."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score


CONDITIONS = ("Bearing_20_0", "Bearing_30_2", "Gear_20_0", "Gear_30_2")
HISTORICAL_WIDTH_NOTE = (
    "old checkpoint/SHAP used 512-width input; current wavelet data are 1024-width; "
    "human researcher confirmed using first 512 points via [..., :512] to recover historical evaluation protocol"
)


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


def expected_width_from_state_dict(state: dict) -> int | None:
    weight = state.get("feature_extractor.5.weight")
    conv_out = state.get("feature_extractor.0.weight")
    if weight is None or conv_out is None:
        return None
    in_features = int(weight.shape[1])
    out_channels = int(conv_out.shape[0])
    # CNNNetWorkNoAttention: Conv2d -> MaxPool2d(kernel=(1,4), stride=(1,4)) -> Flatten.
    denominator = out_channels * 6
    if in_features % denominator:
        return None
    pooled_width = in_features // denominator
    return pooled_width * 4


def condition_from_checkpoint(path: Path) -> str:
    stem = path.stem
    return stem if stem in CONDITIONS else "unknown"


def seed_from_checkpoint(path: Path) -> str:
    for part in path.parts:
        if part.startswith("Seed_"):
            return part.split("_", 1)[1]
    return "unknown"


def dataset_path(dataset_dir: Path, condition: str) -> Path:
    return dataset_dir / f"{condition}.npz"


def split_keys(split: str) -> tuple[str, str]:
    return f"{split}_set", f"{split}_label"


def evaluate_checkpoint(
    root: Path,
    checkpoint: Path,
    data_path: Path,
    split: str,
    batch_size: int,
    device: torch.device,
    crop_width: int | None,
):
    sys.path.insert(0, str(root))
    from Common.NetWorkFrame import CNNNetWorkNoAttention

    state = torch.load(checkpoint, map_location="cpu")
    expected_width = expected_width_from_state_dict(state)
    headers = npz_header(data_path)
    data_key, label_key = split_keys(split)
    if data_key not in headers or label_key not in headers:
        raise ValueError(f"missing keys {data_key}/{label_key}; available={sorted(headers)}")
    data_shape = headers[data_key][0]
    if len(data_shape) != 4:
        raise ValueError(f"expected 4D split array, got {data_shape}")
    if expected_width is None:
        raise ValueError("could_not_infer_checkpoint_expected_width")
    effective_width = crop_width if crop_width else data_shape[-1]
    if crop_width and crop_width > data_shape[-1]:
        raise ValueError(f"crop_width_gt_dataset_width: crop_width={crop_width}, dataset_width={data_shape[-1]}")
    if effective_width != expected_width:
        raise ValueError(f"incompatible_input_width: checkpoint_expected={expected_width}, dataset_width={data_shape[-1]}")

    with np.load(data_path, allow_pickle=False) as npz:
        x = np.asarray(npz[data_key], dtype=np.float32)
        y = np.asarray(npz[label_key], dtype=np.int64)
    if crop_width:
        x = x[..., :crop_width]

    model = CNNNetWorkNoAttention(class_number=int(max(y.max() + 1, 5)))
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    preds = []
    labels = []
    loss_sum = 0.0
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch_x = torch.from_numpy(x[start : start + batch_size]).to(device)
            batch_y = torch.from_numpy(y[start : start + batch_size]).to(device)
            logits = model(batch_x)
            loss_sum += float(criterion(logits, batch_y).item())
            preds.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
            labels.extend(batch_y.detach().cpu().numpy().tolist())

    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    macro_precision = precision_score(labels, preds, average="macro", zero_division=0)
    macro_recall = recall_score(labels, preds, average="macro", zero_division=0)
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "loss": loss_sum / max(len(labels), 1),
        "n_samples": len(labels),
        "report": report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--splits", nargs="*", default=["test"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-checkpoints", type=int, default=0, help="0 means all checkpoints")
    parser.add_argument("--crop-width", type=int, default=0, help="Optional last-axis crop width before evaluation")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    dataset_dir = (root / args.dataset_dir).resolve()
    model_dir = (root / args.model_dir).resolve()
    out = (root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    checkpoints = sorted(model_dir.glob("Seed_*/*.pth"))
    if args.max_checkpoints:
        checkpoints = checkpoints[: args.max_checkpoints]

    crop_width = args.crop_width or ""
    historical_width_caveat = str(bool(args.crop_width)).lower()
    human_confirmed = str(args.crop_width == 512).lower()
    caveat_note = HISTORICAL_WIDTH_NOTE if args.crop_width == 512 else ""

    def caveat_fields() -> dict[str, str | int]:
        return {
            "crop_width": crop_width,
            "historical_width_caveat": historical_width_caveat,
            "human_researcher_confirmed_crop_assumption": human_confirmed,
        }

    metric_rows = []
    report_rows = []
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(root),
        "script": relpath(Path(__file__), root),
        "dataset_dir": relpath(dataset_dir, root),
        "model_dir": relpath(model_dir, root),
        "splits": args.splits,
        "device": str(device),
        "crop_width": args.crop_width or None,
        "historical_width_caveat": bool(args.crop_width),
        "human_researcher_confirmed_crop_assumption": args.crop_width == 512,
        "note": caveat_note,
        "status": "started",
        "runs": [],
    }

    for checkpoint in checkpoints:
        condition = condition_from_checkpoint(checkpoint)
        seed = seed_from_checkpoint(checkpoint)
        data_path = dataset_path(dataset_dir, condition)
        for split in args.splits:
            base = {
                "condition": condition,
                "seed": seed,
                "model_family": "no_attention_6ch",
                "split": split,
                "checkpoint_path": relpath(checkpoint, root),
                "dataset_path": relpath(data_path, root) if data_path.exists() else data_path.as_posix(),
            }
            try:
                if not data_path.exists():
                    raise FileNotFoundError(f"dataset missing: {data_path}")
                result = evaluate_checkpoint(root, checkpoint, data_path, split, args.batch_size, device, args.crop_width or None)
                metric_rows.append(
                    {
                        **base,
                        "accuracy": f"{result['accuracy']:.10f}",
                        "macro_f1": f"{result['macro_f1']:.10f}",
                        "weighted_f1": f"{result['weighted_f1']:.10f}",
                        "macro_precision": f"{result['macro_precision']:.10f}",
                        "macro_recall": f"{result['macro_recall']:.10f}",
                        "loss": f"{result['loss']:.10f}",
                        "n_samples": result["n_samples"],
                        "status": "ok",
                        **caveat_fields(),
                        "notes": f"evaluated_existing_checkpoint_no_cross_split; crop_width={args.crop_width or 'none'}"
                        + (f"; {caveat_note}" if caveat_note else ""),
                    }
                )
                for class_id, metrics in result["report"].items():
                    if not isinstance(metrics, dict) or class_id in {"accuracy", "macro avg", "weighted avg"}:
                        continue
                    report_rows.append(
                        {
                            "condition": condition,
                            "seed": seed,
                            "split": split,
                            "class_id": class_id,
                            "precision": f"{metrics.get('precision', 0.0):.10f}",
                            "recall": f"{metrics.get('recall', 0.0):.10f}",
                            "f1_score": f"{metrics.get('f1-score', 0.0):.10f}",
                            "support": metrics.get("support", ""),
                            "status": "ok",
                            **caveat_fields(),
                            "notes": caveat_note,
                        }
                    )
                manifest["runs"].append(
                    {
                        **base,
                        "status": "ok",
                        "n_samples": result["n_samples"],
                        "crop_width": args.crop_width or None,
                        "historical_width_caveat": bool(args.crop_width),
                        "human_researcher_confirmed_crop_assumption": args.crop_width == 512,
                        "note": caveat_note,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                metric_rows.append(
                    {
                        **base,
                        "accuracy": "",
                        "macro_f1": "",
                        "weighted_f1": "",
                        "macro_precision": "",
                        "macro_recall": "",
                        "loss": "",
                        "n_samples": "",
                        "status": "error",
                        **caveat_fields(),
                        "notes": repr(exc),
                    }
                )
                report_rows.append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "split": split,
                        "class_id": "",
                        "precision": "",
                        "recall": "",
                        "f1_score": "",
                        "support": "",
                        "status": "error",
                        **caveat_fields(),
                        "notes": repr(exc),
                    }
                )
                manifest["runs"].append(
                    {
                        **base,
                        "status": "error",
                        "error": repr(exc),
                        "crop_width": args.crop_width or None,
                        "historical_width_caveat": bool(args.crop_width),
                        "human_researcher_confirmed_crop_assumption": args.crop_width == 512,
                        "note": caveat_note,
                    }
                )

    metric_fields = [
        "condition",
        "seed",
        "model_family",
        "split",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_precision",
        "macro_recall",
        "loss",
        "n_samples",
        "checkpoint_path",
        "dataset_path",
        "status",
        "crop_width",
        "historical_width_caveat",
        "human_researcher_confirmed_crop_assumption",
        "notes",
    ]
    write_csv(out / "baseline_metrics_long.csv", metric_fields, metric_rows)
    report_fields = [
        "condition",
        "seed",
        "split",
        "class_id",
        "precision",
        "recall",
        "f1_score",
        "support",
        "status",
        "crop_width",
        "historical_width_caveat",
        "human_researcher_confirmed_crop_assumption",
        "notes",
    ]
    write_csv(out / "classification_report_long.csv", report_fields, report_rows)

    ok_rows = [row for row in metric_rows if row["status"] == "ok"]
    summary_rows = []
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in ok_rows:
        grouped.setdefault((row["condition"], row["split"]), []).append(row)
    for (condition, split), rows in sorted(grouped.items()):
        for metric in ["accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall", "loss"]:
            values = np.array([float(row[metric]) for row in rows])
            summary_rows.append(
                {
                    "condition": condition,
                    "split": split,
                    "metric": metric,
                    "mean": f"{float(values.mean()):.10f}",
                    "std": f"{float(values.std(ddof=0)):.10f}",
                    "n_seeds": len(rows),
                    "seeds": "|".join(row["seed"] for row in rows),
                    "status": "ok",
                    **caveat_fields(),
                    "notes": caveat_note,
                }
            )
    if not summary_rows:
        summary_rows.append(
            {
                "condition": "",
                "split": "",
                "metric": "",
                "mean": "",
                "std": "",
                "n_seeds": 0,
                "seeds": "",
                "status": "no_successful_evaluations",
                **caveat_fields(),
                "notes": "See baseline_metrics_long.csv error rows.",
            }
        )
    summary_fields = [
        "condition",
        "split",
        "metric",
        "mean",
        "std",
        "n_seeds",
        "seeds",
        "status",
        "crop_width",
        "historical_width_caveat",
        "human_researcher_confirmed_crop_assumption",
        "notes",
    ]
    write_csv(out / "baseline_metrics_summary.csv", summary_fields, summary_rows)

    paper_rows = []
    for (condition, split), rows in sorted(grouped.items()):
        metric_values = {
            metric: np.array([float(row[metric]) for row in rows])
            for metric in ["accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall"]
        }
        paper_rows.append(
            {
                "condition": condition,
                "model_family": "no_attention_6ch",
                "split": split,
                "accuracy_mean": f"{float(metric_values['accuracy'].mean()):.10f}",
                "accuracy_std": f"{float(metric_values['accuracy'].std(ddof=0)):.10f}",
                "macro_f1_mean": f"{float(metric_values['macro_f1'].mean()):.10f}",
                "macro_f1_std": f"{float(metric_values['macro_f1'].std(ddof=0)):.10f}",
                "weighted_f1_mean": f"{float(metric_values['weighted_f1'].mean()):.10f}",
                "weighted_f1_std": f"{float(metric_values['weighted_f1'].std(ddof=0)):.10f}",
                "macro_precision_mean": f"{float(metric_values['macro_precision'].mean()):.10f}",
                "macro_precision_std": f"{float(metric_values['macro_precision'].std(ddof=0)):.10f}",
                "macro_recall_mean": f"{float(metric_values['macro_recall'].mean()):.10f}",
                "macro_recall_std": f"{float(metric_values['macro_recall'].std(ddof=0)):.10f}",
                "n_seeds": len(rows),
                "n_samples": rows[0]["n_samples"] if rows else "",
                **caveat_fields(),
                "paper_use_status": "main_candidate_with_width_caveat" if args.crop_width == 512 else "main_candidate",
                "notes": caveat_note,
            }
        )
    paper_fields = [
        "condition",
        "model_family",
        "split",
        "accuracy_mean",
        "accuracy_std",
        "macro_f1_mean",
        "macro_f1_std",
        "weighted_f1_mean",
        "weighted_f1_std",
        "macro_precision_mean",
        "macro_precision_std",
        "macro_recall_mean",
        "macro_recall_std",
        "n_seeds",
        "n_samples",
        "crop_width",
        "historical_width_caveat",
        "human_researcher_confirmed_crop_assumption",
        "paper_use_status",
        "notes",
    ]
    write_csv(out / "baseline_metrics_paper_table.csv", paper_fields, paper_rows)

    notes = [
        "# Baseline Metrics Notes",
        "",
        f"- Evaluated checkpoints: {len(checkpoints)}.",
        f"- Successful metric rows: {len(ok_rows)}.",
        f"- Splits evaluated: {', '.join(args.splits)}.",
        f"- Crop width: {args.crop_width or 'none'}.",
        f"- Historical width caveat: {historical_width_caveat}.",
        f"- Human researcher confirmed crop assumption: {human_confirmed}.",
        f"- Note: {caveat_note if caveat_note else 'No historical crop assumption used.'}",
        "",
        "## Paper Use",
        "",
        "- These rows are candidate Table 1 baseline metrics only under the recorded historical-width caveat.",
        "- Cross-condition splits are not evaluated here and should not be inferred from this table.",
    ]
    (out / "baseline_metrics_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")

    manifest["status"] = "ok" if ok_rows else "no_successful_evaluations"
    manifest["n_checkpoints"] = len(checkpoints)
    manifest["n_metric_rows"] = len(metric_rows)
    manifest["n_successful_rows"] = len(ok_rows)
    (out / "baseline_eval_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(metric_rows)} metric rows to {relpath(out, root)}")
    print(f"Successful evaluations: {len(ok_rows)}")
    return 0 if ok_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
