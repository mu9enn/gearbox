#!/usr/bin/env python3
"""Streaming helpers for auditing large npz/npy arrays."""

from __future__ import annotations

import csv
import hashlib
import math
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class NpyMember:
    key: str
    shape: tuple[int, ...]
    fortran_order: bool
    dtype: np.dtype

    @property
    def n_samples(self) -> int:
        return int(self.shape[0]) if self.shape else 0

    @property
    def sample_bytes(self) -> int:
        if len(self.shape) <= 1:
            return int(self.dtype.itemsize)
        return int(math.prod(self.shape[1:]) * self.dtype.itemsize)


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def relpath(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def read_npy_header(fp) -> tuple[tuple[int, ...], bool, np.dtype]:
    version = np.lib.format.read_magic(fp)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(fp)
    elif version in {(2, 0), (3, 0)}:
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(fp)
    else:
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(fp)
    return tuple(shape), bool(fortran_order), np.dtype(dtype)


def npz_members(path: Path) -> dict[str, NpyMember]:
    members: dict[str, NpyMember] = {}
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if not info.filename.endswith(".npy"):
                continue
            key = Path(info.filename).stem
            with zf.open(info) as fp:
                shape, fortran_order, dtype = read_npy_header(fp)
            members[key] = NpyMember(key=key, shape=shape, fortran_order=fortran_order, dtype=dtype)
    return members


def iter_npy_sample_bytes(npz_path: Path, key: str):
    """Yield raw bytes for each first-axis sample from a .npy member in an .npz.

    This assumes C-order arrays. Fortran-order arrays are intentionally rejected
    because first-axis samples are not contiguous on disk.
    """
    member_name = f"{key}.npy"
    with zipfile.ZipFile(npz_path) as zf:
        with zf.open(member_name) as fp:
            shape, fortran_order, dtype = read_npy_header(fp)
            member = NpyMember(key=key, shape=shape, fortran_order=fortran_order, dtype=dtype)
            if fortran_order:
                raise ValueError(f"{npz_path}:{key} is Fortran-order; streaming sample hashing is unsupported")
            if member.n_samples == 0:
                return
            if len(shape) == 1:
                sample_bytes = dtype.itemsize
            else:
                sample_bytes = member.sample_bytes
            for _ in range(member.n_samples):
                chunk = fp.read(sample_bytes)
                if len(chunk) != sample_bytes:
                    raise EOFError(f"Unexpected EOF while reading {npz_path}:{key}")
                yield member, chunk


def hash_sample_bytes(member: NpyMember, sample_bytes: bytes, algorithm: str = "sha1") -> str:
    digest = hashlib.new(algorithm)
    digest.update(str(member.shape[1:]).encode("utf-8"))
    digest.update(str(member.dtype).encode("utf-8"))
    digest.update(sample_bytes)
    return digest.hexdigest()


def split_keys(members: dict[str, NpyMember]) -> list[str]:
    preferred = [
        "train_set",
        "valid_set",
        "test_set",
        "cross_finetune_set",
        "cross_test_set",
        "cross_set",
    ]
    observed = [key for key in members if key.endswith("_set") and not key.endswith("_label")]
    ordered = [key for key in preferred if key in observed]
    ordered.extend(sorted(key for key in observed if key not in ordered))
    return ordered


def label_key_for(split_key: str) -> str:
    return split_key[:-4] + "_label" if split_key.endswith("_set") else split_key + "_label"


def hash_split(npz_path: Path, key: str, algorithm: str = "sha1") -> tuple[set[str], dict[str, int]]:
    hashes: set[str] = set()
    duplicate_counter: Counter[str] = Counter()
    n_samples = 0
    for member, sample_bytes in iter_npy_sample_bytes(npz_path, key):
        value = hash_sample_bytes(member, sample_bytes, algorithm=algorithm)
        duplicate_counter[value] += 1
        hashes.add(value)
        n_samples += 1
    internal_duplicate_count = sum(count - 1 for count in duplicate_counter.values() if count > 1)
    return hashes, {"n_samples": n_samples, "internal_duplicate_count": internal_duplicate_count}


def load_label_distribution(npz_path: Path, label_key: str) -> str:
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            labels = np.asarray(data[label_key]).reshape(-1)
    except Exception as exc:  # noqa: BLE001
        return f"unavailable:{repr(exc)}"
    counter = Counter(labels.tolist())
    return "|".join(f"{label}:{count}" for label, count in sorted(counter.items(), key=lambda item: str(item[0])))
