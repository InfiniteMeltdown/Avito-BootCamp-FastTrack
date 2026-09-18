from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def check_probabilities(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Probabilities must be finite and within [0, 1].")
    return values


def score(labels, probabilities) -> float:
    y = np.asarray(labels, dtype=float)
    p = check_probabilities(probabilities)
    if y.shape != p.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be binary and match prediction shape.")
    return float(1 - np.mean((y - p) ** 2))
