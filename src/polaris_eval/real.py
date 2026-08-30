"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

#!/usr/bin/env python3

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from polaris_eval.datasets import load_sdrr
from polaris_eval.io import EvaluationError, sha256

RESULT_FIELDS = (
    "query_id",
    "reference_id",
    "query_seconds",
    "ground_truth_offset_seconds",
    "status",
    "predicted_reference_id",
    "predicted_offset_seconds",
    "top1_absolute_offset_error_seconds",
    "query_fingerprints",
    "total_time",
    "error",
)


@dataclass(frozen=True)
class RealQuery:
    query_id: str
    reference_id: str
    query_path: Path
    reference_path: Path
    reference_begin_seconds: float
    query_duration_seconds: float


def load_real_manifest(path: Path) -> list[RealQuery]:
    path = path.expanduser().resolve()
    rows = load_sdrr(path.parent, enforce_release_shape=False)
    return [
        RealQuery(
            query_id=row.query_id,
            reference_id=row.reference_id,
            query_path=row.query_path,
            reference_path=row.reference_path,
            reference_begin_seconds=row.reference_begin_seconds,
            query_duration_seconds=row.query_seconds,
        )
        for row in rows
    ]


def unique_references(items: Sequence[RealQuery]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for item in items:
        previous = result.setdefault(item.reference_id, item.reference_path)
        if previous != item.reference_path:
            raise EvaluationError(f"inconsistent reference path: {item.reference_id}")
    return result


def file_sha256(path: Path) -> str:
    return sha256(path)


def summarize_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("SD-RR metrics require at least one query")
    top1 = sum(row.get("predicted_reference_id") == row.get("reference_id") for row in rows)
    track_and_offset = sum(
        row.get("predicted_reference_id") == row.get("reference_id")
        and row.get("top1_absolute_offset_error_seconds") not in (None, "")
        and float(row["top1_absolute_offset_error_seconds"]) <= 0.1
        for row in rows
    )
    return {
        "queries": len(rows),
        "errors": sum(str(row.get("status")) == "error" for row in rows),
        "track_top1": top1 / len(rows),
        "track_and_offset_top1_at_0.1s": track_and_offset / len(rows),
    }
