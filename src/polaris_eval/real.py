"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

#!/usr/bin/env python3

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from polaris_eval.datasets import load_sdrr
from polaris_eval.io import EvaluationError, sha256


@dataclass(frozen=True)
class RealQuery:
    query_id: str
    reference_id: str
    query_path: Path
    reference_path: Path
    reference_begin_seconds: float


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
