"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

#!/usr/bin/env python3

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from polaris_eval.io import EvaluationError


@dataclass(frozen=True)
class SdrrQuery:
    query_id: str
    reference_id: str
    query_path: Path
    reference_path: Path
    reference_begin_seconds: float
    query_seconds: float


@dataclass(frozen=True)
class PexAnnotation:
    reference_id: str
    query_id: str
    reference_begin: int
    reference_end: int
    query_begin: int
    query_end: int
    tempo: str
    pitch: str

    @property
    def exact_scale(self) -> bool:
        return self.tempo in ("", "100") and self.pitch in ("", "0")

    @property
    def annotation_id(self) -> str:
        return ":".join(
            str(value)
            for value in (
                self.reference_id,
                self.query_id,
                self.reference_begin,
                self.reference_end,
                self.query_begin,
                self.query_end,
            )
        )


def _resolve_audio(root: Path, raw: str, kind: str) -> Path:
    value = Path(raw).expanduser()
    path = value if value.is_absolute() else root / value
    path = path.resolve()
    if not path.is_file():
        raise EvaluationError(f"{kind} audio is missing: {path}")
    return path


def load_sdrr(root: Path, *, enforce_release_shape: bool = True) -> list[SdrrQuery]:
    root = root.expanduser().resolve()
    manifest = root / "manifest.csv"
    if not manifest.is_file():
        raise EvaluationError(f"SD-RR manifest is missing: {manifest}")
    required = {
        "query_id",
        "reference_id",
        "query_path",
        "reference_path",
        "reference_begin_seconds",
        "query_duration_seconds",
    }
    rows: list[SdrrQuery] = []
    seen: set[str] = set()
    reference_paths: dict[str, Path] = {}
    with manifest.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise EvaluationError(
                "SD-RR manifest is missing columns: " + ", ".join(sorted(missing))
            )
        for line_number, row in enumerate(reader, start=2):
            query_id = row["query_id"].strip()
            reference_id = row["reference_id"].strip()
            if not query_id or query_id in seen:
                raise EvaluationError(f"SD-RR line {line_number}: duplicate/empty query_id")
            seen.add(query_id)
            query_path = _resolve_audio(root, row["query_path"], "query")
            reference_path = _resolve_audio(root, row["reference_path"], "reference")
            if query_path.stem != query_id or reference_path.stem != reference_id:
                raise EvaluationError(f"SD-RR line {line_number}: ID/path stem mismatch")
            previous = reference_paths.setdefault(reference_id, reference_path)
            if previous != reference_path:
                raise EvaluationError(f"SD-RR line {line_number}: inconsistent reference path")
            query_seconds = float(row["query_duration_seconds"])
            if abs(query_seconds - 10.0) > 1e-6:
                raise EvaluationError(f"SD-RR line {line_number}: query is not exactly 10 s")
            rows.append(
                SdrrQuery(
                    query_id=query_id,
                    reference_id=reference_id,
                    query_path=query_path,
                    reference_path=reference_path,
                    reference_begin_seconds=float(row["reference_begin_seconds"]),
                    query_seconds=query_seconds,
                )
            )
    if enforce_release_shape and (len(rows) != 1_488 or len(reference_paths) != 496):
        raise EvaluationError(
            f"SD-RR v1.0 requires 496 references/1,488 queries; found "
            f"{len(reference_paths)}/{len(rows)}"
        )
    return rows


def unique_sdrr_references(rows: list[SdrrQuery]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for row in rows:
        previous = result.setdefault(row.reference_id, row.reference_path)
        if previous != row.reference_path:
            raise EvaluationError(f"inconsistent SD-RR reference: {row.reference_id}")
    return result


def load_pex(root: Path) -> tuple[list[PexAnnotation], dict[str, Path], dict[str, Path]]:
    root = root.expanduser().resolve()
    annotations_path = root / "annotations.csv"
    reference_root = root / "references"
    query_root = root / "queries"
    if not annotations_path.is_file() or not reference_root.is_dir() or not query_root.is_dir():
        raise EvaluationError(
            "PEX directory must contain annotations.csv, references/, and queries/"
        )
    references: dict[str, Path] = {}
    for path in sorted(reference_root.rglob("*")):
        if path.is_file() and not path.name.startswith("."):
            if path.stem in references:
                raise EvaluationError(f"duplicate PEX reference id: {path.stem}")
            references[path.stem] = path.resolve()
    queries = {
        path.stem: path.resolve()
        for path in sorted(query_root.iterdir())
        if path.is_file() and not path.name.startswith(".")
    }
    required = {
        "reference_id",
        "query_id",
        "reference_begin",
        "reference_end",
        "query_begin",
        "query_end",
    }
    annotations: list[PexAnnotation] = []
    with annotations_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise EvaluationError("PEX annotations are missing: " + ", ".join(sorted(missing)))
        for line_number, row in enumerate(reader, start=2):
            reference_id = row["reference_id"].strip()
            query_id = row["query_id"].strip()
            if reference_id not in references or query_id not in queries:
                raise EvaluationError(f"PEX line {line_number}: referenced audio is missing")
            begin = int(row["query_begin"])
            end = int(row["query_end"])
            if end <= begin:
                raise EvaluationError(f"PEX line {line_number}: empty query interval")
            tempo = row.get("tempo", "").strip()
            pitch = row.get("pitch", "").strip()
            annotations.append(
                PexAnnotation(
                    reference_id=reference_id,
                    query_id=query_id,
                    reference_begin=int(row["reference_begin"]),
                    reference_end=int(row["reference_end"]),
                    query_begin=begin,
                    query_end=end,
                    tempo=tempo,
                    pitch=pitch,
                )
            )
    exact = [annotation for annotation in annotations if annotation.exact_scale]
    if len(references) != 953 or len(queries) != 219 or len(exact) != 791:
        raise EvaluationError(
            "PEX Hard Medium protocol requires 953 references, 219 query files, "
            f"and 791 exact-scale annotations; found {len(references)}, {len(queries)}, {len(exact)}"
        )
    return annotations, references, queries
