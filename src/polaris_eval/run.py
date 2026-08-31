"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter

import numpy as np

from polaris import DEFAULT_CONFIG, Polaris
from polaris.engine import build_reference_index, read_audio
from polaris.index import build_packed_index, has_packed_index
from polaris_eval.controls import (
    CONTROL_SYSTEMS,
    build_control_index,
    control_fingerprints,
    reference_key,
    resolved_control,
)
from polaris_eval.datasets import (
    PexAnnotation,
    SdrrQuery,
    load_pex,
    load_sdrr,
    unique_sdrr_references,
)
from polaris_eval.io import EvaluationError, sha256, write_csv, write_json
from polaris_eval.metrics import (
    PEX_RESULT_FIELDS,
    SDRR_RESULT_FIELDS,
    summarize_pex,
    summarize_sdrr,
)

POLARIS_SYSTEMS = ("polaris_o", "polaris_a", "polaris_f")
SYSTEMS = (*POLARIS_SYSTEMS, *CONTROL_SYSTEMS)
POLARIS_PEX_FIELDS = (*PEX_RESULT_FIELDS[:-2], "query_stage", *PEX_RESULT_FIELDS[-2:])
POLARIS_SDRR_FIELDS = (*SDRR_RESULT_FIELDS[:-2], "query_stage", *SDRR_RESULT_FIELDS[-2:])
_RECOGNIZER: Polaris | None = None
_SYSTEM = "polaris_f"


def _initialize_worker(index: str, system: str) -> None:
    global _RECOGNIZER, _SYSTEM
    _RECOGNIZER = Polaris(index, index_backend="packed")
    _SYSTEM = system


def _recognize(samples: np.ndarray, sample_rate: int) -> dict[str, object]:
    if _RECOGNIZER is None:
        raise RuntimeError("worker was not initialized")
    if _SYSTEM in POLARIS_SYSTEMS:
        return _RECOGNIZER.recognize_samples(
            samples,
            mode=_SYSTEM[-1],
            sample_rate=sample_rate,
            topn=2,
        )
    started = perf_counter()
    hashes = control_fingerprints(
        samples,
        sample_rate,
        system=_SYSTEM,
        query=True,
        packed=True,
    )
    results = _RECOGNIZER.recognize_hashes(hashes, topn=2)
    return {
        "query_records": len(hashes),
        "query_stage": "two_hop" if _SYSTEM == "magnitude_maxima" else "original",
        "total_time": perf_counter() - started,
        "results": results,
    }


def _sdrr_result_row(
    *,
    query_id: str,
    reference_id: str,
    ground_truth_offset: float,
    recognition: Mapping[str, object],
) -> dict[str, object]:
    matches = list(recognition.get("results", []))
    top = matches[0] if matches else {}
    predicted_reference = str(top.get("song_name") or "")
    predicted_offset = (
        float(top["offset_seconds"]) if top.get("offset_seconds") is not None else None
    )
    offset_error = (
        abs(predicted_offset - ground_truth_offset)
        if predicted_reference == reference_id
        and predicted_offset is not None
        and ground_truth_offset is not None
        else None
    )
    return {
        "query_id": query_id,
        "reference_id": reference_id,
        "status": "matched" if matches else "no_match",
        "predicted_reference_id": predicted_reference,
        "predicted_offset_seconds": predicted_offset,
        "absolute_offset_error_seconds": offset_error,
        "query_records": int(recognition.get("query_records") or 0),
        "query_stage": str(recognition.get("query_stage") or ""),
        "total_time": recognition.get("total_time"),
        "error": "",
    }


def _sdrr_error_row(
    query_id: str,
    reference_id: str,
    error: Exception,
) -> dict[str, object]:
    return {
        "query_id": query_id,
        "reference_id": reference_id,
        "status": "error",
        "predicted_reference_id": "",
        "predicted_offset_seconds": None,
        "absolute_offset_error_seconds": None,
        "query_records": 0,
        "query_stage": "",
        "total_time": None,
        "error": f"{type(error).__name__}: {error}",
    }


def _pex_result_row(
    item: PexAnnotation,
    recognition: Mapping[str, object],
) -> dict[str, object]:
    matches = list(recognition.get("results", []))
    top = matches[0] if matches else {}
    return {
        "trial_id": item.annotation_id,
        "query_id": item.query_id,
        "reference_id": item.reference_id,
        "query_begin": item.query_begin,
        "status": "matched" if matches else "no_match",
        "predicted_reference_id": str(top.get("song_name") or ""),
        "query_records": int(recognition.get("query_records") or 0),
        "query_stage": str(recognition.get("query_stage") or ""),
        "total_time": recognition.get("total_time"),
        "error": "",
    }


def _pex_error_row(item: PexAnnotation, error: Exception) -> dict[str, object]:
    return {
        "trial_id": item.annotation_id,
        "query_id": item.query_id,
        "reference_id": item.reference_id,
        "query_begin": item.query_begin,
        "status": "error",
        "predicted_reference_id": "",
        "query_records": 0,
        "query_stage": "",
        "total_time": None,
        "error": f"{type(error).__name__}: {error}",
    }


def _sdrr_worker(item: SdrrQuery) -> dict[str, object]:
    started = perf_counter()
    try:
        samples = read_audio(item.query_path, sample_rate=DEFAULT_CONFIG.audio.sample_rate)
        recognition = _recognize(samples, DEFAULT_CONFIG.audio.sample_rate)
        recognition["total_time"] = perf_counter() - started
        return _sdrr_result_row(
            query_id=item.query_id,
            reference_id=item.reference_id,
            ground_truth_offset=item.reference_begin_seconds,
            recognition=recognition,
        )
    except Exception as error:
        return _sdrr_error_row(item.query_id, item.reference_id, error)


def _pex_worker(task: tuple[Path, tuple[PexAnnotation, ...]]) -> list[dict[str, object]]:
    path, annotations = task
    try:
        samples = read_audio(path, sample_rate=DEFAULT_CONFIG.audio.sample_rate)
    except Exception as error:
        return [_pex_error_row(item, error) for item in annotations]
    rows = []
    sample_rate = DEFAULT_CONFIG.audio.sample_rate
    for item in annotations:
        try:
            start = max(0, round(item.query_begin * sample_rate))
            stop = min(len(samples), round(item.query_end * sample_rate))
            if stop <= start:
                raise EvaluationError("PEX oracle segment is empty")
            rows.append(
                _pex_result_row(item, _recognize(samples[start:stop], sample_rate))
            )
        except Exception as error:
            rows.append(_pex_error_row(item, error))
    return rows


def _reference_seconds(dataset: str, root: Path, reference_ids: set[str]) -> float:
    manifest = root / ("reference_manifest.csv" if dataset == "sdrr" else "fma_tracks.csv")
    with manifest.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    if dataset == "sdrr":
        return sum(float(row["source_duration_seconds"]) for row in rows)
    by_id = {row["track_id"].strip().zfill(6): row for row in rows}
    missing = reference_ids - set(by_id)
    if missing:
        raise EvaluationError(f"PEX reference durations are missing: {sorted(missing)[:5]}")
    return sum(float(by_id[reference_id]["track_duration"]) for reference_id in reference_ids)


def _index_summary(index: Path, reference_seconds: float) -> dict[str, object]:
    postings = 0
    with (index / "songs.csv").open(encoding="utf-8") as source:
        for line in source:
            _, remainder = line.rstrip("\n").split(",", maxsplit=1)
            _, count = remainder.rsplit(",", maxsplit=1)
            postings += int(count)
    return {
        "reference_audio_seconds": reference_seconds,
        "reference_hash_postings": postings,
    }


def _summary(
    rows: Sequence[Mapping[str, object]],
    *,
    dataset: str,
    system: str,
    index_summary: Mapping[str, object],
) -> dict[str, object]:
    result: dict[str, object] = {
        "schema": "polaris-paper-results-v1",
        "dataset": dataset,
        "system": system,
        **(summarize_sdrr(rows) if dataset == "sdrr" else summarize_pex(rows)),
        "index": dict(index_summary),
    }
    if system == "polaris_a":
        result["two_hop_avoided_fraction"] = sum(
            row["query_stage"] == "original" for row in rows
        ) / len(rows)
    return result


def _load_completed(
    path: Path,
    *,
    fields: tuple[str, ...],
    id_field: str,
) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows or set(rows[0]) != set(fields):
        return {}
    return {row[id_field]: row for row in rows if row["status"] != "error"}


def _resolved(system: str) -> dict[str, object]:
    if system in CONTROL_SYSTEMS:
        return resolved_control(system)
    evidence = {
        "polaris_o": "original faces with adjacent-bin probing",
        "polaris_a": "POLARIS-O first; multiprobed two-hop hashes on fallback",
        "polaris_f": "POLARIS-O plus multiprobed two-hop hashes",
    }
    return {
        "name": system,
        "query_mode": {"polaris_o": "original", "polaris_a": "adaptive", "polaris_f": "two_hop"}[
            system
        ],
        "query_evidence": evidence[system],
        "polaris": DEFAULT_CONFIG.to_dict(),
    }


def evaluate(
    *,
    dataset: str,
    root: Path,
    system: str,
    index: Path,
    output: Path,
    workers: int,
) -> dict[str, object]:
    if dataset == "pex" and system in CONTROL_SYSTEMS:
        raise EvaluationError("paper controls are evaluated only on SD-RR")
    if dataset == "sdrr":
        items = load_sdrr(root)
        references = unique_sdrr_references(items)
        tasks: list[object] = list(items)
        expected_ids = {item.query_id for item in items}
    else:
        annotations, references, queries = load_pex(root)
        grouped: dict[str, list[PexAnnotation]] = defaultdict(list)
        for item in annotations:
            if item.exact_scale:
                grouped[item.query_id].append(item)
        tasks = [(queries[query_id], tuple(grouped[query_id])) for query_id in sorted(grouped)]
        expected_ids = {
            item.annotation_id for item in annotations if item.exact_scale
        }

    index.mkdir(parents=True, exist_ok=True)
    index_configuration = {
        "reference_key": reference_key(system),
        "configuration": (
            DEFAULT_CONFIG.to_dict() if system in POLARIS_SYSTEMS else _resolved(system)
        ),
    }
    configuration_path = index / "configuration.json"
    if configuration_path.is_file() and json.loads(
        configuration_path.read_text(encoding="utf-8")
    ) != index_configuration:
        raise EvaluationError(f"index was built with a different configuration: {index}")
    if reference_key(system) == "polaris":
        inserted = build_reference_index(
            list(references.values()),
            index,
            config=DEFAULT_CONFIG,
            workers=workers,
        )
    else:
        inserted = build_control_index(
            list(references.values()),
            index,
            system=system,
            workers=workers,
        )
    if not configuration_path.is_file():
        write_json(index_configuration, configuration_path)
    if inserted and has_packed_index(index):
        raise EvaluationError("packed index is stale after adding references")
    if not has_packed_index(index):
        build_packed_index(index)

    output.mkdir(parents=True, exist_ok=True)
    write_json(_resolved(system), output / "resolved_configuration.json")
    results_path = output / "query_results.csv"
    fields = POLARIS_SDRR_FIELDS if dataset == "sdrr" else POLARIS_PEX_FIELDS
    id_field = "query_id" if dataset == "sdrr" else "trial_id"
    rows: dict[str, dict[str, object] | dict[str, str]] = _load_completed(
        results_path,
        fields=fields,
        id_field=id_field,
    )
    if dataset == "sdrr":
        remaining = [task for task in tasks if task.query_id not in rows]  # type: ignore[attr-defined]
        worker = _sdrr_worker
    else:
        remaining = [
            task
            for task in tasks
            if any(item.annotation_id not in rows for item in task[1])  # type: ignore[index]
        ]
        worker = _pex_worker

    with multiprocessing.Pool(
        workers,
        initializer=_initialize_worker,
        initargs=(str(index), system),
    ) as pool:
        for completed, produced in enumerate(
            pool.imap_unordered(worker, remaining, chunksize=1),
            start=1,
        ):
            completed_rows = [produced] if dataset == "sdrr" else produced
            for row in completed_rows:
                rows[str(row[id_field])] = row
            if completed % 10 == 0 or completed == len(remaining):
                write_csv([rows[key] for key in sorted(rows)], results_path, fields)
                print(f"[{completed}/{len(remaining)}] {dataset}/{system}", flush=True)

    if set(rows) != expected_ids:
        raise EvaluationError("result IDs do not exactly match the paper protocol")
    ordered = [rows[key] for key in sorted(rows)]
    summary = _summary(
        ordered,
        dataset=dataset,
        system=system,
        index_summary=_index_summary(
            index,
            _reference_seconds(dataset, root, set(references)),
        ),
    )
    summary["manifest_sha256"] = sha256(
        root / ("manifest.csv" if dataset == "sdrr" else "annotations.csv")
    )
    write_json(summary, output / "summary.json")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the frozen POLARIS paper protocol.")
    parser.add_argument("dataset", choices=("sdrr", "pex"))
    parser.add_argument("--system", choices=SYSTEMS, action="append")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--workers", type=int, default=max(1, min(6, os.cpu_count() or 1)))
    args = parser.parse_args(argv)
    default_systems = POLARIS_SYSTEMS if args.dataset == "pex" else SYSTEMS
    systems = tuple(dict.fromkeys(args.system or default_systems))
    root = (
        args.data_root
        or Path("data") / ("sdrr" if args.dataset == "sdrr" else "pex_hard_medium")
    ).resolve()
    output_root = args.output_root.resolve()
    for system in systems:
        summary = evaluate(
            dataset=args.dataset,
            root=root,
            system=system,
            index=output_root
            / "shared"
            / args.dataset
            / f"{reference_key(system)}_index",
            output=output_root / args.dataset / system,
            workers=args.workers,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
