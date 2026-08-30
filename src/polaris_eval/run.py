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
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter

import numpy as np

from polaris import DEFAULT_CONFIG, Polaris
from polaris.engine import build_reference_index, read_audio
from polaris.index import build_packed_index, has_packed_index
from polaris_eval.datasets import (
    PexAnnotation,
    SdrrQuery,
    load_pex,
    load_sdrr,
    unique_sdrr_references,
)
from polaris_eval.io import EvaluationError, sha256, write_csv, write_json
from polaris_eval.variants import (
    ABLATION_SYSTEMS,
    build_variant_index,
    reference_key,
    variant_hashes,
)

SYSTEMS = ("polaris", "polaris_adaptive", *ABLATION_SYSTEMS)
RESULT_FIELDS = (
    "trial_id",
    "query_id",
    "reference_id",
    "query_begin",
    "query_end",
    "query_seconds",
    "status",
    "predicted_reference_id",
    "predicted_offset_seconds",
    "ground_truth_offset_seconds",
    "absolute_offset_error_seconds",
    "query_fingerprints",
    "stopped_stage",
    "total_time",
    "error",
)

_WORKER: Polaris | None = None
_MODE = "polaris"


def _initialize_worker(index: str, mode: str) -> None:
    global _WORKER, _MODE
    _WORKER = Polaris(index, index_backend="packed")
    _MODE = mode


def _recognize(samples: np.ndarray, sample_rate: int) -> dict[str, object]:
    if _WORKER is None:
        raise RuntimeError("worker was not initialized")
    if _MODE == "polaris_adaptive":
        return _WORKER.recognize_adaptive_samples(samples, sample_rate=sample_rate, topn=10)
    if _MODE == "polaris":
        return _WORKER.recognize_samples(samples, sample_rate=sample_rate, topn=10)
    started = perf_counter()
    hashes = variant_hashes(
        samples,
        sample_rate,
        system=_MODE,
        for_query=True,
        packed=True,
    )
    results = _WORKER.recognize_hashes(hashes, topn=10)
    return {
        "query_fingerprints": len(hashes),
        "total_time": perf_counter() - started,
        "results": results,
    }


def _result_row(
    *,
    trial_id: str,
    query_id: str,
    reference_id: str,
    query_seconds: float,
    query_begin: float | None,
    query_end: float | None,
    ground_truth_offset: float | None,
    recognition: Mapping[str, object],
) -> dict[str, object]:
    matches = list(recognition.get("results", []))
    predicted = matches[0] if matches else {}
    predicted_reference = str(predicted.get("song_name") or "")
    predicted_offset = (
        float(predicted["offset_seconds"]) if predicted.get("offset_seconds") is not None else None
    )
    error = (
        abs(predicted_offset - ground_truth_offset)
        if ground_truth_offset is not None
        and predicted_reference == reference_id
        and predicted_offset is not None
        else None
    )
    return {
        "trial_id": trial_id,
        "query_id": query_id,
        "reference_id": reference_id,
        "query_begin": query_begin,
        "query_end": query_end,
        "query_seconds": query_seconds,
        "status": "matched" if matches else "no_match",
        "predicted_reference_id": predicted_reference,
        "predicted_offset_seconds": predicted_offset,
        "ground_truth_offset_seconds": ground_truth_offset,
        "absolute_offset_error_seconds": error,
        "query_fingerprints": int(recognition.get("query_fingerprints") or 0),
        "stopped_stage": str(recognition.get("stopped_stage") or "full"),
        "total_time": recognition.get("total_time"),
        "error": "",
    }


def _sdrr_worker(item: SdrrQuery) -> dict[str, object]:
    try:
        samples = read_audio(item.query_path, sample_rate=DEFAULT_CONFIG.audio.sample_rate)
        recognition = _recognize(samples, DEFAULT_CONFIG.audio.sample_rate)
        return _result_row(
            trial_id=item.query_id,
            query_id=item.query_id,
            reference_id=item.reference_id,
            query_seconds=item.query_seconds,
            query_begin=None,
            query_end=None,
            ground_truth_offset=item.reference_begin_seconds,
            recognition=recognition,
        )
    except Exception as error:
        return _error_row(
            item.query_id, item.query_id, item.reference_id, item.query_seconds, error
        )


def _error_row(
    trial_id: str,
    query_id: str,
    reference_id: str,
    query_seconds: float,
    error: Exception,
) -> dict[str, object]:
    return {
        "trial_id": trial_id,
        "query_id": query_id,
        "reference_id": reference_id,
        "query_begin": None,
        "query_end": None,
        "query_seconds": query_seconds,
        "status": "error",
        "predicted_reference_id": "",
        "predicted_offset_seconds": None,
        "ground_truth_offset_seconds": None,
        "absolute_offset_error_seconds": None,
        "query_fingerprints": 0,
        "stopped_stage": "error",
        "total_time": None,
        "error": f"{type(error).__name__}: {error}",
    }


def _pex_group_worker(
    task: tuple[Path, tuple[PexAnnotation, ...]],
) -> list[dict[str, object]]:
    path, annotations = task
    try:
        samples = read_audio(path, sample_rate=DEFAULT_CONFIG.audio.sample_rate)
    except Exception as error:
        return [
            _error_row(
                annotation.annotation_id,
                annotation.query_id,
                annotation.reference_id,
                annotation.query_end - annotation.query_begin,
                error,
            )
            for annotation in annotations
        ]
    rows = []
    sample_rate = DEFAULT_CONFIG.audio.sample_rate
    for annotation in annotations:
        try:
            start = max(0, round(annotation.query_begin * sample_rate))
            stop = min(len(samples), round(annotation.query_end * sample_rate))
            if stop <= start:
                raise EvaluationError("PEX oracle segment is empty")
            recognition = _recognize(samples[start:stop], sample_rate)
            rows.append(
                _result_row(
                    trial_id=annotation.annotation_id,
                    query_id=annotation.query_id,
                    reference_id=annotation.reference_id,
                    query_seconds=(stop - start) / sample_rate,
                    query_begin=float(annotation.query_begin),
                    query_end=float(annotation.query_end),
                    ground_truth_offset=None,
                    recognition=recognition,
                )
            )
        except Exception as error:
            rows.append(
                _error_row(
                    annotation.annotation_id,
                    annotation.query_id,
                    annotation.reference_id,
                    annotation.query_end - annotation.query_begin,
                    error,
                )
            )
    return rows


def _cluster_ci(
    rows: Sequence[Mapping[str, object]],
    *,
    cluster_field: str,
    value,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row[cluster_field])].append(float(value(row)))
    cluster_ids = sorted(grouped)
    observed = statistics.fmean(item for values in grouped.values() for item in values)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        sampled = rng.integers(0, len(cluster_ids), size=len(cluster_ids))
        values = [item for index in sampled for item in grouped[cluster_ids[index]]]
        estimates[replicate] = statistics.fmean(values)
    return {
        "estimate": observed,
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
        "replicates": replicates,
        "clusters": len(cluster_ids),
        "cluster_field": cluster_field,
        "seed": seed,
    }


def _reference_seconds(dataset: str, root: Path, reference_ids: set[str]) -> float:
    if dataset == "sdrr":
        path = root / "reference_manifest.csv"
        with path.open(encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))
        return sum(float(row["source_duration_seconds"]) for row in rows)
    path = root / "fma_tracks.csv"
    with path.open(encoding="utf-8-sig", newline="") as source:
        by_id = {row["track_id"].strip().zfill(6): row for row in csv.DictReader(source)}
    missing = reference_ids - set(by_id)
    if missing:
        raise EvaluationError(
            f"PEX fma_tracks.csv is missing reference durations: {sorted(missing)[:5]}"
        )
    return sum(float(by_id[reference_id]["track_duration"]) for reference_id in reference_ids)


def _index_summary(index: Path, reference_seconds: float) -> dict[str, object]:
    total_hashes = 0
    with (index / "songs.csv").open(encoding="utf-8", newline="") as source:
        for line in source:
            _, remainder = line.rstrip("\n").split(",", maxsplit=1)
            _, hashes = remainder.rsplit(",", maxsplit=1)
            total_hashes += int(hashes)
    return {
        "reference_audio_seconds": reference_seconds,
        "reference_hash_postings": total_hashes,
    }


def _summary(
    rows: Sequence[Mapping[str, object]],
    *,
    dataset: str,
    system: str,
    index_summary: Mapping[str, object],
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, object]:
    def track(row: Mapping[str, object]) -> bool:
        return row["predicted_reference_id"] == row["reference_id"]

    def track_offset(row: Mapping[str, object]) -> bool:
        return (
            track(row)
            and row.get("absolute_offset_error_seconds") not in (None, "")
            and float(row["absolute_offset_error_seconds"]) <= 0.1
        )

    cluster = "reference_id" if dataset == "sdrr" else "query_id"
    top1 = _cluster_ci(
        rows,
        cluster_field=cluster,
        value=track,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    offset = (
        _cluster_ci(
            rows,
            cluster_field=cluster,
            value=track_offset,
            replicates=bootstrap_replicates,
            seed=seed,
        )
        if dataset == "sdrr"
        else None
    )
    summary = {
        "schema": "polaris-paper-summary-v1",
        "dataset": dataset,
        "system": system,
        "configuration": DEFAULT_CONFIG.to_dict(),
        "trials": len(rows),
        "errors": sum(row["status"] == "error" for row in rows),
        "track_top1": top1,
        "track_and_offset_top1_at_0.1s": offset,
        "index": dict(index_summary),
    }
    if system == "polaris_adaptive":
        summary["full_expansion_avoided_fraction"] = sum(
            row.get("stopped_stage") not in {"full", "error"} for row in rows
        ) / len(rows)
    return summary


def _load_completed(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows or not set(RESULT_FIELDS).issubset(rows[0]):
        return {}
    return {row["trial_id"]: row for row in rows if row["status"] != "error"}


def _evaluate(
    *,
    dataset: str,
    root: Path,
    system: str,
    index: Path,
    output: Path,
    workers: int,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, object]:
    if dataset == "sdrr":
        all_items = load_sdrr(root)
        references = unique_sdrr_references(all_items)
        tasks: list[object] = list(all_items)
        task_ids = [item.query_id for item in all_items]
    else:
        annotations, references, queries = load_pex(root)
        exact = [annotation for annotation in annotations if annotation.exact_scale]
        grouped: dict[str, list[PexAnnotation]] = defaultdict(list)
        for annotation in exact:
            grouped[annotation.query_id].append(annotation)
        tasks = [(queries[query_id], tuple(grouped[query_id])) for query_id in sorted(grouped)]
        task_ids = [annotation.annotation_id for annotation in exact]
    index.mkdir(parents=True, exist_ok=True)
    configuration_path = index / "configuration.json"
    index_key = reference_key(system)
    expected_configuration = {
        "reference_key": index_key,
        "polaris": DEFAULT_CONFIG.to_dict(),
    }
    if configuration_path.is_file():
        if json.loads(configuration_path.read_text(encoding="utf-8")) != expected_configuration:
            raise EvaluationError(f"incompatible POLARIS index: {index}")
    inserted = (
        build_reference_index(
            list(references.values()),
            index,
            workers=workers,
            config=DEFAULT_CONFIG,
        )
        if index_key == "polaris"
        else build_variant_index(
            list(references.values()),
            index,
            system=system,
            workers=workers,
        )
    )
    if not configuration_path.is_file():
        write_json(expected_configuration, configuration_path)
    if inserted and has_packed_index(index):
        raise EvaluationError(f"packed index became stale after adding references: {index}")
    if not has_packed_index(index):
        build_packed_index(index)
    print(f"POLARIS index ready: {len(references)} references ({inserted} added)", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    write_json(
        {
            "system": system,
            "reference_key": index_key,
            "polaris": DEFAULT_CONFIG.to_dict(),
            "ablation": (
                None
                if system in {"polaris", "polaris_adaptive"}
                else {
                    "magnitude_maxima": "spectrogram-magnitude maxima replace saliency maxima",
                    "two_hop_only": "standard-density two-hop query expansion only",
                    "dense_only": "64-landmarks/s one-hop query pass only",
                    "canonical_only": "four-shift canonical Delaunay query only",
                    "target_region": "OLAF-style sequential triplets, usage cap 12",
                }[system]
            ),
        },
        output / "resolved_configuration.json",
    )
    results_path = output / "query_results.csv"
    existing = _load_completed(results_path)
    remaining_tasks = tasks
    if dataset == "sdrr":
        remaining_tasks = [task for task in tasks if task.query_id not in existing]  # type: ignore[attr-defined]
    else:
        remaining_tasks = [
            task
            for task in tasks
            if any(annotation.annotation_id not in existing for annotation in task[1])  # type: ignore[index]
        ]
    rows: dict[str, dict[str, object] | dict[str, str]] = dict(existing)
    worker_function = _sdrr_worker if dataset == "sdrr" else _pex_group_worker
    with multiprocessing.Pool(
        processes=workers,
        initializer=_initialize_worker,
        initargs=(str(index), system),
    ) as pool:
        iterator = pool.imap_unordered(worker_function, remaining_tasks, chunksize=1)
        for completed, result in enumerate(iterator, start=1):
            produced = [result] if dataset == "sdrr" else result
            for row in produced:
                rows[str(row["trial_id"])] = row
            if completed % 10 == 0 or completed == len(remaining_tasks):
                ordered = [rows[key] for key in sorted(rows)]
                write_csv(ordered, results_path, RESULT_FIELDS)
                print(
                    f"[{completed}/{len(remaining_tasks)}] {dataset}/{system}; "
                    f"saved {len(ordered)} trials",
                    flush=True,
                )

    expected_ids = set(task_ids)
    complete = expected_ids <= set(rows)
    if not complete or set(rows) != expected_ids:
        raise EvaluationError("result IDs do not exactly match the frozen protocol")
    ordered_rows = [rows[key] for key in sorted(expected_ids)]
    reference_seconds = _reference_seconds(dataset, root, set(references))
    summary = _summary(
        ordered_rows,
        dataset=dataset,
        system=system,
        index_summary=_index_summary(index, reference_seconds),
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    summary.update(
        {
            "dataset_root": str(root),
            "manifest_sha256": sha256(
                root / ("manifest.csv" if dataset == "sdrr" else "annotations.csv")
            ),
            "results": "query_results.csv",
        }
    )
    write_json(summary, output / "summary.json")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("sdrr", "pex"))
    parser.add_argument("--system", choices=SYSTEMS, action="append")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--workers", type=int, default=max(1, min(6, os.cpu_count() or 1)))
    parser.add_argument("--bootstrap-replicates", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    systems = tuple(dict.fromkeys(args.system or SYSTEMS))
    root = (
        args.data_root or Path("data") / ("sdrr" if args.dataset == "sdrr" else "pex_hard_medium")
    ).resolve()
    output_root = args.output_root.resolve() / args.dataset
    started = perf_counter()
    for system in systems:
        index = (
            args.output_root.resolve() / "shared" / args.dataset / f"{reference_key(system)}_index"
        )
        summary = _evaluate(
            dataset=args.dataset,
            root=root,
            system=system,
            index=index,
            output=output_root / system,
            workers=args.workers,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"elapsed: {perf_counter() - started:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
