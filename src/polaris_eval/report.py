"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from polaris_eval.io import write_csv


@dataclass(frozen=True)
class ResultSource:
    dataset: str
    system: str
    family: str
    summary: str
    rows: str


SOURCES = (
    ResultSource(
        "pex",
        "audfp_m",
        "hash",
        "pex/audfp_m/summary.json",
        "pex/audfp_m/query_results.csv",
    ),
    ResultSource(
        "pex",
        "audfp_q",
        "hash",
        "pex/audfp_q/summary.json",
        "pex/audfp_q/query_results.csv",
    ),
    ResultSource(
        "pex", "panako", "panako", "pex/panako/summary.json", "pex/panako/query_results.csv"
    ),
    ResultSource(
        "pex",
        "olaf",
        "hash",
        "pex/olaf/summary.json",
        "pex/olaf/query_results.csv",
    ),
    ResultSource(
        "pex",
        "nmfp",
        "nmfp",
        "pex/nmfp/summary.json",
        "pex/nmfp/query_results.csv",
    ),
    ResultSource(
        "pex",
        "polaris_adaptive",
        "hash",
        "pex/polaris_adaptive/summary.json",
        "pex/polaris_adaptive/query_results.csv",
    ),
    ResultSource(
        "pex",
        "polaris",
        "hash",
        "pex/polaris/summary.json",
        "pex/polaris/query_results.csv",
    ),
    ResultSource(
        "sdrr",
        "audfp_m",
        "hash",
        "sdrr/audfp_m/summary.json",
        "sdrr/audfp_m/query_results.csv",
    ),
    ResultSource(
        "sdrr",
        "audfp_q",
        "hash",
        "sdrr/audfp_q/summary.json",
        "sdrr/audfp_q/query_results.csv",
    ),
    ResultSource(
        "sdrr",
        "panako",
        "panako",
        "sdrr/panako/summary.json",
        "sdrr/panako/query_results.csv",
    ),
    ResultSource(
        "sdrr",
        "olaf",
        "hash",
        "sdrr/olaf/summary.json",
        "sdrr/olaf/query_results.csv",
    ),
    ResultSource(
        "sdrr",
        "nmfp",
        "nmfp",
        "sdrr/nmfp/summary.json",
        "sdrr/nmfp/query_results.csv",
    ),
    ResultSource(
        "sdrr",
        "polaris_adaptive",
        "hash",
        "sdrr/polaris_adaptive/summary.json",
        "sdrr/polaris_adaptive/query_results.csv",
    ),
    ResultSource(
        "sdrr",
        "polaris",
        "hash",
        "sdrr/polaris/summary.json",
        "sdrr/polaris/query_results.csv",
    ),
    *(
        ResultSource(
            "sdrr", name, "hash", f"sdrr/{name}/summary.json", f"sdrr/{name}/query_results.csv"
        )
        for name in (
            "magnitude_maxima",
            "two_hop_only",
            "dense_only",
            "canonical_only",
            "target_region",
        )
    ),
)


def nested(value: Mapping[str, object], *path: str) -> object | None:
    current: object = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def first(value: Mapping[str, object], paths: Sequence[tuple[str, ...]]) -> object | None:
    for path in paths:
        result = nested(value, *path)
        if result is not None:
            return result
    return None


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def _top1(summary: Mapping[str, object]) -> float:
    value = first(
        summary,
        (
            ("track_top1", "estimate"),
            ("track_top1",),
            ("paper_metrics", "track_top1"),
        ),
    )
    if value is None:
        raise ValueError("summary has no Top-1 metric")
    return float(value)


def _offset(summary: Mapping[str, object]) -> float | None:
    value = first(
        summary,
        (
            ("track_and_offset_top1_at_0.1s", "estimate"),
            ("track_and_offset_top1_at_0.1s",),
        ),
    )
    return float(value) if value is not None else None


def _index(summary: Mapping[str, object], family: str) -> tuple[int, float, int]:
    index = summary.get("index")
    if not isinstance(index, Mapping):
        raise ValueError("summary has no index metadata")
    if family == "nmfp":
        count = int(index["reference_embeddings"])
        seconds = float(
            index.get("reference_audio_seconds_extracted_this_run")
            or index.get("reference_audio_seconds")
            or 0
        )
        return count, seconds, 520
    count = int(
        index.get("reference_hash_postings")
        or index.get("stored_postings")
        or index.get("stored_fingerprints")
    )
    seconds = float(index.get("reference_audio_seconds") or index.get("audio_seconds") or 0)
    return count, seconds, 20 if family == "panako" else 16


def _mean_query_evidence(
    rows: Sequence[Mapping[str, str]], family: str
) -> tuple[float | None, float | None]:
    if family == "panako":
        return None, None
    if family == "nmfp":
        field = "query_embeddings"
    else:
        field = "query_fingerprints"
    values = [float(row[field]) for row in rows if row.get(field) not in (None, "")]
    if not values:
        return None, None
    mean = statistics.fmean(values)
    bytes_per = 516 if family == "nmfp" else 12
    return mean, mean * bytes_per / (1024**2)


def _mean_runtime(rows: Sequence[Mapping[str, str]]) -> float | None:
    values = [float(row["total_time"]) for row in rows if row.get("total_time") not in (None, "")]
    return statistics.fmean(values) if values else None


def _trial_key(dataset: str, row: Mapping[str, str]) -> tuple[str, ...]:
    if dataset == "sdrr":
        return (row["query_id"],)
    reference_id = row.get("reference_id") or row.get("expected_reference_id")
    if not reference_id:
        raise ValueError("PEX result row has no expected reference ID")
    begin = row.get("query_begin")
    if begin in (None, ""):
        trial_id = row.get("trial_id") or row.get("annotation_id") or ""
        parts = trial_id.split(":")
        if len(parts) >= 3:
            begin = parts[2] if parts[0].startswith("query") else parts[-2]
    return (row["query_id"], reference_id, str(float(begin or 0)))


def _correct(row: Mapping[str, str]) -> int:
    expected = row.get("reference_id") or row.get("expected_reference_id")
    return int(bool(expected) and row.get("predicted_reference_id") == expected)


def _protocol_reference_seconds(root: Path, dataset: str) -> float | None:
    """Return the common corpus duration used for every method's payload rate.

    The native systems disagree slightly about decoder padding and failures.
    A payload-per-hour comparison must use one denominator, so the compact
    POLARIS index summary supplies the frozen protocol duration.
    """

    path = root / dataset / "polaris" / "summary.json"
    if not path.is_file():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    value = nested(summary, "index", "reference_audio_seconds")
    return float(value) if value is not None and float(value) > 0 else None


def _offset_correct(row: Mapping[str, str]) -> int:
    value = row.get("top1_absolute_offset_error_seconds")
    if value in (None, ""):
        value = row.get("absolute_offset_error_seconds")
    return int(_correct(row) and value not in (None, "") and float(value) <= 0.1)


def paired_bootstrap(
    left: Sequence[Mapping[str, str]],
    right: Sequence[Mapping[str, str]],
    *,
    dataset: str,
    metric,
    replicates: int = 5_000,
    seed: int = 7,
) -> dict[str, object]:
    left_by_key = {_trial_key(dataset, row): row for row in left}
    right_by_key = {_trial_key(dataset, row): row for row in right}
    if set(left_by_key) != set(right_by_key):
        raise ValueError("paired result rows do not have identical trial keys")
    grouped: dict[str, list[float]] = defaultdict(list)
    for key in sorted(left_by_key):
        cluster = left_by_key[key]["query_id" if dataset == "pex" else "reference_id"]
        grouped[cluster].append(metric(left_by_key[key]) - metric(right_by_key[key]))
    clusters = sorted(grouped)
    observed = statistics.fmean(value for values in grouped.values() for value in values)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates)
    for replicate in range(replicates):
        sample = rng.integers(0, len(clusters), size=len(clusters))
        estimates[replicate] = statistics.fmean(
            value for index in sample for value in grouped[clusters[index]]
        )
    cluster_totals = np.asarray([sum(grouped[cluster]) for cluster in clusters])
    observed_total = abs(float(np.sum(cluster_totals)))
    signs = rng.choice((-1.0, 1.0), size=(replicates, len(clusters)))
    permuted = np.abs(signs @ cluster_totals)
    return {
        "difference": observed,
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
        "two_sided_permutation_p": float(
            (np.count_nonzero(permuted >= observed_total) + 1) / (replicates + 1)
        ),
        "clusters": len(clusters),
        "trials": len(left_by_key),
        "replicates": replicates,
        "seed": seed,
    }


def cluster_bootstrap(
    rows: Sequence[Mapping[str, str]],
    *,
    dataset: str,
    metric,
    replicates: int = 5_000,
    seed: int = 7,
) -> dict[str, float]:
    cluster_field = "query_id" if dataset == "pex" else "reference_id"
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row[cluster_field]].append(float(metric(row)))
    clusters = sorted(grouped)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates)
    for replicate in range(replicates):
        sample = rng.integers(0, len(clusters), size=len(clusters))
        estimates[replicate] = statistics.fmean(
            value for index in sample for value in grouped[clusters[index]]
        )
    return {
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    root = args.outputs.resolve()
    protocol_seconds = {
        dataset: _protocol_reference_seconds(root, dataset) for dataset in ("pex", "sdrr")
    }
    records = []
    rows_by_system: dict[tuple[str, str], list[dict[str, str]]] = {}
    missing = []
    for source in SOURCES:
        summary_path = root / source.summary
        rows_path = root / source.rows
        if not summary_path.is_file() or not rows_path.is_file():
            missing.append({"dataset": source.dataset, "system": source.system})
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = read_rows(rows_path)
        count, native_seconds, record_bytes = _index(summary, source.family)
        seconds = protocol_seconds[source.dataset] or native_seconds
        if seconds <= 0:
            raise ValueError(
                f"cannot determine reference duration for {source.dataset}/{source.system}"
            )
        _, query_mib = _mean_query_evidence(rows, source.family)
        track_ci = cluster_bootstrap(rows, dataset=source.dataset, metric=_correct)
        offset_ci = (
            cluster_bootstrap(rows, dataset=source.dataset, metric=_offset_correct)
            if source.dataset == "sdrr"
            else None
        )
        records.append(
            {
                "dataset": source.dataset,
                "system": source.system,
                "trials": len(rows),
                "track_top1": _top1(summary),
                "track_top1_lower_95": track_ci["lower_95"],
                "track_top1_upper_95": track_ci["upper_95"],
                "track_and_offset_top1_at_0.1s": _offset(summary),
                "track_and_offset_lower_95": offset_ci["lower_95"] if offset_ci else None,
                "track_and_offset_upper_95": offset_ci["upper_95"] if offset_ci else None,
                "reference_payload_mib_per_hour": count
                * record_bytes
                / (1024**2)
                / (seconds / 3600),
                "mean_query_payload_mib": query_mib,
                "mean_runtime_seconds": _mean_runtime(rows),
            }
        )
        rows_by_system[(source.dataset, source.system)] = rows
    write_csv(records, root / "results.csv", tuple(records[0]) if records else ())

    comparisons = []
    for (dataset, system), rows in rows_by_system.items():
        if system == "polaris":
            continue
        full = rows_by_system.get((dataset, "polaris"))
        if full is None:
            continue
        for metric_name, metric in (
            ("track_top1", _correct),
            ("track_and_offset_top1_at_0.1s", _offset_correct),
        ):
            if dataset == "pex" and metric_name.startswith("track_and_offset"):
                continue
            result = paired_bootstrap(full, rows, dataset=dataset, metric=metric)
            comparisons.append(
                {
                    "dataset": dataset,
                    "left": "polaris",
                    "right": system,
                    "metric": metric_name,
                    **result,
                }
            )
    write_csv(
        comparisons,
        root / "comparisons.csv",
        tuple(comparisons[0]) if comparisons else (),
    )
    print(
        f"wrote {len(records)} result rows and {len(comparisons)} paired comparisons; "
        f"missing={len(missing)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
