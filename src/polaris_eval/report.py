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
from polaris_eval.metrics import PEX_RESULT_FIELDS, SDRR_RESULT_FIELDS


@dataclass(frozen=True)
class ResultSource:
    dataset: str
    system: str
    family: str

    @property
    def summary(self) -> str:
        return f"{self.dataset}/{self.system}/summary.json"

    @property
    def rows(self) -> str:
        return f"{self.dataset}/{self.system}/query_results.csv"


BASELINES = (
    ("audfp_m", "hash"),
    ("audfp_q", "hash"),
    ("panako", "panako"),
    ("olaf", "hash"),
    ("nmfp", "nmfp"),
)
POLARIS = (("polaris_o", "hash"), ("polaris_a", "hash"), ("polaris_f", "hash"))
CONTROLS = (("magnitude_maxima", "hash"), ("target_region", "hash"))
PROTOCOLS = {
    "pex": "pex_oracle_segment_exact_scale_v1",
    "sdrr": "sdrr_closed_set_retrieval_and_offset_v1",
}
SOURCES = tuple(
    ResultSource(dataset, system, family)
    for dataset in ("pex", "sdrr")
    for system, family in (*BASELINES, *POLARIS, *(CONTROLS if dataset == "sdrr" else ()))
)

CONTRASTS = {
    "pex": (
        *(f"polaris_{mode}:audfp_m" for mode in "oaf"),
        *(f"polaris_{mode}:panako" for mode in "oaf"),
        *(f"polaris_{mode}:olaf" for mode in "oaf"),
        "polaris_f:polaris_o",
    ),
    "sdrr": (
        "polaris_o:audfp_m",
        "polaris_o:audfp_q",
        "polaris_o:panako",
        "polaris_o:olaf",
        "polaris_o:nmfp",
        "polaris_a:nmfp",
        "polaris_f:audfp_q",
        "polaris_f:magnitude_maxima",
        "polaris_o:target_region",
        "polaris_f:polaris_o",
    ),
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def _uses_query_stage(system: str) -> bool:
    return system.startswith("polaris_")


def _top1(summary: Mapping[str, object]) -> float:
    return float(summary["track_top1"])


def _offset(summary: Mapping[str, object]) -> float:
    return float(summary["track_and_offset_top1_at_0.1s"])


def _index(summary: Mapping[str, object], family: str) -> tuple[int, int]:
    index = summary.get("index")
    if not isinstance(index, Mapping):
        raise ValueError("summary has no index metadata")
    if family == "nmfp":
        return int(index["reference_embeddings"]), 520
    if family == "panako":
        return int(index["stored_fingerprints"]), 20
    if family == "hash":
        for field in ("reference_hash_postings", "stored_postings", "stored_fingerprints"):
            if field in index:
                return int(index[field]), 16
    raise ValueError("unknown current index schema")


def _mean_query_payload(
    rows: Sequence[Mapping[str, str]],
    family: str,
) -> float | None:
    if family == "panako":
        return None
    values = [float(row["query_records"]) for row in rows if row.get("query_records") not in (None, "")]
    if not values:
        return None
    count = statistics.fmean(values)
    bytes_per_record = 516 if family == "nmfp" else 12
    return count * bytes_per_record / (1024**2)


def _mean_runtime(rows: Sequence[Mapping[str, str]]) -> float | None:
    values = [float(row["total_time"]) for row in rows if row.get("total_time") not in (None, "")]
    return statistics.fmean(values) if values else None


def _trial_key(dataset: str, row: Mapping[str, str]) -> tuple[str, ...]:
    if dataset == "sdrr":
        return (row["query_id"],)
    return (row["trial_id"],)


def _correct(row: Mapping[str, str]) -> int:
    return int(row["predicted_reference_id"] == row["reference_id"])


def _offset_correct(row: Mapping[str, str]) -> int:
    error = row.get("absolute_offset_error_seconds")
    return int(_correct(row) and error not in (None, "") and float(error) <= 0.1)


def _protocol_reference_seconds(root: Path, dataset: str) -> float:
    path = root / dataset / "polaris_o" / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"POLARIS-O summary is required: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    value = float(summary["index"]["reference_audio_seconds"])
    if value <= 0:
        raise ValueError(f"invalid reference duration in {path}")
    return value


def cluster_bootstrap(
    rows: Sequence[Mapping[str, str]],
    *,
    dataset: str,
    metric,
    replicates: int = 5_000,
    seed: int = 7,
) -> dict[str, float]:
    field = "query_id" if dataset == "pex" else "reference_id"
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row[field]].append(float(metric(row)))
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


def paired_comparison(
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
        raise ValueError("paired systems do not contain identical trials")
    grouped: dict[str, list[float]] = defaultdict(list)
    for key in sorted(left_by_key):
        cluster_field = "query_id" if dataset == "pex" else "reference_id"
        cluster = left_by_key[key][cluster_field]
        grouped[cluster].append(metric(left_by_key[key]) - metric(right_by_key[key]))
    clusters = sorted(grouped)
    difference = statistics.fmean(value for values in grouped.values() for value in values)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(replicates)
    for replicate in range(replicates):
        sample = rng.integers(0, len(clusters), size=len(clusters))
        bootstrap[replicate] = statistics.fmean(
            value for index in sample for value in grouped[clusters[index]]
        )
    cluster_totals = np.asarray([sum(grouped[cluster]) for cluster in clusters])
    observed = abs(float(np.sum(cluster_totals)))
    signs = rng.choice((-1.0, 1.0), size=(replicates, len(clusters)))
    permuted = np.abs(signs @ cluster_totals)
    return {
        "difference": difference,
        "lower_95": float(np.quantile(bootstrap, 0.025)),
        "upper_95": float(np.quantile(bootstrap, 0.975)),
        "two_sided_permutation_p": float(
            (np.count_nonzero(permuted >= observed) + 1) / (replicates + 1)
        ),
        "clusters": len(clusters),
        "trials": len(left_by_key),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the paper result tables.")
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    root = args.outputs.resolve()
    sdrr_reference_seconds = _protocol_reference_seconds(root, "sdrr")
    records = []
    rows_by_system: dict[tuple[str, str], list[dict[str, str]]] = {}
    for source in SOURCES:
        summary_path = root / source.summary
        rows_path = root / source.rows
        if not summary_path.is_file() or not rows_path.is_file():
            raise FileNotFoundError(
                f"complete paper matrix required; missing {summary_path} or {rows_path}"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("schema") != "polaris-paper-results-v1":
            raise ValueError(f"unsupported result schema: {summary_path}")
        if summary.get("system") != source.system:
            raise ValueError(f"unexpected system ID in {summary_path}")
        if summary.get("protocol") != PROTOCOLS[source.dataset]:
            raise ValueError(f"unexpected protocol ID in {summary_path}")
        rows = read_rows(rows_path)
        base_fields = SDRR_RESULT_FIELDS if source.dataset == "sdrr" else PEX_RESULT_FIELDS
        expected_fields = set(base_fields)
        if source.family == "panako":
            expected_fields.discard("query_records")
        if _uses_query_stage(source.system):
            expected_fields.add("query_stage")
        if not rows or set(rows[0]) != expected_fields:
            raise ValueError(f"unexpected result columns: {rows_path}")
        expected_trials = 791 if source.dataset == "pex" else 1_488
        if len(rows) != expected_trials or int(summary["trials"]) != expected_trials:
            raise ValueError(f"incomplete result set: {source.dataset}/{source.system}")
        if int(summary["errors"]) != 0:
            raise ValueError(f"failed queries remain: {source.dataset}/{source.system}")
        if source.dataset == "sdrr":
            count, record_bytes = _index(summary, source.family)
            reference_payload = (
                count
                * record_bytes
                / (1024**2)
                / (sdrr_reference_seconds / 3600)
            )
            query_payload = _mean_query_payload(rows, source.family)
            query_time = _mean_runtime(rows)
        else:
            reference_payload = None
            query_payload = None
            query_time = None
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
                "track_and_offset_top1_at_0.1s": (
                    _offset(summary) if source.dataset == "sdrr" else None
                ),
                "track_and_offset_lower_95": offset_ci["lower_95"] if offset_ci else None,
                "track_and_offset_upper_95": offset_ci["upper_95"] if offset_ci else None,
                "reference_payload_mib_per_hour": reference_payload,
                "mean_query_payload_mib": query_payload,
                "mean_query_time_seconds": query_time,
            }
        )
        rows_by_system[(source.dataset, source.system)] = rows
    write_csv(records, root / "results.csv", tuple(records[0]) if records else ())

    comparisons = []
    for dataset, contrasts in CONTRASTS.items():
        for contrast in contrasts:
            left_name, right_name = contrast.split(":")
            left = rows_by_system.get((dataset, left_name))
            right = rows_by_system.get((dataset, right_name))
            if left is None or right is None:
                continue
            metrics = (("track_top1", _correct),)
            if dataset == "sdrr":
                metrics += (("track_and_offset_top1_at_0.1s", _offset_correct),)
            for metric_name, metric in metrics:
                comparisons.append(
                    {
                        "dataset": dataset,
                        "left": left_name,
                        "right": right_name,
                        "metric": metric_name,
                        **paired_comparison(left, right, dataset=dataset, metric=metric),
                    }
                )
    write_csv(
        comparisons,
        root / "comparisons.csv",
        tuple(comparisons[0]) if comparisons else (),
    )
    print(f"wrote {len(records)} result rows and {len(comparisons)} comparisons")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
