"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

#!/usr/bin/env python3

# ruff: noqa: E402 -- repository imports follow an explicit sys.path bootstrap.

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
ADAPTER_DIRECTORY = Path(__file__).resolve().parent
for directory in (REPOSITORY_ROOT, ADAPTER_DIRECTORY):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from run_pex import (
    DEFAULT_SOURCE_DIRECTORY,
    EXPECTED_SOURCE_COMMIT,
    build_or_load_index,
    configure_analyzer,
    load_audfprint,
    source_commit,
)

from polaris_eval.io import EvaluationError
from polaris_eval.io import write_csv as write_csv_rows
from polaris_eval.io import write_json as write_summary
from polaris_eval.real import (
    RealQuery,
    file_sha256,
    load_real_manifest,
    summarize_rows,
    unique_references,
)


@dataclass(frozen=True)
class AudfprintProfile:
    name: str
    density: float
    fanout: int
    bucket_depth: int
    search_depth: int
    hashbits: int = 20
    sample_rate: int = 11_025
    query_shifts: int = 4
    match_window: int = 2
    # Closed-set protocol: return a candidate whenever one match exists.
    min_count: int = 1
    top_k: int = 10
    seed: int = 7
    query_density: float | None = None
    query_fanout: int | None = None
    query_max_peaks_per_frame: int | None = None
    reference_index_key: str | None = None


PROFILES = {
    "audfp_m": AudfprintProfile(
        name="audfp_m",
        density=28.0,
        fanout=4,
        bucket_depth=100,
        search_depth=100,
    ),
    "audfp_q": AudfprintProfile(
        name="audfp_q",
        density=28.0,
        fanout=4,
        bucket_depth=100,
        search_depth=2_000,
        query_density=1_440.0,
        query_fanout=86,
        query_max_peaks_per_frame=11,
        reference_index_key="audfp_m",
    ),
}

POSTING_MATCH_TARGET = 2_022_662

RESULT_FIELDS = (
    "profile",
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
class DecodedMatch:
    reference_id: str
    score: float
    offset_frames: int
    offset_seconds: float


_WORKER_ANALYZER: Any | None = None
_WORKER_MATCHER: Any | None = None
_WORKER_TABLE: Any | None = None
_WORKER_PROFILE: AudfprintProfile | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIRECTORY)
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    parser.add_argument(
        "--index-root",
        type=Path,
        help="reuse profile indexes from another output root",
    )
    parser.add_argument(
        "--query-workers",
        type=int,
        default=1,
        help="parallel query processes (default: 1)",
    )
    return parser.parse_args()


def profile_arguments(profile: AudfprintProfile) -> argparse.Namespace:
    """Return the argument shape expected by the shared Audfprint adapter."""

    return argparse.Namespace(**asdict(profile), index=None)


def configure_matcher(module: Any, profile: AudfprintProfile) -> Any:
    matcher = module.Matcher()
    matcher.window = profile.match_window
    matcher.threshcount = profile.min_count
    matcher.max_returns = profile.top_k
    matcher.search_depth = profile.search_depth
    matcher.exact_count = False
    return matcher


def configure_profile_analyzer(module: Any, profile: AudfprintProfile) -> Any:
    analyzer = configure_analyzer(module, profile_arguments(profile), query=True)
    if profile.query_max_peaks_per_frame is not None:
        analyzer.maxpksperframe = profile.query_max_peaks_per_frame
    return analyzer


def decode_matches(
    table: Any,
    matches: Any,
    *,
    seconds_per_frame: float,
    top_k: int,
) -> list[DecodedMatch]:
    """Decode Audfprint rows and retain the best alignment per reference.

    ``match_hashes`` returns rows of ``(id, filtered_count, time_skew, ...)``.
    The time skew is reference-frame time minus query-frame time, so it is the
    desired reference offset for a query that begins at query time zero.
    """

    ranking = []
    seen_references = set()
    for match in matches:
        reference_id = Path(str(table.names[int(match[0])])).stem
        if reference_id in seen_references:
            continue
        seen_references.add(reference_id)
        offset_frames = int(match[2])
        ranking.append(
            DecodedMatch(
                reference_id=reference_id,
                score=float(match[1]),
                offset_frames=offset_frames,
                offset_seconds=offset_frames * seconds_per_frame,
            )
        )
        if len(ranking) == top_k:
            break
    return ranking


def _common_result(item: RealQuery, profile: AudfprintProfile) -> dict[str, object]:
    return {
        "profile": profile.name,
        "query_id": item.query_id,
        "reference_id": item.reference_id,
        "query_seconds": item.query_duration_seconds,
        "ground_truth_offset_seconds": item.reference_begin_seconds,
    }


def evaluate_query(
    analyzer: Any,
    matcher: Any,
    table: Any,
    item: RealQuery,
    profile: AudfprintProfile,
) -> dict[str, object]:
    """Fingerprint, retrieve, and score one real recording query."""

    common = _common_result(item, profile)
    try:
        feature_started = time.perf_counter()
        query_hashes = analyzer.wavfile2hashes(str(item.query_path))
        feature_seconds = time.perf_counter() - feature_started

        lookup_started = time.perf_counter()
        raw_matches = matcher.match_hashes(table, query_hashes)
        lookup_seconds = time.perf_counter() - lookup_started
        ranking = decode_matches(
            table,
            raw_matches,
            seconds_per_frame=analyzer.n_hop / analyzer.target_sr,
            top_k=profile.top_k,
        )
        predicted = ranking[0] if ranking else None
        top1_error = (
            predicted.offset_seconds - item.reference_begin_seconds
            if predicted is not None and predicted.reference_id == item.reference_id
            else None
        )
        row: dict[str, object] = {
            **common,
            "status": "matched" if ranking else "no_match",
            "predicted_reference_id": predicted.reference_id if predicted else "",
            "predicted_offset_seconds": predicted.offset_seconds if predicted else None,
            "top1_absolute_offset_error_seconds": (
                abs(top1_error) if top1_error is not None else None
            ),
            "query_fingerprints": len(query_hashes),
            "total_time": feature_seconds + lookup_seconds,
            "error": "",
        }
        return row
    except Exception as exc:
        return {
            **common,
            "status": "error",
            "predicted_reference_id": "",
            "predicted_offset_seconds": None,
            "top1_absolute_offset_error_seconds": None,
            "query_fingerprints": 0,
            "total_time": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _initialize_query_worker(
    source_directory: str,
    index_path: str,
    profile_configuration: dict[str, object],
) -> None:
    """Load an independent Audfprint analyzer, matcher, and table per process."""

    global _WORKER_ANALYZER, _WORKER_MATCHER, _WORKER_TABLE, _WORKER_PROFILE
    analyze_module, match_module, table_module = load_audfprint(Path(source_directory))
    profile = AudfprintProfile(**profile_configuration)
    _WORKER_ANALYZER = configure_profile_analyzer(analyze_module, profile)
    _WORKER_MATCHER = configure_matcher(match_module, profile)
    _WORKER_TABLE = table_module.HashTable(index_path)
    _WORKER_PROFILE = profile


def _evaluate_query_worker(item: RealQuery) -> dict[str, object]:
    if any(
        value is None
        for value in (
            _WORKER_ANALYZER,
            _WORKER_MATCHER,
            _WORKER_TABLE,
            _WORKER_PROFILE,
        )
    ):
        raise RuntimeError("Audfprint query worker was not initialized")
    return evaluate_query(
        _WORKER_ANALYZER,
        _WORKER_MATCHER,
        _WORKER_TABLE,
        item,
        _WORKER_PROFILE,
    )


def run_profile(
    profile: AudfprintProfile,
    items: list[RealQuery],
    manifest: Path,
    output: Path,
    source_directory: Path,
    analyze_module: Any,
    match_module: Any,
    table_module: Any,
    *,
    references: dict[str, Path],
    index_root: Path | None,
    query_workers: int,
) -> dict[str, object]:
    profile_output = output
    profile_output.mkdir(parents=True, exist_ok=True)
    adapter_args = profile_arguments(profile)
    index_key = profile.reference_index_key or profile.name
    index_output = index_root / index_key if index_root is not None else profile_output
    table, index_path, build_metadata = build_or_load_index(
        adapter_args,
        references,
        analyze_module,
        table_module,
        index_output,
    )
    rows = []
    if query_workers == 1:
        analyzer = configure_profile_analyzer(analyze_module, profile)
        matcher = configure_matcher(match_module, profile)
        iterator = (evaluate_query(analyzer, matcher, table, item, profile) for item in items)
        pool = None
    else:
        pool = multiprocessing.Pool(
            processes=query_workers,
            initializer=_initialize_query_worker,
            initargs=(str(source_directory), str(index_path), asdict(profile)),
        )
        iterator = pool.imap(_evaluate_query_worker, items, chunksize=1)
    try:
        for number, (_item, row) in enumerate(zip(items, iterator, strict=True), start=1):
            rows.append(row)
            if number % 10 == 0 or number == len(items):
                print(f"[{profile.name} {number}/{len(items)}]", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    results_path = profile_output / "query_results.csv"
    write_csv_rows(rows, results_path, RESULT_FIELDS)
    stored_postings = int(np.sum(np.minimum(table.depth, table.counts), dtype=np.int64))
    summary = {
        "protocol": "real_phone_closed_set_retrieval_and_offset_localization",
        "system": "audfprint",
        "profile": profile.name,
        "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "configuration": asdict(profile),
        "profile_selection": (
            {
                "criterion": "reference_postings_only",
                "target_reference_postings": POSTING_MATCH_TARGET,
                "query_labels_used": False,
                "note": (
                    "density and fanout were selected from reference audio only; "
                    "all other settings remain at stock defaults"
                ),
            }
            if profile.name == "audfp_m"
            else (
                {
                    "criterion": "query_payload_matching",
                    "shared_reference_profile": profile.reference_index_key,
                    "target_reference_postings": POSTING_MATCH_TARGET,
                    "query_labels_used": False,
                    "note": (
                        "query density, fanout, and peak cap were selected from "
                        "query evidence counts without recognition labels"
                    ),
                }
                if profile.name == "audfp_q"
                else None
            )
        ),
        "source": {
            "repository": "https://github.com/dpwe/audfprint",
            "directory": str(source_directory),
            "commit": source_commit(source_directory),
            "expected_commit": EXPECTED_SOURCE_COMMIT,
        },
        "offset_definition": (
            "Audfprint modal time_skew frames multiplied by n_hop/sample_rate; "
            "ground truth is reference_begin_seconds from the real-query manifest"
        ),
        **summarize_rows(rows),
        "index": {
            "path": str(index_path),
            "stored_postings": stored_postings,
            **build_metadata,
        },
        "files": {"query_results_csv": str(results_path)},
    }
    summary_path = profile_output / "summary.json"
    write_summary(summary, summary_path)
    print(f"{profile.name} results: {results_path}")
    print(f"{profile.name} summary: {summary_path}")
    return summary


def main() -> None:
    args = parse_args()
    if args.query_workers <= 0:
        raise SystemExit("--query-workers must be positive")
    manifest = args.manifest.expanduser().resolve()
    output = args.output.expanduser().resolve()
    source_directory = args.source_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        all_items = load_real_manifest(manifest)
        references = unique_references(all_items)
        items = list(all_items)
        analyze_module, match_module, table_module = load_audfprint(source_directory)
    except (EvaluationError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

    summary = run_profile(
        PROFILES[args.profile],
        items,
        manifest,
        output,
        source_directory,
        analyze_module,
        match_module,
        table_module,
        references=references,
        index_root=args.index_root.expanduser().resolve() if args.index_root else None,
        query_workers=args.query_workers,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
