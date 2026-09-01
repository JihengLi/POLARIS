"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

#!/usr/bin/env python3

# ruff: noqa: E402 -- repository imports follow an explicit sys.path bootstrap.

from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from pydub import AudioSegment

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from polaris_eval.datasets import load_pex as load_pex_dataset
from polaris_eval.io import EvaluationError
from polaris_eval.io import write_csv as write_csv_rows
from polaris_eval.io import write_json as write_summary
from polaris_eval.metrics import PEX_RESULT_FIELDS, summarize_pex
from polaris_eval.protocol import OracleSegment, build_oracle_segments, group_oracle_segments

ADAPTER_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIRECTORY = ADAPTER_DIRECTORY / ".cache" / "audfprint"
EXPECTED_SOURCE_COMMIT = "cb03ba99feafd41b8874307f0f4e808a6ce34362"
TRIAL_FIELDS = PEX_RESULT_FIELDS

_WORKER_ANALYZER: Any | None = None
_WORKER_ANALYZE_MODULE: Any | None = None
_WORKER_MATCHER: Any | None = None
_WORKER_TABLE: Any | None = None
_WORKER_SAMPLE_RATE: int | None = None

PROFILES = {
    "audfp_m": {
        "query_density": 28.0,
        "query_fanout": 4,
        "query_max_peaks_per_frame": None,
        "search_depth": 100,
    },
    "audfp_q": {
        "query_density": 504.0,
        "query_fanout": 30,
        "query_max_peaks_per_frame": 11,
        "search_depth": 2000,
    },
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Audfprint on PEX Hard Medium.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIRECTORY)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--profile", required=True, choices=tuple(PROFILES))
    parser.add_argument(
        "--query-workers",
        type=int,
        default=1,
        help="Parallel query-file processes (default: 1).",
    )
    args = parser.parse_args()
    args.density = 28.0
    args.fanout = 4
    args.hashbits = 20
    args.bucket_depth = 100
    args.sample_rate = 11_025
    args.query_shifts = 4
    args.match_window = 2
    args.min_count = 1
    args.top_k = 10
    args.seed = 7
    for key, value in PROFILES[args.profile].items():
        setattr(args, key, value)
    return args


def validate_arguments(args: argparse.Namespace) -> None:
    if args.query_workers <= 0:
        raise SystemExit("--query-workers must be positive")


def load_audfprint(source_directory: Path) -> tuple[Any, Any, Any]:
    source_directory = source_directory.expanduser().resolve()
    required = ("audfprint_analyze.py", "audfprint_match.py", "hash_table.py")
    missing = [name for name in required if not (source_directory / name).is_file()]
    if missing:
        raise EvaluationError(
            f"Audfprint source is incomplete at {source_directory}; missing {', '.join(missing)}"
        )
    sys.path.insert(0, str(source_directory))
    return (
        importlib.import_module("audfprint_analyze"),
        importlib.import_module("audfprint_match"),
        importlib.import_module("hash_table"),
    )


def source_commit(source_directory: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(source_directory), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def configure_analyzer(module: Any, args: argparse.Namespace, *, query: bool) -> Any:
    density = args.query_density if query and args.query_density is not None else args.density
    fanout = args.query_fanout if query and args.query_fanout is not None else args.fanout
    analyzer = module.Analyzer(density=density)
    analyzer.target_sr = args.sample_rate
    analyzer.n_fft = 512
    analyzer.n_hop = 256
    analyzer.maxpairsperpeak = fanout
    analyzer.shifts = args.query_shifts if query else 1
    if query and args.query_max_peaks_per_frame is not None:
        analyzer.maxpksperframe = args.query_max_peaks_per_frame
    analyzer.fail_on_error = True
    return analyzer


def samples_to_hashes(analyzer: Any, analyze_module: Any, samples: np.ndarray) -> np.ndarray:
    peak_lists = []
    for shift in range(analyzer.shifts):
        shift_samples = int(shift / analyzer.shifts * analyzer.n_hop)
        peak_lists.append(analyzer.find_peaks(samples[shift_samples:], analyzer.target_sr))
    hashes = []
    for peaks in peak_lists:
        if peaks:
            hashes.append(analyze_module.landmarks2hashes(analyzer.peaks2landmarks(peaks)))
    if not hashes:
        return np.empty((0, 2), dtype=np.int32)
    combined = np.concatenate(hashes)
    packed = (combined[:, 0].astype(np.uint64) << 32) + combined[:, 1].astype(np.uint64)
    unique = np.sort(np.unique(packed))
    return np.column_stack((unique >> 32, unique & ((1 << 32) - 1))).astype(np.int32)


def read_mono_audio(path: Path, sample_rate: int) -> np.ndarray:
    audio = AudioSegment.from_file(str(path))
    audio = audio.set_sample_width(2).set_frame_rate(sample_rate).set_channels(1)
    return np.frombuffer(audio.raw_data, dtype=np.int16)


def configure_matcher(module: Any, args: argparse.Namespace) -> Any:
    matcher = module.Matcher()
    matcher.window = args.match_window
    matcher.threshcount = args.min_count
    matcher.max_returns = args.top_k
    matcher.search_depth = args.search_depth
    matcher.exact_count = False
    return matcher


def evaluate_query_group(
    analyzer: Any,
    matcher: Any,
    table: Any,
    analyze_module: Any,
    task: tuple[
        str,
        Path,
        list[OracleSegment],
    ],
    sample_rate: int,
) -> list[dict[str, object]]:
    """Evaluate all oracle segments belonging to one PEX query file."""

    query_id, query_path, query_segments = task
    samples = read_mono_audio(query_path, sample_rate)
    rows: list[dict[str, object]] = []
    for segment in query_segments:
        annotation = segment.annotation
        start_sample = max(0, round(segment.begin * sample_rate))
        end_sample = min(len(samples), round(segment.end * sample_rate))
        excerpt = samples[start_sample:end_sample]
        feature_started = time.perf_counter()
        query_hashes = samples_to_hashes(analyzer, analyze_module, excerpt)
        feature_seconds = time.perf_counter() - feature_started
        lookup_started = time.perf_counter()
        matches = matcher.match_hashes(table, query_hashes)
        lookup_seconds = time.perf_counter() - lookup_started
        ranking = [(str(table.names[int(match[0])]), float(match[1])) for match in matches]
        predicted_id = Path(ranking[0][0]).stem if ranking else ""
        rows.append(
            {
                "trial_id": segment.trial_id,
                "query_id": query_id,
                "reference_id": annotation.reference_id,
                "query_begin": segment.begin,
                "status": "matched" if ranking else "no_match",
                "predicted_reference_id": predicted_id,
                "total_time": feature_seconds + lookup_seconds,
                "error": "",
            }
        )
    return rows


def initialize_query_worker(
    source_directory: str,
    index_path: str,
    configuration: dict[str, object],
) -> None:
    """Load an independent analyzer, matcher, and index in each process."""

    global _WORKER_ANALYZER, _WORKER_ANALYZE_MODULE
    global _WORKER_MATCHER, _WORKER_TABLE, _WORKER_SAMPLE_RATE
    args = argparse.Namespace(**configuration)
    analyze_module, match_module, table_module = load_audfprint(Path(source_directory))
    _WORKER_ANALYZER = configure_analyzer(analyze_module, args, query=True)
    _WORKER_ANALYZE_MODULE = analyze_module
    _WORKER_MATCHER = configure_matcher(match_module, args)
    _WORKER_TABLE = table_module.HashTable(index_path)
    _WORKER_SAMPLE_RATE = args.sample_rate


def evaluate_query_worker(
    task: tuple[str, Path, list[OracleSegment]],
) -> list[dict[str, object]]:
    if any(
        value is None
        for value in (
            _WORKER_ANALYZER,
            _WORKER_ANALYZE_MODULE,
            _WORKER_MATCHER,
            _WORKER_TABLE,
            _WORKER_SAMPLE_RATE,
        )
    ):
        raise RuntimeError("Audfprint PEX query worker was not initialized")
    return evaluate_query_group(
        _WORKER_ANALYZER,
        _WORKER_MATCHER,
        _WORKER_TABLE,
        _WORKER_ANALYZE_MODULE,
        task,
        _WORKER_SAMPLE_RATE,
    )
def build_or_load_index(
    args: argparse.Namespace,
    reference_paths: dict[str, Path],
    analyze_module: Any,
    table_module: Any,
    output: Path,
) -> tuple[Any, Path, dict[str, object]]:
    index_path = (
        args.index.expanduser().resolve()
        if args.index is not None
        else output / "index" / "audfprint.pklz"
    )
    metadata_path = index_path.with_suffix(".json")
    if index_path.is_file():
        table = table_module.HashTable(str(index_path))
        expected_parameters = {
            "samplerate": args.sample_rate,
            "density": args.density,
            "fanout": args.fanout,
        }
        actual_parameters = {key: table.params.get(key) for key in expected_parameters}
        if actual_parameters != expected_parameters:
            raise EvaluationError(
                "Audfprint index parameters do not match the requested benchmark: "
                f"expected {expected_parameters}, found {actual_parameters}. "
                "Use a separate --index path or rebuild the index."
            )
        if table.hashbits != args.hashbits or table.depth != args.bucket_depth:
            raise EvaluationError(
                "Audfprint index hash table does not match the requested benchmark: "
                f"expected hashbits={args.hashbits}, depth={args.bucket_depth}; "
                f"found hashbits={table.hashbits}, depth={table.depth}."
            )
        indexed_ids = {Path(str(name)).stem for name in table.names}
        missing_ids = sorted(set(reference_paths) - indexed_ids)
        extra_ids = sorted(indexed_ids - set(reference_paths))
        if missing_ids or extra_ids:
            raise EvaluationError(
                "Audfprint index/reference mismatch: "
                f"missing={missing_ids[:5]}, extra={extra_ids[:5]}"
            )
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        )
        return table, index_path, metadata

    index_path.parent.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    analyzer = configure_analyzer(analyze_module, args, query=False)
    table = table_module.HashTable(
        hashbits=args.hashbits,
        depth=args.bucket_depth,
        maxtime=16_384,
    )
    for number, (reference_id, path) in enumerate(sorted(reference_paths.items()), start=1):
        hashes = analyzer.wavfile2hashes(str(path))
        table.store(reference_id, hashes)
        print(
            f"reference {number}/{len(reference_paths)}: {reference_id} ({len(hashes)} hashes)",
            flush=True,
        )
    table.save(
        str(index_path),
        params={
            "samplerate": args.sample_rate,
            "density": args.density,
            "fanout": args.fanout,
        },
    )
    metadata = {
        "reference_audio_seconds": analyzer.soundfiletotaldur,
        "reference_files": len(reference_paths),
    }
    write_summary(metadata, metadata_path)
    return table, index_path, metadata


def main() -> int:
    args = parse_arguments()
    validate_arguments(args)
    dataset = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_directory = args.source_dir.expanduser().resolve()
    analyze_module, match_module, table_module = load_audfprint(source_directory)
    annotations, reference_paths, query_paths = load_pex_dataset(dataset)
    segments = build_oracle_segments(annotations)
    table, index_path, build_metadata = build_or_load_index(
        args,
        reference_paths,
        analyze_module,
        table_module,
        output,
    )

    grouped = group_oracle_segments(segments)
    tasks = [(query_id, query_paths[query_id], grouped[query_id]) for query_id in sorted(grouped)]
    rows: list[dict[str, object]] = []
    completed = 0
    if args.query_workers == 1:
        analyzer = configure_analyzer(analyze_module, args, query=True)
        matcher = configure_matcher(match_module, args)
        iterator = (
            evaluate_query_group(
                analyzer,
                matcher,
                table,
                analyze_module,
                task,
                args.sample_rate,
            )
            for task in tasks
        )
        pool = None
    else:
        pool = multiprocessing.Pool(
            processes=args.query_workers,
            initializer=initialize_query_worker,
            initargs=(str(source_directory), str(index_path), vars(args)),
        )
        iterator = pool.imap(evaluate_query_worker, tasks, chunksize=1)
    try:
        for group_rows in iterator:
            rows.extend(group_rows)
            for row in group_rows:
                completed += 1
                print(
                    f"trial {completed}/{len(segments)}: {row['trial_id']} "
                    f"predicted={row['predicted_reference_id'] or '-'}",
                    flush=True,
                )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    trials_path = output / "query_results.csv"
    write_csv_rows(rows, trials_path, TRIAL_FIELDS)
    stored_postings = int(np.sum(np.minimum(table.depth, table.counts), dtype=np.int64))
    summary = {
        "schema": "polaris-paper-results-v1",
        "protocol": "pex_oracle_segment_exact_scale_v1",
        "system": args.profile,
        "dataset": str(dataset),
        "source": {
            "repository": "https://github.com/dpwe/audfprint",
            "directory": str(source_directory),
            "commit": source_commit(source_directory),
            "expected_commit": EXPECTED_SOURCE_COMMIT,
        },
        "configuration": {
            "profile": args.profile,
            "reference_density": args.density,
            "reference_fanout": args.fanout,
            "query_density": (
                args.query_density if args.query_density is not None else args.density
            ),
            "query_fanout": (args.query_fanout if args.query_fanout is not None else args.fanout),
            "query_max_peaks_per_frame": args.query_max_peaks_per_frame,
            "hashbits": args.hashbits,
            "bucket_depth": args.bucket_depth,
            "sample_rate": args.sample_rate,
            "query_shifts": args.query_shifts,
            "match_window": args.match_window,
            "min_count": args.min_count,
            "search_depth": args.search_depth,
            "top_k": args.top_k,
        },
        **summarize_pex(rows),
        "index": {
            "path": str(index_path),
            "stored_postings": stored_postings,
            **build_metadata,
        },
        "files": {"query_results_csv": str(trials_path)},
        "limitations": [
            "Ground-truth boundaries provide oracle segmentation.",
            "Tempo- and pitch-modified annotations are excluded.",
        ],
    }
    summary_path = output / "summary.json"
    write_summary(summary, summary_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Results: {trials_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
