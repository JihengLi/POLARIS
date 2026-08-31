"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from common import (
    DEFAULT_JAR,
    PANAKO_VERSION,
    PanakoCLI,
    build_reference_database,
    file_sha256,
    git_commit_for_jar,
    parse_stats_output,
    query_many_files,
    resolve_java,
    resolve_native_library_path,
)
from pydub import AudioSegment

from polaris_eval.datasets import load_pex as load_pex_dataset
from polaris_eval.io import EvaluationError
from polaris_eval.io import write_csv as write_csv_rows
from polaris_eval.io import write_json as write_summary
from polaris_eval.metrics import PEX_RESULT_FIELDS, summarize_pex
from polaris_eval.protocol import (
    build_oracle_segments,
    group_oracle_segments,
)

TRIAL_FIELDS = PEX_RESULT_FIELDS

CLOSED_SET_CONFIGURATION = (
    "PANAKO_MIN_HITS_UNFILTERED=1",
    "PANAKO_MIN_HITS_FILTERED=1",
    "PANAKO_MIN_MATCH_DURATION=0",
    "PANAKO_MIN_SEC_WITH_MATCH=0",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Panako on PEX Hard Medium.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--jar", type=Path, default=DEFAULT_JAR)
    parser.add_argument("--build-index", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    system = "panako"
    strategy = "PANAKO"

    dataset = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    database_directory = args.index.expanduser().resolve()
    jar = args.jar.expanduser().resolve()

    try:
        if not jar.is_file():
            raise EvaluationError(f"Panako JAR does not exist: {jar}")
        java = resolve_java(None)
        native_library_path = resolve_native_library_path(None)
        annotations, reference_paths, query_paths = load_pex_dataset(dataset)
        _, segments = build_oracle_segments(annotations)
        grouped = group_oracle_segments(segments)
        cli = PanakoCLI(
            jar=jar,
            java=java,
            database_directory=database_directory,
            native_library_path=native_library_path,
            workers=1,
            number_of_results=len(reference_paths),
            strategy=strategy,
        )
        if args.build_index:
            database_directory.parent.mkdir(parents=True, exist_ok=True)
            build_reference_database(
                cli,
                reference_paths,
                database_directory.parent / "reference_files.txt",
                batch_size=None,
            )
        if not database_directory.is_dir():
            raise EvaluationError(
                f"Panako index does not exist; rerun with --build-index: {database_directory}"
            )
        index_stats = parse_stats_output(cli.run("stats").stdout)
        if index_stats["reference_songs"] != len(reference_paths):
            raise EvaluationError(
                "Panako index/reference mismatch: "
                f"expected {len(reference_paths)}, found {index_stats['reference_songs']}"
            )
    except (EvaluationError, OSError) as exc:
        raise SystemExit(str(exc)) from exc

    rows: list[dict[str, object]] = []
    completed = 0
    with tempfile.TemporaryDirectory(prefix="panako-pex-segments-") as temporary_directory:
        temporary = Path(temporary_directory)
        prepared = []
        for query_id, query_segments in grouped.items():
            audio = (
                AudioSegment.from_file(query_paths[query_id])
                .set_channels(1)
                .set_frame_rate(16000)
                .set_sample_width(2)
            )
            for segment in query_segments:
                begin_ms = max(0, round(segment.begin * 1000))
                end_ms = min(len(audio), round(segment.end * 1000))
                excerpt_path = temporary / f"trial-{len(prepared):06d}.wav"
                audio[begin_ms:end_ms].export(excerpt_path, format="wav")
                prepared.append((query_id, segment, excerpt_path, (end_ms - begin_ms) / 1000))

        for batch_start in range(0, len(prepared), 32):
            batch = prepared[batch_start : batch_start + 32]
            try:
                batch_results = query_many_files(
                    cli,
                    [(segment.trial_id, path) for _, segment, path, _ in batch],
                    window_seconds=25.0,
                    hop_seconds=20.0,
                    query_configuration=CLOSED_SET_CONFIGURATION,
                )
            except (EvaluationError, OSError) as exc:
                first_trial = batch[0][1].trial_id
                raise SystemExit(f"Panako query batch failed from {first_trial}: {exc}") from exc

            for query_id, segment, _, _ in batch:
                ranked, _, elapsed, _ = batch_results[segment.trial_id]
                predicted_id = ranked[0][0] if ranked else ""
                rows.append(
                    {
                        "trial_id": segment.trial_id,
                        "query_id": query_id,
                        "reference_id": segment.annotation.reference_id,
                        "query_begin": segment.begin,
                        "status": "matched" if ranked else "no_match",
                        "predicted_reference_id": predicted_id,
                        "query_records": None,
                        "total_time": elapsed,
                        "error": "",
                    }
                )
                completed += 1
                print(
                    f"trial {completed}/{len(segments)}: "
                    f"{segment.trial_id} predicted={predicted_id or '-'}",
                    flush=True,
                )

    trials_path = output / "query_results.csv"
    write_csv_rows(rows, trials_path, TRIAL_FIELDS)
    java_version = subprocess.run(
        [str(java), "-version"], check=False, capture_output=True, text=True
    ).stderr.splitlines()[0]
    summary = {
        "schema": "polaris-paper-results-v1",
        "protocol": "pex_oracle_segment_exact_scale_v1",
        "system": system,
        "system_version": PANAKO_VERSION,
        "system_commit": git_commit_for_jar(jar),
        "jar_path": str(jar),
        "jar_sha256": file_sha256(jar),
        "java": {"path": str(java), "version": java_version},
        "dataset": str(dataset),
        **summarize_pex(rows),
        "configuration": {
            "strategy": strategy,
            "algorithm_parameters": (
                f"official {strategy.title()} strategy from the Panako 2.1 release"
            ),
            "retrieval_profile": "closed-set-ranking",
            "query_configuration": list(CLOSED_SET_CONFIGURATION),
            "workers": 1,
            "query_batch_size": 32,
            "query_batching": "multiple trials per official single-worker JVM",
            "java_max_heap": cli.java_max_heap or "JVM ergonomic default",
            "window_seconds": 25.0,
            "hop_seconds": 20.0,
        },
        "index": {
            **index_stats,
            "path": str(database_directory),
        },
        "files": {"query_results_csv": str(trials_path)},
        "limitations": [
            "Ground-truth boundaries provide oracle segmentation.",
            "Tempo- and pitch-modified annotations are excluded.",
            (
                "Closed-set ranking relaxes Panako's product-level minimum-duration and "
                "acceptance gates; hash extraction, candidate lookup, scale fitting, line "
                "filtering and hit-count ranking are unchanged."
            ),
        ],
    }
    summary_path = output / "summary.json"
    write_summary(summary, summary_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
