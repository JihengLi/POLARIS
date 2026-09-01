"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydub import AudioSegment

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / "src", ROOT, Path(__file__).resolve().parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from common import (  # noqa: E402
    DEFAULT_BINARY,
    OLAF_COMMIT,
    OLAF_REPOSITORY,
    OLAF_TAG,
    OLAF_VERSION,
    PROFILE_DESCRIPTION,
    PROFILE_NAME,
    OlafAdapterError,
    OlafCLI,
    build_reference_index,
    ffmpeg_version,
    numeric_reference_map,
    validate_index,
)

from polaris_eval.datasets import load_pex  # noqa: E402
from polaris_eval.io import write_csv, write_json  # noqa: E402
from polaris_eval.metrics import PEX_RESULT_FIELDS, summarize_pex  # noqa: E402
from polaris_eval.protocol import build_oracle_segments, group_oracle_segments  # noqa: E402

FIELDS = PEX_RESULT_FIELDS


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OLAF v2.0.10 on PEX Hard Medium.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", "-o", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args()


def evaluate(prepared, cli: OlafCLI, reference_ids: dict[int, str]) -> dict[str, object]:
    query_id, segment, path = prepared
    result = cli.query(path, reference_ids)
    matches = result.matches
    predicted = matches[0] if matches else None
    return {
        "trial_id": segment.trial_id,
        "query_id": query_id,
        "reference_id": segment.annotation.reference_id,
        "query_begin": segment.begin,
        "status": "matched" if matches else "no_match",
        "predicted_reference_id": predicted.reference_id if predicted else "",
        "total_time": result.wall_seconds,
        "error": "",
    }


def main() -> int:
    args = arguments()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    dataset = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        annotations, references, query_paths = load_pex(dataset)
        segments = build_oracle_segments(annotations)
        grouped = group_oracle_segments(segments)
        reference_ids = numeric_reference_map(references)
        cli = OlafCLI(args.binary, args.index)
        if args.build_index:
            build_reference_index(cli, references, workers=args.workers, batch_size=100)
        index_stats = validate_index(cli, references)
    except (OlafAdapterError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="polaris-olaf-pex-") as temp_name:
        temporary = Path(temp_name)
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
                path = temporary / f"query-{len(prepared):06d}.wav"
                audio[begin_ms:end_ms].export(path, format="wav")
                prepared.append((query_id, segment, path))

        def evaluate_one(item):
            return evaluate(item, cli, reference_ids)

        pool = ThreadPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
        iterator = pool.map(evaluate_one, prepared) if pool else map(evaluate_one, prepared)
        try:
            for completed, row in enumerate(iterator, 1):
                rows.append(row)
                if completed % 10 == 0 or completed == len(prepared):
                    print(f"[{completed}/{len(prepared)}]", flush=True)
        finally:
            if pool:
                pool.shutdown(wait=True, cancel_futures=True)

    results_path = output / "query_results.csv"
    write_csv(rows, results_path, FIELDS)
    summary = {
        "schema": "polaris-paper-results-v1",
        "protocol": "pex_oracle_segment_exact_scale_v1",
        "system": "olaf",
        "system_version": OLAF_VERSION,
        "dataset": str(dataset),
        **summarize_pex(rows),
        "configuration": {
            "implementation": "official standalone C/Zig OLAF",
            "retrieval_profile": PROFILE_NAME,
            "profile_description": PROFILE_DESCRIPTION,
            "profile_overrides": cli.profile_overrides,
            "runtime_paths": ["db_folder", "cache_folder"],
            "max_query_results": 50,
            "workers": args.workers,
            "index_batch_size": 100,
        },
        "index": {**index_stats, "path": str(cli.database)},
        "provenance": {
            **cli.provenance(),
            "repository": OLAF_REPOSITORY,
            "tag": OLAF_TAG,
            "commit": OLAF_COMMIT,
            "ffmpeg": ffmpeg_version(),
        },
        "files": {"query_results_csv": str(results_path)},
        "limitations": [
            "Ground-truth boundaries provide oracle segmentation.",
            "Tempo- and pitch-modified annotations are excluded.",
            "The storage-matched paper profile is separately labelled.",
        ],
    }
    summary_path = output / "summary.json"
    write_json(summary, summary_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
