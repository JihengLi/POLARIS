"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
ADAPTER_DIRECTORY = Path(__file__).resolve().parent
for directory in (REPOSITORY_ROOT, ADAPTER_DIRECTORY):
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
    candidate_at_rank,
    ffmpeg_version,
    file_sha256,
    numeric_reference_map,
    validate_index,
)

from polaris_eval.io import write_csv as write_csv_rows  # noqa: E402
from polaris_eval.io import write_json as write_summary  # noqa: E402
from polaris_eval.metrics import SDRR_RESULT_FIELDS, summarize_sdrr  # noqa: E402
from polaris_eval.real import (  # noqa: E402
    RealQuery,
    load_real_manifest,
    unique_references,
)

RESULT_FIELDS = SDRR_RESULT_FIELDS


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OLAF v2.0.10 on SD-RR.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args()


def _common(item: RealQuery) -> dict[str, object]:
    return {
        "query_id": item.query_id,
        "reference_id": item.reference_id,
    }


def evaluate_query(
    cli: OlafCLI,
    item: RealQuery,
    reference_ids: dict[int, str],
) -> dict[str, object]:
    common = _common(item)
    try:
        result = cli.query(item.query_path, reference_ids)
        matches = result.matches
        predicted = candidate_at_rank(matches, 1)
        predicted_offset = predicted.offset_seconds if predicted else None
        predicted_reference = predicted.reference_id if predicted else ""
        top1_error = (
            predicted_offset - item.reference_begin_seconds
            if predicted_reference == item.reference_id and predicted_offset is not None
            else None
        )
        row: dict[str, object] = {
            **common,
            "status": "matched" if matches else "no_match",
            "predicted_reference_id": predicted_reference,
            "predicted_offset_seconds": predicted_offset,
            "absolute_offset_error_seconds": (
                abs(top1_error) if top1_error is not None else None
            ),
            "query_records": result.query_records,
            "total_time": result.wall_seconds,
            "error": "",
        }
        return row
    except Exception as exc:
        return {
            **common,
            "status": "error",
            "predicted_reference_id": "",
            "predicted_offset_seconds": None,
            "absolute_offset_error_seconds": None,
            "query_records": 0,
            "total_time": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    args = parse_arguments()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    manifest = args.manifest.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        all_items = load_real_manifest(manifest)
        references = unique_references(all_items)
        reference_ids = numeric_reference_map(references)
        items = list(all_items)
        cli = OlafCLI(args.binary, args.index)
        if args.build_index:
            build_reference_index(
                cli,
                references,
                workers=args.workers,
                batch_size=100,
            )
        index_stats = validate_index(cli, references)
    except (OlafAdapterError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    def evaluate(item: RealQuery) -> dict[str, object]:
        return evaluate_query(cli, item, reference_ids)

    rows: list[dict[str, object]] = []
    if args.workers == 1:
        iterator = map(evaluate, items)
        pool = None
    else:
        pool = ThreadPoolExecutor(max_workers=args.workers)
        iterator = pool.map(evaluate, items)
    try:
        for completed, (_item, row) in enumerate(zip(items, iterator, strict=True), start=1):
            rows.append(row)
            if completed % 10 == 0 or completed == len(items):
                print(f"[{completed}/{len(items)}]", flush=True)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)

    results_path = output / "query_results.csv"
    write_csv_rows(rows, results_path, RESULT_FIELDS)
    summary = {
        "schema": "polaris-paper-results-v1",
        "protocol": "real_phone_closed_set_retrieval_and_offset_localization",
        "system": "olaf",
        "system_version": OLAF_VERSION,
        "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "configuration": {
            "implementation": "standalone official C/Zig OLAF repository",
            "retrieval_profile": PROFILE_NAME,
            "algorithm_parameters": PROFILE_DESCRIPTION,
            "runtime_paths": ["db_folder", "cache_folder"],
            "profile_overrides": cli.profile_overrides,
            "query_duration_seconds": 10,
            "max_query_results": 50,
            "workers": args.workers,
            "index_batch_size": 100,
        },
        "offset_definition": (
            "native OLAF reference_start minus query_start; ground truth is "
            "reference_begin_seconds from the real-query manifest"
        ),
        **summarize_sdrr(rows),
        "index": {
            **index_stats,
            "path": str(cli.database),
        },
        "provenance": {
            **cli.provenance(),
            "repository": OLAF_REPOSITORY,
            "tag": OLAF_TAG,
            "commit": OLAF_COMMIT,
            "ffmpeg": ffmpeg_version(),
        },
        "files": {"query_results_csv": str(results_path)},
    }
    summary_path = output / "summary.json"
    write_summary(summary, summary_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"summary: {summary_path}")
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
