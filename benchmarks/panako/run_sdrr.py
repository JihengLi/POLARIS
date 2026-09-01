"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
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
    DEFAULT_JAR,
    PANAKO_VERSION,
    PanakoCLI,
    build_reference_database,
    file_sha256,
    git_commit_for_jar,
    parse_stats_output,
    query_one_file,
    resolve_java,
    resolve_native_library_path,
)

from polaris_eval.datasets import (  # noqa: E402
    SdrrQuery,
    load_sdrr,
    unique_sdrr_references,
)
from polaris_eval.io import EvaluationError  # noqa: E402
from polaris_eval.io import write_csv as write_csv_rows  # noqa: E402
from polaris_eval.io import write_json as write_summary  # noqa: E402
from polaris_eval.metrics import SDRR_RESULT_FIELDS, summarize_sdrr  # noqa: E402

CLOSED_SET_CONFIGURATION = (
    # Reject-disabled Panako protocol used by Serrano and Scarpa
    # (Multimedia Tools and Applications, 2023).
    "PANAKO_MIN_HITS_UNFILTERED=1",
    "PANAKO_MIN_HITS_FILTERED=1",
    "PANAKO_MIN_MATCH_DURATION=0",
    "PANAKO_MIN_SEC_WITH_MATCH=0",
)

RESULT_FIELDS = tuple(field for field in SDRR_RESULT_FIELDS if field != "query_records")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Panako on SD-RR.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--jar", type=Path, default=DEFAULT_JAR)
    parser.add_argument("--build-index", action="store_true")
    return parser.parse_args()


def _common(item: SdrrQuery) -> dict[str, object]:
    return {
        "query_id": item.query_id,
        "reference_id": item.reference_id,
    }


def evaluate_query(
    cli: PanakoCLI,
    item: SdrrQuery,
    query_configuration: tuple[str, ...] = CLOSED_SET_CONFIGURATION,
) -> dict[str, object]:
    try:
        ranked, elapsed = query_one_file(
            cli,
            item.query_id,
            item.query_path,
            window_seconds=25.0,
            hop_seconds=20.0,
            query_configuration=query_configuration,
        )
        predicted = ranked[0] if ranked else None

        def offset(candidate: tuple | None) -> float | None:
            if candidate is None:
                return None
            _, match, window_start = candidate
            return float(match.reference_start - (window_start + match.query_start))

        predicted_offset = offset(predicted)
        predicted_reference = predicted[0] if predicted is not None else ""
        top1_error = (
            predicted_offset - item.reference_begin_seconds
            if predicted_reference == item.reference_id and predicted_offset is not None
            else None
        )
        row: dict[str, object] = {
            **_common(item),
            "status": "matched" if ranked else "no_match",
            "predicted_reference_id": predicted_reference,
            "predicted_offset_seconds": predicted_offset,
            "absolute_offset_error_seconds": abs(top1_error)
            if top1_error is not None
            else None,
            "total_time": elapsed,
            "error": "",
        }
        return row
    except Exception as exc:
        row = {
            **_common(item),
            "status": "error",
            "predicted_reference_id": "",
            "predicted_offset_seconds": None,
            "absolute_offset_error_seconds": None,
            "total_time": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
        return row


def main() -> int:
    args = parse_arguments()
    system = "panako"
    strategy = "PANAKO"
    query_configuration = CLOSED_SET_CONFIGURATION
    dataset = args.dataset.expanduser().resolve()
    manifest = dataset / "manifest.csv"
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    index = args.index.expanduser().resolve()
    database = index / "lmdb"
    jar = args.jar.expanduser().resolve()
    try:
        if not jar.is_file():
            raise EvaluationError(f"Panako JAR does not exist: {jar}")
        all_items = load_sdrr(dataset)
        references = unique_sdrr_references(all_items)
        items = list(all_items)
        java = resolve_java(None)
        native_library = resolve_native_library_path(None)
        cli = PanakoCLI(
            jar=jar,
            java=java,
            database_directory=database,
            native_library_path=native_library,
            workers=1,
            number_of_results=len(references),
            strategy=strategy,
        )
        if args.build_index:
            index.mkdir(parents=True, exist_ok=True)
            build_reference_database(
                cli,
                references,
                index / "reference_files.txt",
                batch_size=None,
            )
        if not database.is_dir():
            raise EvaluationError(
                f"Panako index does not exist; rerun with --build-index: {database}"
            )
        index_stats = parse_stats_output(cli.run("stats").stdout)
        if int(index_stats["reference_songs"]) != len(references):
            raise EvaluationError(
                f"Panako index/reference mismatch: expected {len(references)}, "
                f"found {index_stats['reference_songs']}"
            )
    except (EvaluationError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

    rows: list[dict[str, object]] = []
    for number, item in enumerate(items, start=1):
        row = evaluate_query(cli, item, query_configuration)
        rows.append(row)
        print(
            f"[{number}/{len(items)}] {item.query_id}: "
            f"predicted={row['predicted_reference_id'] or '-'} "
            f"offset_error={row['absolute_offset_error_seconds']}",
            flush=True,
        )

    results_path = output / "query_results.csv"
    write_csv_rows(rows, results_path, RESULT_FIELDS)
    java_version = subprocess.run(
        [str(java), "-version"], check=False, capture_output=True, text=True
    ).stderr.splitlines()[0]
    summary = {
        "schema": "polaris-paper-results-v1",
        "protocol": "sdrr_closed_set_retrieval_and_offset_v1",
        "system": system,
        "system_version": PANAKO_VERSION,
        "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "configuration": {
            "retrieval_profile": "closed-set-ranking",
            "strategy": strategy,
            "query_configuration": list(query_configuration),
            "query_duration_seconds": 10,
            "number_of_query_results": len(references),
        },
        "offset_definition": (
            "reference_match_start minus query_match_start; ground truth is "
            "reference_begin_seconds from the SD-RR manifest"
        ),
        **summarize_sdrr(rows),
        "index": {**index_stats, "path": str(database)},
        "provenance": {
            "commit": git_commit_for_jar(jar),
            "jar_sha256": file_sha256(jar),
            "java": {"path": str(java), "version": java_version},
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
