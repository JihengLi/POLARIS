"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

#!/usr/bin/env python3

# ruff: noqa: E402 -- repository imports follow an explicit sys.path bootstrap.

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
ADAPTER_DIRECTORY = Path(__file__).resolve().parent
for directory in (REPOSITORY_ROOT, ADAPTER_DIRECTORY):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

DEFAULT_SOURCE_DIR = ADAPTER_DIRECTORY / ".cache" / "neural-music-fp"
DEFAULT_MODEL_DIR = (
    DEFAULT_SOURCE_DIR / "logs" / "nmfp" / "fma-nmfp_deg" / "checkpoint" / "nmfp-triplet"
)
SOURCE_COMMIT = "e95e2b4009751274b060a6b74c26ae1323daae59"

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

RESULT_FIELDS = (
    "query_id",
    "reference_id",
    "query_seconds",
    "ground_truth_offset_seconds",
    "status",
    "predicted_reference_id",
    "predicted_offset_seconds",
    "top1_absolute_offset_error_seconds",
    "query_embeddings",
    "total_time",
    "error",
)

HOP_SECONDS = 0.5
BATCH_SIZE = 32
FRAME_TOP_K = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--reference-cache",
        type=Path,
        help="reusable directory containing one NMFP embedding array per reference",
    )
    return parser.parse_args()


def _common_result(item: RealQuery) -> dict[str, object]:
    return {
        "query_id": item.query_id,
        "reference_id": item.reference_id,
        "query_seconds": item.query_duration_seconds,
        "ground_truth_offset_seconds": item.reference_begin_seconds,
    }


def result_from_ranking(
    item: RealQuery,
    ranking: list[tuple[str, float, int]],
    *,
    track_start_by_id: dict[str, int],
    hop_seconds: float,
    query_embeddings: int,
    total_time: float,
) -> dict[str, object]:
    """Convert NMFP's global candidate starts into comparable real-query metrics."""

    predicted = ranking[0] if ranking else None

    def local_start(match: tuple[str, float, int] | None) -> int | None:
        if match is None:
            return None
        return int(match[2]) - track_start_by_id[match[0]]

    predicted_start = local_start(predicted)
    predicted_offset = predicted_start * hop_seconds if predicted_start is not None else None
    top1_error = (
        predicted_offset - item.reference_begin_seconds
        if predicted is not None
        and predicted[0] == item.reference_id
        and predicted_offset is not None
        else None
    )
    row: dict[str, object] = {
        **_common_result(item),
        "status": "matched" if ranking else "no_match",
        "predicted_reference_id": predicted[0] if predicted else "",
        "predicted_offset_seconds": predicted_offset,
        "top1_absolute_offset_error_seconds": (abs(top1_error) if top1_error is not None else None),
        "query_embeddings": query_embeddings,
        "total_time": total_time,
        "error": "",
    }
    return row


def error_result(item: RealQuery, exc: Exception) -> dict[str, object]:
    return {
        **_common_result(item),
        "status": "error",
        "predicted_reference_id": "",
        "predicted_offset_seconds": None,
        "top1_absolute_offset_error_seconds": None,
        "query_embeddings": 0,
        "total_time": None,
        "error": f"{type(exc).__name__}: {exc}",
    }


def evaluate_query(
    item: RealQuery,
    *,
    decode: Callable[[Path], np.ndarray],
    embed: Callable[[np.ndarray], np.ndarray],
    database: np.ndarray,
    track_starts: np.ndarray,
    track_ends: np.ndarray,
    track_ids: list[str],
    track_start_by_id: dict[str, int],
    hop_seconds: float,
    frame_top_k: int,
    ranker: Callable[..., tuple[list[tuple[str, float, int]], int]],
) -> dict[str, object]:
    trial_started = time.perf_counter()
    try:
        query_audio = decode(item.query_path)
        query_embedding = embed(query_audio)
        ranking, _candidate_count = ranker(
            query_embedding,
            database,
            track_starts,
            track_ends,
            track_ids,
            top_k=frame_top_k,
        )
        return result_from_ranking(
            item,
            ranking,
            track_start_by_id=track_start_by_id,
            hop_seconds=hop_seconds,
            query_embeddings=len(query_embedding),
            total_time=time.perf_counter() - trial_started,
        )
    except Exception as exc:
        return error_result(item, exc)


def main() -> None:
    args = parse_args()
    # The existing official adapter imports TensorFlow at module load time.
    # Keep that import inside the executable path so helpers and unit tests can
    # inspect this evaluator without requiring the isolated NMFP environment.
    from common import file_md5, infer_embeddings, load_model, source_commit
    from run_pex import atomic_save_array, load_cached_array, rank_query

    manifest = args.manifest.expanduser().resolve()
    output = args.output.expanduser().resolve()
    source_dir = args.source_dir.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    reference_cache = (
        args.reference_cache.expanduser().resolve()
        if args.reference_cache is not None
        else output / "embedding_cache" / "references"
    )
    try:
        all_items = load_real_manifest(manifest)
        references = unique_references(all_items)
        items = list(all_items)
    except (EvaluationError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

    model, frontend, config, es, segment_audio = load_model(model_dir, source_dir)
    sample_rate = int(config["MODEL"]["AUDIO"]["FS"])
    segment_seconds = float(config["MODEL"]["AUDIO"]["SEGMENT_DUR"])
    dimensions = int(config["MODEL"]["ARCHITECTURE"]["EMB_SZ"])

    def decode(path: Path) -> np.ndarray:
        return np.asarray(
            es.MonoLoader(filename=str(path), sampleRate=sample_rate, resampleQuality=0)(),
            dtype=np.float32,
        )

    def embed(audio: np.ndarray) -> np.ndarray:
        return infer_embeddings(
            audio,
            model,
            frontend,
            segment_audio,
            sample_rate=sample_rate,
            segment_seconds=segment_seconds,
            hop_seconds=HOP_SECONDS,
            batch_size=BATCH_SIZE,
        )

    reference_embeddings = []
    track_ids = []
    for number, (reference_id, path) in enumerate(sorted(references.items()), start=1):
        cached_path = reference_cache / f"{reference_id}.npy"
        embeddings = load_cached_array(cached_path, dimensions)
        status = "cached"
        if embeddings is None:
            audio = decode(path)
            embeddings = embed(audio)
            atomic_save_array(cached_path, embeddings)
            status = "extracted"
        reference_embeddings.append(embeddings)
        track_ids.append(reference_id)
        print(
            f"reference {number}/{len(references)}: {reference_id} "
            f"({len(embeddings)} embeddings, {status})",
            flush=True,
        )

    lengths = np.asarray([len(array) for array in reference_embeddings], dtype=np.int64)
    track_ends = np.cumsum(lengths)
    track_starts = np.concatenate((np.asarray([0], dtype=np.int64), track_ends[:-1]))
    database = np.concatenate(reference_embeddings, axis=0).astype(np.float32, copy=False)
    del reference_embeddings
    track_start_by_id = {
        reference_id: int(track_starts[index]) for index, reference_id in enumerate(track_ids)
    }

    rows = []
    for number, item in enumerate(items, start=1):
        row = evaluate_query(
            item,
            decode=decode,
            embed=embed,
            database=database,
            track_starts=track_starts,
            track_ends=track_ends,
            track_ids=track_ids,
            track_start_by_id=track_start_by_id,
            hop_seconds=HOP_SECONDS,
            frame_top_k=FRAME_TOP_K,
            ranker=rank_query,
        )
        rows.append(row)
        print(
            f"[{number}/{len(items)}] {item.query_id}: "
            f"predicted={row['predicted_reference_id'] or '-'} "
            f"offset_error={row['top1_absolute_offset_error_seconds']}",
            flush=True,
        )

    results_path = output / "query_results.csv"
    write_csv_rows(rows, results_path, RESULT_FIELDS)
    checkpoint_data = model_dir / "ckpt-100.data-00000-of-00001"
    summary = {
        "protocol": "real_phone_closed_set_retrieval_and_offset_localization",
        "system": "nmfp-triplet",
        "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "source": {
            "repository": "https://github.com/raraz15/neural-music-fp",
            "directory": str(source_dir),
            "commit": source_commit(source_dir),
            "expected_commit": SOURCE_COMMIT,
        },
        "checkpoint": {
            "directory": str(model_dir),
            "data_bytes": checkpoint_data.stat().st_size,
            "data_md5": file_md5(checkpoint_data),
        },
        "configuration": {
            "sample_rate": sample_rate,
            "segment_seconds": segment_seconds,
            "hop_seconds": HOP_SECONDS,
            "embedding_dimensions": dimensions,
            "frame_top_k": FRAME_TOP_K,
            "batch_size": BATCH_SIZE,
            "device": "cpu",
            "candidate_generation": "exact cosine frame search",
            "reranking": "NMFP continuity candidates plus aligned mean cosine",
        },
        "offset_definition": (
            "best NMFP reference sequence start embedding multiplied by hop_seconds; "
            "ground truth is reference_begin_seconds from the real-query manifest"
        ),
        **summarize_rows(rows),
        "index": {
            "reference_files": len(references),
            "reference_embeddings": len(database),
        },
        "model": {
            "parameters": int(model.count_params()),
            "fingerprints_per_second": 1.0 / HOP_SECONDS,
        },
        "files": {
            "query_results_csv": str(results_path),
            "reference_embedding_cache": str(reference_cache),
        },
        "limitations": [
            "Closed-set evaluation over the references named by the real manifest.",
            "Exact flat frame search replaces approximate FAISS pruning.",
            "Offset resolution is limited by the configured embedding hop.",
        ],
    }
    summary_path = output / "summary.json"
    write_summary(summary, summary_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"per-query results: {results_path}")
    print(f"summary: {summary_path}")
    if summary["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
