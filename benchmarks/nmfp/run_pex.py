"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

#!/usr/bin/env python3

# ruff: noqa: E402 -- repository imports follow an explicit sys.path bootstrap.

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from common import (
    DEFAULT_MODEL_DIR,
    DEFAULT_SOURCE_DIR,
    SOURCE_COMMIT,
    Trial,
    file_md5,
    infer_embeddings,
    load_model,
    load_trials,
    source_commit,
)

HOP_SECONDS = 0.5
BATCH_SIZE = 32
TOP_K = 10

from polaris_eval.metrics import summarize_single_reference_trials


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Pex dataset directory")
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--reference-cache",
        type=Path,
        help="Reusable directory of per-reference NMFP embeddings",
    )
    return parser.parse_args()


def scale_exact_trials(trials: list[Trial]) -> list[Trial]:
    return [trial for trial in trials if trial.scale_exact]


def atomic_save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp.npy",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        np.save(temporary_path, array, allow_pickle=False)
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def load_cached_array(path: Path, dimensions: int) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return None
    if array.ndim != 2 or array.shape[1] != dimensions or len(array) == 0:
        return None
    return np.asarray(array, dtype=np.float32)


def exact_top_k(query: np.ndarray, database: np.ndarray, top_k: int) -> np.ndarray:
    """Return exact cosine-nearest database rows for every query row."""

    k = min(top_k, len(database))
    similarities = np.matmul(query, database.T)
    if k == len(database):
        indices = np.argsort(-similarities, axis=1)
    else:
        partitioned = np.argpartition(similarities, -k, axis=1)[:, -k:]
        scores = np.take_along_axis(similarities, partitioned, axis=1)
        indices = np.take_along_axis(
            partitioned,
            np.argsort(-scores, axis=1),
            axis=1,
        )
    return np.asarray(indices, dtype=np.int64)


def candidate_starts(
    nearest_indices: np.ndarray,
    track_starts: np.ndarray,
    track_ends: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce NMFP continuity-based candidate sequence generation."""

    sequence_length = len(nearest_indices)
    offsets = np.arange(sequence_length, dtype=np.int64)[:, None]
    starts = (nearest_indices - offsets).reshape(-1)
    starts = starts[(starts >= 0) & (starts + sequence_length <= track_ends[-1])]
    if len(starts) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    starts = np.unique(starts)
    start_tracks = np.searchsorted(track_starts, starts, side="right") - 1
    last_indices = starts + sequence_length - 1
    end_tracks = np.searchsorted(track_starts, last_indices, side="right") - 1
    valid = (
        (start_tracks >= 0)
        & (start_tracks == end_tracks)
        & (last_indices < track_ends[start_tracks])
    )
    return starts[valid], start_tracks[valid]


def rerank_candidates(
    query: np.ndarray,
    database: np.ndarray,
    starts: np.ndarray,
    track_indices: np.ndarray,
    track_ids: list[str],
    *,
    chunk_size: int = 512,
) -> list[tuple[str, float, int]]:
    """Return the best aligned sequence score for every candidate track."""

    best: dict[int, tuple[float, int]] = {}
    relative = np.arange(len(query), dtype=np.int64)[None, :]
    for begin in range(0, len(starts), chunk_size):
        end = min(begin + chunk_size, len(starts))
        chunk_starts = starts[begin:end]
        sequences = database[chunk_starts[:, None] + relative]
        scores = np.mean(np.sum(sequences * query[None, :, :], axis=2), axis=1)
        for start, track_index, score in zip(
            chunk_starts,
            track_indices[begin:end],
            scores,
            strict=True,
        ):
            previous = best.get(int(track_index))
            candidate = (float(score), int(start))
            if previous is None or candidate > previous:
                best[int(track_index)] = candidate

    ranking = [
        (track_ids[track_index], score, start) for track_index, (score, start) in best.items()
    ]
    ranking.sort(key=lambda item: (-item[1], item[0], item[2]))
    return ranking


def rank_query(
    query: np.ndarray,
    database: np.ndarray,
    track_starts: np.ndarray,
    track_ends: np.ndarray,
    track_ids: list[str],
    *,
    top_k: int,
) -> tuple[list[tuple[str, float, int]], int]:
    nearest = exact_top_k(query, database, top_k)
    starts, track_indices = candidate_starts(nearest, track_starts, track_ends)
    ranking = rerank_candidates(
        query,
        database,
        starts,
        track_indices,
        track_ids,
    )
    return ranking, len(starts)


def main() -> int:
    args = parse_arguments()
    dataset = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    source_dir = args.source_dir.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    reference_cache = (
        args.reference_cache.expanduser().resolve()
        if args.reference_cache is not None
        else output / "embedding_cache" / "references"
    )

    all_trials, reference_paths, query_paths = load_trials(dataset)
    source_trials = scale_exact_trials(all_trials)
    if not source_trials:
        raise SystemExit("Pex dataset contains no scale-exact annotations")

    model, frontend, config, es, segment_audio = load_model(model_dir, source_dir)
    sample_rate = int(config["MODEL"]["AUDIO"]["FS"])
    segment_seconds = float(config["MODEL"]["AUDIO"]["SEGMENT_DUR"])
    dimensions = int(config["MODEL"]["ARCHITECTURE"]["EMB_SZ"])

    def decode(path: Path) -> np.ndarray:
        return np.asarray(
            es.MonoLoader(
                filename=str(path),
                sampleRate=sample_rate,
                resampleQuality=0,
            )(),
            dtype=np.float32,
        )

    trials = source_trials

    decoded_queries: dict[str, np.ndarray] = {}

    reference_embeddings: list[np.ndarray] = []
    track_ids: list[str] = []
    for number, (reference_id, path) in enumerate(sorted(reference_paths.items()), start=1):
        cached_path = reference_cache / f"{reference_id}.npy"
        embeddings = load_cached_array(cached_path, dimensions)
        status = "cached"
        if embeddings is None:
            audio = decode(path)
            embeddings = infer_embeddings(
                audio,
                model,
                frontend,
                segment_audio,
                sample_rate=sample_rate,
                segment_seconds=segment_seconds,
                hop_seconds=HOP_SECONDS,
                batch_size=BATCH_SIZE,
            )
            atomic_save_array(cached_path, embeddings)
            status = "extracted"
        reference_embeddings.append(embeddings)
        track_ids.append(reference_id)
        print(
            f"reference {number}/{len(reference_paths)}: {reference_id} "
            f"({len(embeddings)} embeddings, {status})",
            flush=True,
        )

    lengths = np.asarray([len(item) for item in reference_embeddings], dtype=np.int64)
    track_ends = np.cumsum(lengths)
    track_starts = np.concatenate((np.asarray([0], dtype=np.int64), track_ends[:-1]))
    database = np.concatenate(reference_embeddings, axis=0).astype(np.float32, copy=False)
    del reference_embeddings

    rows: list[dict[str, Any]] = []
    for number, trial in enumerate(trials, start=1):
        trial_started = time.perf_counter()
        query_audio = decoded_queries.get(trial.query_id)
        if query_audio is None:
            query_audio = decode(query_paths[trial.query_id])
            decoded_queries[trial.query_id] = query_audio
        begin = max(0, int(round(trial.query_begin * sample_rate)))
        end = min(len(query_audio), int(round(trial.query_end * sample_rate)))
        excerpt = query_audio[begin:end]
        query_embedding = infer_embeddings(
            excerpt,
            model,
            frontend,
            segment_audio,
            sample_rate=sample_rate,
            segment_seconds=segment_seconds,
            hop_seconds=HOP_SECONDS,
            batch_size=BATCH_SIZE,
        )
        ranking, _candidate_count = rank_query(
            query_embedding,
            database,
            track_starts,
            track_ends,
            track_ids,
            top_k=TOP_K,
        )
        predicted_id = ranking[0][0] if ranking else ""
        elapsed = time.perf_counter() - trial_started
        rows.append(
            {
                "trial": number,
                "query_id": trial.query_id,
                "query_begin": trial.query_begin,
                "query_end": trial.query_end,
                "query_seconds": len(excerpt) / sample_rate,
                "query_embeddings": len(query_embedding),
                "expected_reference_id": trial.reference_id,
                "predicted_reference_id": predicted_id,
                "total_time": elapsed,
            }
        )
        print(
            f"trial {number}/{len(trials)}: {trial.query_id}"
            f"[{trial.query_begin}:{trial.query_end}] expected={trial.reference_id} "
            f"predicted={predicted_id or '-'}",
            flush=True,
        )

    trials_path = output / "query_results.csv"
    with trials_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    checkpoint_data = model_dir / "ckpt-100.data-00000-of-00001"
    summary = {
        "protocol": "pex_scale_exact_annotation_retrieval",
        "is_full_pex_file_level_benchmark": False,
        "system": "nmfp-triplet",
        "dataset": str(dataset),
        "definition": (
            "All eligible Pex annotations with tempo=100 and pitch=0/empty; "
            "each ground-truth excerpt is searched against all references."
        ),
        "source": {
            "repository": "https://github.com/raraz15/neural-music-fp",
            "commit": source_commit(source_dir),
            "expected_commit": SOURCE_COMMIT,
        },
        "checkpoint": {
            "directory": str(model_dir),
            "data_bytes": checkpoint_data.stat().st_size,
            "data_md5": file_md5(checkpoint_data),
        },
        "model": {
            "parameters": int(model.count_params()),
            "embedding_dimensions": dimensions,
            "segment_seconds": segment_seconds,
            "hop_seconds": HOP_SECONDS,
            "fingerprints_per_second": 1.0 / HOP_SECONDS,
            "device": "cpu",
        },
        "dataset_counts": {
            "all_annotation_rows": len(all_trials),
            "evaluated_scale_exact_rows": len(trials),
            "source_scale_exact_rows": len(source_trials),
            "excluded_scale_changed_rows": len(all_trials) - len(scale_exact_trials(all_trials)),
            "query_files_represented": len({trial.query_id for trial in trials}),
            "reference_files_indexed": len(reference_paths),
            "expected_reference_files": len({trial.reference_id for trial in trials}),
        },
        "retrieval_configuration": {
            "candidate_generation": (
                f"exact cosine Top-{TOP_K} per query embedding, equivalent "
                "to an uncompressed flat NMFP index"
            ),
            "reranking": "official continuity candidates + aligned mean cosine",
        },
        "paper_metrics": summarize_single_reference_trials(rows),
        "index": {
            "reference_embeddings": len(database),
        },
        "files": {
            "query_results_csv": str(trials_path),
            "reference_embedding_cache": str(reference_cache),
        },
        "limitations": [
            "Ground-truth boundaries split montage queries into single-reference excerpts.",
            "This measures identification accuracy, not Pex file-level multi-label detection.",
            "Tempo- or pitch-modified annotations are excluded before evaluation.",
            "Exact flat nearest-neighbour search replaces approximate FAISS pruning.",
        ],
    }
    summary_path = output / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Results: {trials_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
