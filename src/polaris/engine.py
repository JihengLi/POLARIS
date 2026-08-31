"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import multiprocessing
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from time import perf_counter

import numpy as np
from pydub import AudioSegment

from polaris.config import DEFAULT_CONFIG, PolarisConfig
from polaris.fingerprint import Fingerprint, FingerprintLike, fingerprint_reference, prepare_query
from polaris.index import FileIndex, PackedIndex, has_packed_index, merge_file_index


def read_audio(filename: str | Path, *, sample_rate: int) -> np.ndarray:
    audio = AudioSegment.from_file(str(filename))
    audio = audio.set_sample_width(2).set_frame_rate(sample_rate).set_channels(1)
    return np.frombuffer(audio.raw_data, dtype=np.int16)


def _confident(results: Sequence[dict[str, object]], config: PolarisConfig) -> bool:
    if not results:
        return False
    first = float(results[0]["ranking_score"])
    second = float(results[1]["ranking_score"]) if len(results) > 1 else 0.0
    margin = (first - second) / first if first > 0 else 0.0
    return (
        int(results[0]["hashes_matched_in_input"])
        >= config.adaptive.min_matching_hashes
        and margin >= config.adaptive.min_score_margin
    )


class VoteAccumulator:
    def __init__(self, recognizer: Polaris) -> None:
        self.recognizer = recognizer
        self.seen: set[FingerprintLike] = set()
        self.votes: dict[int, dict[int, list[float]]] = {}

    @property
    def query_records(self) -> int:
        return len(self.seen)

    def add(self, fingerprints: Iterable[FingerprintLike]) -> None:
        new = set(fingerprints) - self.seen
        self.seen.update(new)
        query_offsets: dict[str | int, list[int]] = defaultdict(list)
        for fingerprint_hash, offset in new:
            query_offsets[fingerprint_hash].append(offset)
        for batch in self.recognizer.index.iter_posting_batches(
            query_offsets,
            self.recognizer.config.matcher,
        ):
            for track_id, reference_offset in batch.postings:
                track_votes = self.votes.setdefault(track_id, {})
                for query_offset in batch.query_offsets:
                    offset = reference_offset - query_offset
                    cell = track_votes.setdefault(offset, [0.0, 0.0])
                    cell[0] += 1.0
                    cell[1] += batch.idf_weight

    def rank(self, topn: int) -> list[dict[str, object]]:
        return self.recognizer._rank(self.votes, topn)


class Polaris:
    def __init__(
        self,
        index_directory: str | Path,
        *,
        config: PolarisConfig = DEFAULT_CONFIG,
        index_backend: str = "auto",
    ) -> None:
        if index_backend not in {"auto", "file", "packed"}:
            raise ValueError("index_backend must be auto, file, or packed")
        self.config = config
        path = Path(index_directory).expanduser().resolve()
        packed = index_backend == "packed" or (
            index_backend == "auto" and has_packed_index(path)
        )
        self.index: FileIndex | PackedIndex = (
            PackedIndex(path) if packed else FileIndex(path, writable=False)
        )
        self.packed = packed

    def close(self) -> None:
        self.index.close()

    def __enter__(self) -> Polaris:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def recognize_hashes(
        self,
        fingerprints: Iterable[FingerprintLike],
        *,
        topn: int,
    ) -> list[dict[str, object]]:
        accumulator = VoteAccumulator(self)
        accumulator.add(fingerprints)
        return accumulator.rank(topn)

    def _rank(
        self,
        votes: dict[int, dict[int, list[float]]],
        topn: int,
    ) -> list[dict[str, object]]:
        matcher = self.config.matcher
        ranked: list[tuple[int, int, int, float]] = []
        for track_id, track_votes in votes.items():
            candidates: list[tuple[int, int, float]] = []
            for offset, (raw_count, weighted_score) in track_votes.items():
                count = int(raw_count)
                score = weighted_score
                for distance in range(1, matcher.offset_smoothing_radius + 1):
                    weight = matcher.offset_neighbor_weight**distance
                    for neighbor_offset in (offset - distance, offset + distance):
                        neighbor = track_votes.get(neighbor_offset)
                        if neighbor is not None:
                            count += int(neighbor[0])
                            score += weight * neighbor[1]
                candidates.append((offset, count, score))
            offset, count, score = max(
                candidates,
                key=lambda value: (value[2], value[1], -value[0]),
            )
            ranked.append((track_id, offset, count, score))

        ranked.sort(key=lambda value: (value[3], value[2]), reverse=True)
        results = []
        for track_id, offset, count, score in ranked[:topn]:
            track = self.index.get_song_by_id(track_id)
            results.append(
                {
                    "song_name": track["song_name"],
                    "hashes_matched_in_input": count,
                    "ranking_score": round(score, 5),
                    "offset_seconds": round(
                        offset
                        * self.config.audio.hop_size
                        / self.config.audio.sample_rate,
                        5,
                    ),
                }
            )
        return results

    def recognize_samples(
        self,
        samples: np.ndarray,
        *,
        mode: str,
        sample_rate: int | None = None,
        topn: int | None = None,
    ) -> dict[str, object]:
        if mode not in {"o", "a", "f"}:
            raise ValueError("mode must be 'o', 'a', or 'f'")
        requested_topn = self.config.matcher.topn if topn is None else topn
        internal_topn = max(2, requested_topn)
        effective_rate = sample_rate or self.config.audio.sample_rate
        started = perf_counter()
        prepared = prepare_query(
            samples,
            effective_rate,
            config=self.config,
            packed=self.packed,
        )
        accumulator = VoteAccumulator(self)
        first_stage = "two_hop" if mode == "f" else "original"
        accumulator.add(prepared.hashes(first_stage))
        results = accumulator.rank(internal_topn)
        query_stage = "two_hop" if mode == "f" else "original"
        if mode == "a" and not _confident(results, self.config):
            accumulator.add(prepared.hashes("two_hop"))
            results = accumulator.rank(internal_topn)
            query_stage = "two_hop"
        return {
            "query_stage": query_stage,
            "query_records": accumulator.query_records,
            "total_time": perf_counter() - started,
            "results": results[:requested_topn],
        }

    def recognize_file(
        self,
        filename: str | Path,
        *,
        mode: str,
        topn: int | None = None,
    ) -> dict[str, object]:
        started = perf_counter()
        samples = read_audio(filename, sample_rate=self.config.audio.sample_rate)
        result = self.recognize_samples(samples, mode=mode, topn=topn)
        result["total_time"] = perf_counter() - started
        return result


def _fingerprint_worker(arguments: tuple[str, PolarisConfig]) -> tuple[str, set[Fingerprint]]:
    filename, config = arguments
    samples = read_audio(filename, sample_rate=config.audio.sample_rate)
    hashes = fingerprint_reference(samples, config.audio.sample_rate, config=config)
    return Path(filename).stem, hashes  # type: ignore[return-value]


def build_reference_index(
    files: Sequence[str | Path],
    index_directory: str | Path,
    *,
    config: PolarisConfig = DEFAULT_CONFIG,
    workers: int = 1,
    merge_chunk_size: int = 1_000_000,
) -> int:
    if workers <= 0:
        raise ValueError("workers must be positive")
    index = FileIndex(index_directory, writable=True)
    missing = [
        Path(path).expanduser().resolve()
        for path in files
        if not index.is_song_fingerprinted(Path(path).stem)
    ]
    inputs = [(str(path), config) for path in missing]
    inserted = 0
    iterator: Iterable[tuple[str, set[Fingerprint]]]
    if workers == 1:
        iterator = map(_fingerprint_worker, inputs)
        for track_name, hashes in iterator:
            track_id = index.insert_song(track_name, len(hashes))
            index.insert_hashes(track_id, hashes)
            inserted += 1
    else:
        with multiprocessing.Pool(workers) as pool:
            for track_name, hashes in pool.imap_unordered(_fingerprint_worker, inputs):
                track_id = index.insert_song(track_name, len(hashes))
                index.insert_hashes(track_id, hashes)
                inserted += 1
    index.close()
    merge_file_index(index_directory, chunk_size=merge_chunk_size)
    return inserted


def main() -> int:
    parser = argparse.ArgumentParser(description="Recognize one query with POLARIS.")
    parser.add_argument("index", type=Path)
    parser.add_argument("query", type=Path)
    parser.add_argument("--mode", choices=("o", "a", "f"), default="f")
    parser.add_argument("--index-backend", choices=("auto", "file", "packed"), default="auto")
    parser.add_argument("--topn", type=int, default=2)
    args = parser.parse_args()
    with Polaris(args.index, index_backend=args.index_backend) as recognizer:
        result = recognizer.recognize_file(args.query, mode=args.mode, topn=args.topn)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
