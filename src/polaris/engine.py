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
from polaris.fingerprint import (
    QUERY_STAGES,
    Fingerprint,
    FingerprintLike,
    fingerprint,
    prepare_query,
)
from polaris.index import (
    FileIndex,
    PackedIndex,
    has_packed_index,
    merge_file_index,
)


def read_audio(
    file_name: str | Path,
    *,
    sample_rate: int,
) -> np.ndarray:
    """Decode one file to mono 16-bit PCM at the configured sample rate."""

    audio = AudioSegment.from_file(str(file_name))
    audio = audio.set_sample_width(2).set_frame_rate(sample_rate).set_channels(1)
    return np.frombuffer(audio.raw_data, dtype=np.int16)


def _track_identity_confident(
    results: Sequence[dict[str, object]],
    *,
    minimum_margin_fraction: float,
    minimum_matching_count: int,
) -> bool:
    if not results:
        return False
    top1_score = float(results[0]["ranking_score"])
    top2_score = float(results[1]["ranking_score"]) if len(results) > 1 else 0.0
    margin_fraction = (top1_score - top2_score) / top1_score if top1_score > 0 else 0.0
    return (
        margin_fraction >= minimum_margin_fraction
        and int(results[0]["hashes_matched_in_input"]) >= minimum_matching_count
    )


class VoteAccumulator:
    """Incrementally add disjoint query evidence to one offset histogram."""

    def __init__(self, recognizer: Polaris) -> None:
        self.recognizer = recognizer
        self.seen_hashes: set[FingerprintLike] = set()
        self.votes: dict[int, dict[int, list[float]]] = {}

    @property
    def queried_hashes(self) -> int:
        return len(self.seen_hashes)

    def add_hashes(self, hashes: Iterable[FingerprintLike]) -> int:
        new_hashes = set(hashes) - self.seen_hashes
        if not new_hashes:
            return 0
        self.seen_hashes.update(new_hashes)
        query_offsets: dict[str | int, list[int]] = defaultdict(list)
        for fingerprint_hash, offset in new_hashes:
            query_offsets[fingerprint_hash].append(offset)

        for batch in self.recognizer.db.iter_posting_batches(
            query_offsets,
            self.recognizer.config.matcher,
        ):
            for song_id, reference_offset in batch.postings:
                song_votes = self.votes.setdefault(song_id, {})
                for sampled_offset in batch.query_offsets:
                    offset = reference_offset - sampled_offset
                    cell = song_votes.get(offset)
                    if cell is None:
                        song_votes[offset] = [1.0, batch.idf_weight]
                    else:
                        cell[0] += 1.0
                        cell[1] += batch.idf_weight
        return len(new_hashes)

    def rank(
        self,
        *,
        topn: int | None = None,
    ) -> list[dict[str, object]]:
        return self.recognizer._rank_votes(
            self.votes,
            topn=topn,
        )


class Polaris:
    """The frozen POLARIS recognizer backed by a compatible file index."""

    def __init__(
        self,
        index_directory: str | Path,
        *,
        config: PolarisConfig = DEFAULT_CONFIG,
        index_backend: str = "auto",
    ) -> None:
        self.config = config
        index_path = Path(index_directory).expanduser().resolve()
        if index_backend not in {"auto", "file", "packed"}:
            raise ValueError("index_backend must be 'auto', 'file', or 'packed'")
        use_packed = index_backend == "packed" or (
            index_backend == "auto" and has_packed_index(index_path)
        )
        self.db: FileIndex | PackedIndex = (
            PackedIndex(index_path) if use_packed else FileIndex(index_path, writable=False)
        )
        self.uses_packed_index = use_packed

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> Polaris:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def recognize_hashes(
        self,
        hashes: Iterable[FingerprintLike],
        *,
        topn: int | None = None,
    ) -> list[dict[str, object]]:
        """Stream postings directly into offset histograms, then rank songs."""

        query_hashes = set(hashes)
        if not query_hashes:
            return []
        accumulator = VoteAccumulator(self)
        accumulator.add_hashes(query_hashes)
        return accumulator.rank(topn=topn)

    def _rank_votes(
        self,
        votes: dict[int, dict[int, list[float]]],
        *,
        topn: int | None,
    ) -> list[dict[str, object]]:
        matcher = self.config.matcher
        if matcher.offset_smoothing_radius < 0:
            raise ValueError("offset_smoothing_radius must be non-negative")
        if not 0 <= matcher.offset_neighbor_weight <= 1:
            raise ValueError("offset_neighbor_weight must be within [0, 1]")
        if matcher.song_hash_normalization_exponent < 0:
            raise ValueError("song_hash_normalization_exponent must be non-negative")

        ranked_song_matches: list[tuple[int, int, int, float, float]] = []
        for song_id, song_votes in votes.items():
            candidates: list[tuple[int, int, int, float]] = []
            for offset, cell in song_votes.items():
                matching_count = int(cell[0])
                weighted_score = cell[1]
                for distance in range(1, matcher.offset_smoothing_radius + 1):
                    distance_weight = matcher.offset_neighbor_weight**distance
                    for neighbor_offset in (offset - distance, offset + distance):
                        neighbor = song_votes.get(neighbor_offset)
                        if neighbor is None:
                            continue
                        matching_count += int(neighbor[0])
                        weighted_score += distance_weight * neighbor[1]
                candidates.append((song_id, offset, matching_count, weighted_score))
            best = max(
                candidates,
                key=lambda value: (value[3], value[2], -value[1]),
            )
            _, offset, matching_count, weighted_score = best
            song_hashes = int(self.db.get_song_by_id(song_id)["total_hashes"])
            ranking_score = weighted_score / (song_hashes**matcher.song_hash_normalization_exponent)
            ranked_song_matches.append(
                (
                    song_id,
                    offset,
                    matching_count,
                    weighted_score,
                    ranking_score,
                )
            )

        song_matches = sorted(
            ranked_song_matches,
            key=lambda match: (match[4], match[3], match[2]),
            reverse=True,
        )
        results = []
        result_limit = matcher.topn if topn is None else topn
        for song_id, offset, matching_count, _weighted_score, ranking_score in song_matches[
            :result_limit
        ]:
            song = self.db.get_song_by_id(song_id)
            song_hashes = int(song["total_hashes"])
            offset_seconds = round(
                offset * self.config.audio.hop_length / self.config.audio.sample_rate,
                5,
            )
            results.append(
                {
                    "song_name": song["song_name"],
                    "hashes_matched_in_input": matching_count,
                    "ranking_score": round(ranking_score, 5),
                    "offset_seconds": offset_seconds,
                }
            )
        return results

    def recognize_samples(
        self,
        samples: np.ndarray,
        *,
        sample_rate: int | None = None,
        topn: int | None = None,
    ) -> dict[str, object]:
        effective_sample_rate = sample_rate or self.config.audio.sample_rate
        started = perf_counter()
        hashes = fingerprint(
            samples,
            effective_sample_rate,
            config=self.config,
            for_query=True,
            packed=self.uses_packed_index,
        )
        results = self.recognize_hashes(hashes, topn=topn)
        return {
            "total_time": perf_counter() - started,
            "query_fingerprints": len(set(hashes)),
            "results": results,
        }

    def recognize_adaptive_samples(
        self,
        samples: np.ndarray,
        *,
        sample_rate: int | None = None,
        topn: int | None = None,
    ) -> dict[str, object]:
        """Increase query evidence only while track identity is uncertain.

        This deployment mode is tuned for track Top-1 identification.  The
        fixed full query remains available through :meth:`recognize_samples`
        when maximum offset accuracy is required.
        """

        effective_sample_rate = sample_rate or self.config.audio.sample_rate
        requested_topn = self.config.matcher.topn if topn is None else topn
        internal_topn = max(2, requested_topn)
        total_started = perf_counter()
        prepared = prepare_query(
            samples,
            effective_sample_rate,
            config=self.config,
            packed=self.uses_packed_index,
        )
        accumulator = VoteAccumulator(self)
        results: list[dict[str, object]] = []
        stopped_stage = "full"
        adaptive = self.config.adaptive_query
        for stage in QUERY_STAGES:
            hashes = prepared.hashes(stage)
            accumulator.add_hashes(hashes)
            results = accumulator.rank(topn=internal_topn)

            if stage == "canonical":
                confident = _track_identity_confident(
                    results,
                    minimum_margin_fraction=adaptive.canonical_margin_fraction,
                    minimum_matching_count=adaptive.canonical_min_matching_count,
                )
            elif stage == "two_hop":
                confident = _track_identity_confident(
                    results,
                    minimum_margin_fraction=adaptive.two_hop_margin_fraction,
                    minimum_matching_count=adaptive.two_hop_min_matching_count,
                )
            else:
                confident = True
            if confident:
                stopped_stage = stage
                break

        return {
            "mode": "adaptive_track",
            "stopped_stage": stopped_stage,
            "query_fingerprints": accumulator.queried_hashes,
            "total_time": perf_counter() - total_started,
            "results": results[:requested_topn],
        }

    def recognize_file(
        self,
        filename: str | Path,
        *,
        topn: int | None = None,
    ) -> dict[str, object]:
        samples = read_audio(
            filename,
            sample_rate=self.config.audio.sample_rate,
        )
        return self.recognize_samples(samples, topn=topn)

    def recognize_adaptive_file(
        self,
        filename: str | Path,
        *,
        topn: int | None = None,
    ) -> dict[str, object]:
        samples = read_audio(
            filename,
            sample_rate=self.config.audio.sample_rate,
        )
        return self.recognize_adaptive_samples(samples, topn=topn)


def _fingerprint_worker(arguments: tuple[str, PolarisConfig]) -> tuple[str, set[Fingerprint]]:
    filename, config = arguments
    samples = read_audio(
        filename,
        sample_rate=config.audio.sample_rate,
    )
    hashes = set(
        fingerprint(
            samples,
            config.audio.sample_rate,
            config=config,
            for_query=False,
        )
    )
    return Path(filename).stem, hashes


def build_reference_index(
    files: Sequence[str | Path],
    index_directory: str | Path,
    *,
    config: PolarisConfig = DEFAULT_CONFIG,
    workers: int = 1,
    merge_chunk_size: int = 1_000_000,
) -> int:
    """Add missing references and merge them into a queryable index."""

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
    if workers == 1:
        iterator = map(_fingerprint_worker, inputs)
        for song_name, hashes in iterator:
            song_id = index.insert_song(song_name, len(hashes))
            index.insert_hashes(song_id, hashes)
            inserted += 1
    else:
        with multiprocessing.Pool(processes=workers) as pool:
            for song_name, hashes in pool.imap_unordered(
                _fingerprint_worker,
                inputs,
            ):
                song_id = index.insert_song(song_name, len(hashes))
                index.insert_hashes(song_id, hashes)
                inserted += 1
    index.close()
    merge_file_index(index_directory, chunk_size=merge_chunk_size)
    return inserted


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recognize one audio file with the compact POLARIS engine."
    )
    parser.add_argument("index", type=Path, help="POLARIS index directory")
    parser.add_argument("query", type=Path, help="Query audio file")
    parser.add_argument(
        "--index-backend",
        choices=("auto", "file", "packed"),
        default="auto",
    )
    parser.add_argument("--topn", type=int, default=2)
    parser.add_argument(
        "--adaptive-track",
        action="store_true",
        help="stop query expansion when track identity is confident",
    )
    args = parser.parse_args()
    with Polaris(args.index, index_backend=args.index_backend) as recognizer:
        if args.adaptive_track:
            result = recognizer.recognize_adaptive_file(args.query, topn=args.topn)
        else:
            result = recognizer.recognize_file(args.query, topn=args.topn)
    print(json.dumps(result, indent=2))
    return 0
