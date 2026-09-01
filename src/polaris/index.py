"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

#!/usr/bin/env python3

from __future__ import annotations

import heapq
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from math import log
from pathlib import Path
from typing import TextIO

import numpy as np

from polaris.config import MatcherConfig
from polaris.fingerprint import Fingerprint, pack_hash


@dataclass(frozen=True)
class PostingBatch:
    query_offsets: Sequence[int]
    postings: Iterable[tuple[int, int]]
    idf_weight: float


class FileIndex:
    """Read or extend an existing text index without building global matches."""

    def __init__(self, directory: str | Path, *, writable: bool = False) -> None:
        self.directory = Path(directory).expanduser().resolve()
        if writable:
            self.directory.mkdir(parents=True, exist_ok=True)
        elif not self.directory.is_dir():
            raise FileNotFoundError(f"index directory does not exist: {self.directory}")

        self.songs_path = self.directory / "songs.csv"
        self.pending_fingerprints_path = self.directory / "fingerprints.csv"
        self.inverted_path = self.directory / "fingerprints_inverted.csv"
        self.songs: dict[str, int] = {}
        self.songs_by_id: dict[int, str] = {}
        self.max_song_id = 0
        self._load_songs()
        self.inverted_positions = self._load_inverted_positions()
        self.songs_file: TextIO | None = None
        self.fingerprints_file: TextIO | None = None
        if writable:
            self.songs_file = self.songs_path.open("a", encoding="utf-8")
            self.fingerprints_file = self.pending_fingerprints_path.open("a", encoding="utf-8")

    def _load_songs(self) -> None:
        if not self.songs_path.is_file():
            return
        with self.songs_path.open(encoding="utf-8") as songs_file:
            for line in songs_file:
                song_id_text, remainder = line.rstrip("\n").split(",", maxsplit=1)
                song_name, _ = remainder.rsplit(",", maxsplit=1)
                song_id = int(song_id_text)
                self.songs[song_name] = song_id
                self.songs_by_id[song_id] = song_name
                self.max_song_id = max(self.max_song_id, song_id)

    def _load_inverted_positions(self) -> dict[str, int]:
        positions: dict[str, int] = {}
        if not self.inverted_path.is_file():
            return positions
        with self.inverted_path.open(encoding="utf-8") as inverted_file:
            while True:
                position = inverted_file.tell()
                line = inverted_file.readline()
                if not line:
                    break
                positions[line.split(",", maxsplit=1)[0]] = position
        return positions

    def close(self) -> None:
        for handle in (self.songs_file, self.fingerprints_file):
            if handle is not None and not handle.closed:
                handle.flush()
                handle.close()

    def __enter__(self) -> FileIndex:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def is_song_fingerprinted(self, song_name: str) -> bool:
        return song_name in self.songs

    def get_song_name(self, song_id: int) -> str:
        return self.songs_by_id[song_id]

    def insert_song(self, song_name: str, total_hashes: int) -> int:
        if self.songs_file is None:
            raise RuntimeError("index was opened read-only")
        self.max_song_id += 1
        song_id = self.max_song_id
        self.songs_file.write(f"{song_id},{song_name},{total_hashes}\n")
        self.songs_file.flush()
        self.songs[song_name] = song_id
        self.songs_by_id[song_id] = song_name
        return song_id

    def insert_hashes(
        self,
        song_id: int,
        hashes: Iterable[Fingerprint],
    ) -> None:
        if self.fingerprints_file is None:
            raise RuntimeError("index was opened read-only")
        for fingerprint_hash, offset in sorted(hashes):
            self.fingerprints_file.write(f"{fingerprint_hash},1,{song_id}:{offset}\n")
        self.fingerprints_file.flush()

    def iter_posting_batches(
        self,
        query_offsets: Mapping[str, Sequence[int]],
        matcher: MatcherConfig,
    ) -> Iterator[PostingBatch]:
        """Yield one parsed posting list with its final IDF weight."""

        if matcher.idf_smoothing <= 0:
            raise ValueError("idf_smoothing must be positive")
        if matcher.idf_power < 0:
            raise ValueError("idf_power must be non-negative")
        if not query_offsets or not self.inverted_path.is_file():
            return

        with self.inverted_path.open(encoding="utf-8") as inverted_file:
            for fingerprint_hash, offsets in query_offsets.items():
                position = self.inverted_positions.get(fingerprint_hash)
                if position is None:
                    continue
                inverted_file.seek(position)
                stored_hash, _, postings_text = (
                    inverted_file.readline().rstrip("\n").split(",", maxsplit=2)
                )
                if stored_hash != fingerprint_hash:
                    raise RuntimeError("inverted-index position table points to the wrong hash")
                postings: list[tuple[int, int]] = []
                document_ids: set[int] = set()
                for posting in postings_text.split("-"):
                    song_id_text, reference_offset_text = posting.split(":")
                    song_id = int(song_id_text)
                    postings.append((song_id, int(reference_offset_text)))
                    document_ids.add(song_id)

                document_frequency = len(document_ids)
                total_songs = max(len(self.songs), document_frequency)
                idf_weight = (
                    1.0
                    + log(
                        (total_songs + matcher.idf_smoothing)
                        / (document_frequency + matcher.idf_smoothing)
                    )
                ) ** matcher.idf_power
                yield PostingBatch(offsets, postings, idf_weight)


PACKED_INDEX_FILES = {
    "keys": "polaris_keys.npy",
    "starts": "polaris_posting_starts.npy",
    "song_ids": "polaris_song_ids.npy",
    "offsets": "polaris_offsets.npy",
    "document_frequencies": "polaris_document_frequencies.npy",
    "metadata": "polaris_packed_index.json",
}


def has_packed_index(directory: str | Path) -> bool:
    path = Path(directory).expanduser().resolve()
    return all((path / filename).is_file() for filename in PACKED_INDEX_FILES.values())


class PackedIndex:
    """Memory-mapped integer-key index for the frozen triplet descriptor."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).expanduser().resolve()
        if not has_packed_index(self.directory):
            raise FileNotFoundError(f"packed POLARIS index is incomplete: {self.directory}")
        self.songs_path = self.directory / "songs.csv"
        self.songs: dict[str, int] = {}
        self.songs_by_id: dict[int, str] = {}
        self._load_songs()
        self.keys = np.load(
            self.directory / PACKED_INDEX_FILES["keys"],
            mmap_mode="r",
        )
        self.starts = np.load(
            self.directory / PACKED_INDEX_FILES["starts"],
            mmap_mode="r",
        )
        self.song_ids = np.load(
            self.directory / PACKED_INDEX_FILES["song_ids"],
            mmap_mode="r",
        )
        self.offsets = np.load(
            self.directory / PACKED_INDEX_FILES["offsets"],
            mmap_mode="r",
        )
        self.document_frequencies = np.load(
            self.directory / PACKED_INDEX_FILES["document_frequencies"],
            mmap_mode="r",
        )

    def _load_songs(self) -> None:
        with self.songs_path.open(encoding="utf-8") as songs_file:
            for line in songs_file:
                song_id_text, remainder = line.rstrip("\n").split(",", maxsplit=1)
                song_name, _ = remainder.rsplit(",", maxsplit=1)
                song_id = int(song_id_text)
                self.songs[song_name] = song_id
                self.songs_by_id[song_id] = song_name

    def close(self) -> None:
        # NumPy memmaps close with their owning arrays; no writable handle exists.
        return None

    def get_song_name(self, song_id: int) -> str:
        return self.songs_by_id[song_id]

    def iter_posting_batches(
        self,
        query_offsets: Mapping[int, Sequence[int]],
        matcher: MatcherConfig,
    ) -> Iterator[PostingBatch]:
        if matcher.idf_smoothing <= 0:
            raise ValueError("idf_smoothing must be positive")
        if matcher.idf_power < 0:
            raise ValueError("idf_power must be non-negative")
        if not query_offsets:
            return

        query_keys = np.fromiter(query_offsets, dtype=np.uint64)
        positions = np.searchsorted(self.keys, query_keys)
        for key, position in zip(query_keys, positions, strict=True):
            row = int(position)
            if row >= len(self.keys) or self.keys[row] != key:
                continue
            start = int(self.starts[row])
            stop = int(self.starts[row + 1])
            document_frequency = int(self.document_frequencies[row])
            total_songs = max(len(self.songs), document_frequency)
            idf_weight = (
                1.0
                + log(
                    (total_songs + matcher.idf_smoothing)
                    / (document_frequency + matcher.idf_smoothing)
                )
            ) ** matcher.idf_power
            postings = (
                (int(song_id), int(offset))
                for song_id, offset in zip(
                    self.song_ids[start:stop],
                    self.offsets[start:stop],
                    strict=True,
                )
            )
            yield PostingBatch(query_offsets[int(key)], postings, idf_weight)


def build_packed_index(
    source_directory: str | Path,
    destination_directory: str | Path | None = None,
) -> None:
    """Convert an existing grouped CSV index to sorted mmap arrays.

    No audio is decoded and no fingerprints are regenerated.  Existing packed
    files are never overwritten, making conversion safe to run beside a frozen
    CSV index.
    """

    source = Path(source_directory).expanduser().resolve()
    destination = (
        source
        if destination_directory is None
        else Path(destination_directory).expanduser().resolve()
    )
    inverted_path = source / "fingerprints_inverted.csv"
    songs_path = source / "songs.csv"
    if not inverted_path.is_file() or not songs_path.is_file():
        raise FileNotFoundError("source CSV index is incomplete")
    existing = [
        destination / filename
        for filename in PACKED_INDEX_FILES.values()
        if (destination / filename).exists()
    ]
    if existing:
        raise FileExistsError(f"packed index already exists: {existing[0]}")

    row_count = 0
    posting_count = 0
    maximum_song_id = 0
    maximum_offset = 0
    with inverted_path.open(encoding="utf-8") as inverted_file:
        for line in inverted_file:
            if not line.strip():
                continue
            fingerprint_hash, count_text, postings_text = line.rstrip("\n").split(",", maxsplit=2)
            postings = postings_text.split("-")
            if len(postings) != int(count_text):
                raise ValueError(f"posting count mismatch for {fingerprint_hash!r}")
            for posting in postings:
                song_id_text, offset_text = posting.split(":")
                maximum_song_id = max(maximum_song_id, int(song_id_text))
                maximum_offset = max(maximum_offset, int(offset_text))
            row_count += 1
            posting_count += int(count_text)

    def storage_dtype(maximum: int) -> np.dtype:
        if maximum <= np.iinfo(np.uint16).max:
            return np.dtype(np.uint16)
        if maximum <= np.iinfo(np.uint32).max:
            return np.dtype(np.uint32)
        return np.dtype(np.uint64)

    song_id_dtype = storage_dtype(maximum_song_id)
    offset_dtype = storage_dtype(maximum_offset)

    row_keys = np.empty(row_count, dtype=np.uint64)
    row_positions = np.empty(row_count, dtype=np.uint64)
    with inverted_path.open(encoding="utf-8") as inverted_file:
        row = 0
        while True:
            position = inverted_file.tell()
            line = inverted_file.readline()
            if not line:
                break
            if not line.strip():
                continue
            fingerprint_hash, count_text, _ = line.split(",", maxsplit=2)
            row_keys[row] = pack_hash(fingerprint_hash)
            row_positions[row] = position
            if int(count_text) < 0:
                raise ValueError("posting counts must be non-negative")
            row += 1
    order = np.argsort(row_keys, kind="stable")
    sorted_keys = row_keys[order]
    if len(sorted_keys) > 1 and np.any(sorted_keys[1:] == sorted_keys[:-1]):
        raise ValueError("source index contains duplicate grouped hash rows")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="polaris-packed-",
        dir=destination.parent,
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        keys = np.lib.format.open_memmap(
            temporary / PACKED_INDEX_FILES["keys"],
            mode="w+",
            dtype=np.uint64,
            shape=(row_count,),
        )
        starts = np.lib.format.open_memmap(
            temporary / PACKED_INDEX_FILES["starts"],
            mode="w+",
            dtype=np.uint64,
            shape=(row_count + 1,),
        )
        song_ids = np.lib.format.open_memmap(
            temporary / PACKED_INDEX_FILES["song_ids"],
            mode="w+",
            dtype=song_id_dtype,
            shape=(posting_count,),
        )
        offsets = np.lib.format.open_memmap(
            temporary / PACKED_INDEX_FILES["offsets"],
            mode="w+",
            dtype=offset_dtype,
            shape=(posting_count,),
        )
        document_frequencies = np.lib.format.open_memmap(
            temporary / PACKED_INDEX_FILES["document_frequencies"],
            mode="w+",
            dtype=np.uint32,
            shape=(row_count,),
        )

        cursor = 0
        with inverted_path.open(encoding="utf-8") as inverted_file:
            for output_row, source_row_value in enumerate(order):
                source_row = int(source_row_value)
                inverted_file.seek(int(row_positions[source_row]))
                fingerprint_hash, count_text, postings_text = (
                    inverted_file.readline().rstrip("\n").split(",", maxsplit=2)
                )
                expected_count = int(count_text)
                postings = postings_text.split("-")
                if len(postings) != expected_count:
                    raise ValueError(f"posting count mismatch for {fingerprint_hash!r}")
                keys[output_row] = pack_hash(fingerprint_hash)
                starts[output_row] = cursor
                documents: set[int] = set()
                for posting in postings:
                    song_id_text, offset_text = posting.split(":")
                    song_id = int(song_id_text)
                    song_ids[cursor] = song_id
                    offsets[cursor] = int(offset_text)
                    documents.add(song_id)
                    cursor += 1
                document_frequencies[output_row] = len(documents)
        starts[row_count] = cursor
        if cursor != posting_count:
            raise ValueError("packed posting total does not match source index")
        for array in (keys, starts, song_ids, offsets, document_frequencies):
            array.flush()
        del keys, starts, song_ids, offsets, document_frequencies

        metadata = {
            "format": "polaris-packed-mmap-v1",
            "hash_rows": row_count,
            "postings": posting_count,
            "key_bits": 40,
            "song_id_dtype": song_id_dtype.name,
            "offset_dtype": offset_dtype.name,
        }
        (temporary / PACKED_INDEX_FILES["metadata"]).write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        destination.mkdir(parents=True, exist_ok=True)
        if destination != source:
            shutil.copy2(songs_path, destination / "songs.csv")
            baseline_path = source / "baseline.json"
            if baseline_path.is_file():
                shutil.copy2(baseline_path, destination / "baseline.json")
        for filename in PACKED_INDEX_FILES.values():
            os.replace(temporary / filename, destination / filename)


def _hash_key(line: str) -> str:
    return line.split(",", maxsplit=1)[0]


def _write_sorted_chunk(lines: list[str], output_path: Path) -> None:
    lines.sort(key=_hash_key)
    output_path.write_text("".join(lines), encoding="utf-8")


def _external_sort(
    input_path: Path,
    output_path: Path,
    chunks_directory: Path,
    chunk_size: int,
) -> None:
    chunks_directory.mkdir(parents=True, exist_ok=True)
    chunk_paths: list[Path] = []
    with input_path.open(encoding="utf-8") as input_file:
        chunk_number = 0
        while True:
            lines = []
            for _ in range(chunk_size):
                line = input_file.readline()
                if not line:
                    break
                lines.append(line)
            if not lines:
                break
            chunk_number += 1
            chunk_path = chunks_directory / f"chunk-{chunk_number:05d}.csv"
            _write_sorted_chunk(lines, chunk_path)
            chunk_paths.append(chunk_path)

    with ExitStack() as stack, output_path.open("w", encoding="utf-8") as output_file:
        chunk_files = [stack.enter_context(path.open(encoding="utf-8")) for path in chunk_paths]
        for line in heapq.merge(*chunk_files, key=_hash_key):
            output_file.write(line)


def _group_postings(sorted_path: Path, grouped_path: Path) -> None:
    with (
        sorted_path.open(encoding="utf-8") as input_file,
        grouped_path.open("w", encoding="utf-8") as output_file,
    ):
        current_hash: str | None = None
        count = 0
        postings: list[str] = []

        def flush() -> None:
            if current_hash is not None:
                output_file.write(f"{current_hash},{count},{'-'.join(postings)}\n")

        for line in input_file:
            fingerprint_hash, row_count, row_postings = line.rstrip("\n").split(",", maxsplit=2)
            if current_hash is not None and fingerprint_hash != current_hash:
                flush()
                count = 0
                postings = []
            current_hash = fingerprint_hash
            count += int(row_count)
            postings.extend(row_postings.split("-"))
        flush()


def merge_file_index(index_directory: str | Path, *, chunk_size: int = 1_000_000) -> None:
    """Atomically merge pending fingerprint rows into the grouped index."""

    directory = Path(index_directory).expanduser().resolve()
    raw_path = directory / "fingerprints.csv"
    inverted_path = directory / "fingerprints_inverted.csv"
    if not raw_path.is_file() or raw_path.stat().st_size == 0:
        return
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    with tempfile.TemporaryDirectory(
        prefix="polaris-index-merge-",
        dir=directory,
    ) as temporary_directory:
        work_directory = Path(temporary_directory)
        merged_path = work_directory / "fingerprints_merged.csv"
        sorted_path = work_directory / "fingerprints_sorted.csv"
        candidate_path = work_directory / "fingerprints_inverted.csv"
        with merged_path.open("wb") as merged_file:
            with raw_path.open("rb") as pending_file:
                shutil.copyfileobj(pending_file, merged_file)
            if inverted_path.is_file():
                with inverted_path.open("rb") as inverted_file:
                    shutil.copyfileobj(inverted_file, merged_file)
        _external_sort(
            merged_path,
            sorted_path,
            work_directory / "chunks",
            chunk_size,
        )
        _group_postings(sorted_path, candidate_path)
        os.replace(candidate_path, inverted_path)
    raw_path.write_text("", encoding="utf-8")
