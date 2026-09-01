"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

OLAF_VERSION = "2.0.10"
OLAF_TAG = "v2.0.10"
OLAF_COMMIT = "532f1991ba170b39d2156b43935428814e833614"
OLAF_REPOSITORY = "https://github.com/JorenSix/Olaf"
BUILD_COMPATIBILITY_PATCH = (
    "query database-existence probes use read-only LMDB transactions; the official "
    "query runner was already read-only and extraction/matching are unchanged"
)
ADAPTER_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_SOURCE = ADAPTER_DIRECTORY / ".cache" / "Olaf"
DEFAULT_BINARY = DEFAULT_SOURCE / "zig-out" / "bin" / "olaf"
INDEX_METADATA_FILENAME = "olaf_index.json"
PROFILE_NAME = "paper"
PROFILE_DESCRIPTION = "frozen storage-matched OLAF v2.0.10 profile"
PROFILE_OVERRIDES = {
    "max_event_point_usages": 24,
    "max_fingerprints": 1100,
    "search_range": 9,
    "max_db_collisions": 8000,
    # Both paper protocols are closed-set identification tasks.
    "min_match_count": 1,
}


class OlafAdapterError(RuntimeError):
    """Raised when the external OLAF process or its output is invalid."""


@dataclass(frozen=True)
class OlafMatch:
    reference_id: str
    query_start: float
    reference_start: float

    @property
    def offset_seconds(self) -> float:
        return self.reference_start - self.query_start


@dataclass(frozen=True)
class OlafQueryResult:
    matches: tuple[OlafMatch, ...]
    query_records: int
    wall_seconds: float


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def numeric_reference_map(references: Mapping[str, Path]) -> dict[int, str]:
    """Allocate stable uint32 CLI IDs while preserving arbitrary dataset IDs."""

    if len(references) > 0xFFFFFFFF:
        raise OlafAdapterError("reference collection exceeds OLAF's uint32 ID space")
    return {
        internal_id: reference_id
        for internal_id, reference_id in enumerate(sorted(references), start=1)
    }


def parse_json_documents(text: str) -> list[dict[str, object]]:
    """Parse one or more whitespace-separated JSON documents."""

    decoder = json.JSONDecoder()
    documents: list[dict[str, object]] = []
    cursor = 0
    while cursor < len(text):
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text):
            break
        try:
            value, cursor = decoder.raw_decode(text, cursor)
        except json.JSONDecodeError as exc:
            preview = text[cursor : cursor + 240].replace("\n", "\\n")
            raise OlafAdapterError(f"invalid OLAF JSON near {preview!r}") from exc
        if not isinstance(value, dict):
            raise OlafAdapterError("OLAF emitted a non-object JSON document")
        documents.append(value)
    return documents


def parse_store_records(stderr: str) -> list[dict[str, object]]:
    records = []
    for line in stderr.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise OlafAdapterError(f"invalid OLAF store JSON line: {line[:240]!r}") from exc
        if value.get("action") == "store":
            records.append(value)
    return records


def parse_stats_output(stdout: str) -> dict[str, float | int]:
    patterns = {
        "reference_songs": r"Number of songs \(#\):\s*(\d+)",
        "audio_seconds": r"Total duration \(s\):\s*([0-9.]+)",
        "stored_fingerprints": r"Number of items in databases:\s*(\d+)",
    }
    values: dict[str, float | int] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, stdout)
        if match is None:
            raise OlafAdapterError(f"OLAF stats output is missing {key}:\n{stdout}")
        values[key] = (
            int(match.group(1))
            if key in {"reference_songs", "stored_fingerprints"}
            else float(match.group(1))
        )
    return values


def rank_query_document(
    document: Mapping[str, object],
    reference_ids: Mapping[int, str],
) -> tuple[OlafMatch, ...]:
    """Keep OLAF's best native offset bin for each reference, in native order."""

    raw_matches = document.get("matches")
    if not isinstance(raw_matches, list):
        raise OlafAdapterError("OLAF query JSON has no matches array")
    ranked: list[OlafMatch] = []
    seen: set[int] = set()
    for raw in raw_matches:
        if not isinstance(raw, dict):
            raise OlafAdapterError("OLAF query JSON contains a non-object match")
        internal_id = int(raw["match_identifier"])
        if internal_id in seen:
            continue
        reference_id = reference_ids.get(internal_id)
        if reference_id is None:
            raise OlafAdapterError(f"OLAF returned unknown reference ID {internal_id}")
        seen.add(internal_id)
        ranked.append(
            OlafMatch(
                reference_id=reference_id,
                query_start=float(raw["query_start"]),
                reference_start=float(raw["reference_start"]),
            )
        )
    return tuple(ranked)


class OlafCLI:
    """An isolated OLAF binary, HOME, runtime configuration and LMDB index."""

    def __init__(
        self,
        binary: Path,
        index: Path,
        *,
        runtime_root: Path | None = None,
    ):
        self.binary = binary.expanduser().resolve()
        self.index = index.expanduser().resolve()
        self.profile_name = PROFILE_NAME
        self.profile_overrides = dict(PROFILE_OVERRIDES)
        self.database = self.index / "db"
        runtime_root = (
            runtime_root.expanduser().resolve() if runtime_root is not None else self.index
        )
        self.cache = runtime_root / "cache"
        self.runtime_home = runtime_root / "runtime_home"
        self.config_path = self.runtime_home / ".olaf" / "olaf_config.json"
        if not self.binary.is_file():
            raise OlafAdapterError(
                f"OLAF v{OLAF_VERSION} binary does not exist: {self.binary}. "
                "Run benchmarks/olaf/prepare_source.py first."
            )
        self._write_config()

    def _write_config(self) -> None:
        self.database.mkdir(parents=True, exist_ok=True)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        config = {
            "db_folder": str(self.database) + os.sep,
            "cache_folder": str(self.cache) + os.sep,
            **self.profile_overrides,
        }
        serialized = json.dumps(config, indent=2, sort_keys=True) + "\n"
        if not self.config_path.is_file() or self.config_path.read_text() != serialized:
            self.config_path.write_text(serialized, encoding="utf-8")

    @property
    def environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment["HOME"] = str(self.runtime_home)
        environment["LC_ALL"] = "C"
        return environment

    def run(
        self, *arguments: str, timeout: float | None = None
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [str(self.binary), *arguments],
            check=False,
            capture_output=True,
            text=True,
            env=self.environment,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise OlafAdapterError(
                f"OLAF command failed ({result.returncode}): {' '.join(arguments)}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        return result

    def stats(self) -> dict[str, float | int]:
        return parse_stats_output(self.run("stats").stdout)

    def query(self, path: Path, reference_ids: Mapping[int, str]) -> OlafQueryResult:
        started = perf_counter()
        result = self.run("query", "--format", "json", str(path.resolve()))
        wall_seconds = perf_counter() - started
        documents = parse_json_documents(result.stdout)
        if len(documents) != 1:
            raise OlafAdapterError(
                f"expected one OLAF JSON query document for {path}, found {len(documents)}"
            )
        document = documents[0]
        return OlafQueryResult(
            matches=rank_query_document(document, reference_ids),
            query_records=int(document["fingerprints_matched"]),
            wall_seconds=wall_seconds,
        )

    def provenance(self) -> dict[str, object]:
        config_output = self.run("config").stdout
        source = DEFAULT_SOURCE.resolve()
        commit = None
        if (source / ".git").is_dir():
            commit = subprocess.run(
                ["git", "-C", str(source), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        if commit != OLAF_COMMIT:
            raise OlafAdapterError(
                f"expected OLAF {OLAF_TAG} source commit {OLAF_COMMIT}, found {commit}"
            )
        return {
            "repository": OLAF_REPOSITORY,
            "tag": OLAF_TAG,
            "commit": commit,
            "expected_commit": OLAF_COMMIT,
            "source_path": str(source),
            "binary_path": str(self.binary),
            "binary_sha256": file_sha256(self.binary),
            "runtime_config_path": str(self.config_path),
            "runtime_config_sha256": file_sha256(self.config_path),
            "profile": self.profile_name,
            "profile_description": PROFILE_DESCRIPTION,
            "profile_overrides": self.profile_overrides,
            "compiled_configuration": config_output,
            "build_compatibility_patch": BUILD_COMPATIBILITY_PATCH,
            "platform": platform.platform(),
        }


def build_reference_index(
    cli: OlafCLI,
    references: Mapping[str, Path],
    *,
    workers: int,
    batch_size: int,
) -> float:
    if workers <= 0 or batch_size <= 0:
        raise OlafAdapterError("OLAF workers and store batch size must be positive")
    reference_ids = numeric_reference_map(references)
    internal_ids = {
        reference_id: internal_id for internal_id, reference_id in reference_ids.items()
    }
    metadata_path = cli.index / INDEX_METADATA_FILENAME
    expected_map = {str(key): value for key, value in sorted(reference_ids.items())}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        current_stats = parse_stats_output(cli.run("stats").stdout)
        metadata_profile = metadata.get("profile")
        metadata_overrides = metadata.get("profile_overrides", {})
        if (
            metadata.get("commit") == OLAF_COMMIT
            and metadata_profile == cli.profile_name
            and metadata_overrides == cli.profile_overrides
            and metadata.get("reference_ids") == expected_map
            and int(current_stats["reference_songs"]) == len(references)
        ):
            print(f"OLAF index is complete; reusing {len(references)} references", flush=True)
            return 0.0
        raise OlafAdapterError(
            "existing OLAF v2.0.10 index metadata does not match this reference "
            "collection; choose a new --index directory"
        )
    if (cli.database / "data.mdb").is_file():
        raise OlafAdapterError(
            "OLAF index contains stored data but has no completion metadata; "
            "remove this incomplete generated index or choose a new --index directory"
        )
    started = perf_counter()
    records: dict[int, dict[str, object]] = {}
    ordered = sorted(references.items())
    for batch_start in range(0, len(ordered), batch_size):
        batch = ordered[batch_start : batch_start + batch_size]
        arguments = ["store", "--threads", str(workers), "--format", "json", "--with-ids"]
        for reference_id, path in batch:
            arguments.extend((str(path.resolve()), str(internal_ids[reference_id])))
        result = cli.run(*arguments)
        for record in parse_store_records(result.stderr):
            records[int(record["internal_id"])] = record
        completed = min(batch_start + len(batch), len(ordered))
        print(f"OLAF indexed {completed}/{len(ordered)} references", flush=True)

    stats = parse_stats_output(cli.run("stats").stdout)
    if int(stats["reference_songs"]) != len(references):
        raise OlafAdapterError(
            f"OLAF index/reference mismatch after build: expected {len(references)}, "
            f"found {stats['reference_songs']}"
        )
    missing_records = sorted(set(reference_ids) - set(records))
    if missing_records:
        raise OlafAdapterError(
            "OLAF did not emit store summaries for reference IDs: "
            + ", ".join(map(str, missing_records[:10]))
        )
    metadata = {
        "system": "olaf",
        "version": OLAF_VERSION,
        "tag": OLAF_TAG,
        "commit": OLAF_COMMIT,
        "profile": cli.profile_name,
        "profile_overrides": cli.profile_overrides,
        "references": len(references),
        "audio_seconds": sum(float(record["audio_seconds"]) for record in records.values()),
        "reference_ids": {str(key): value for key, value in sorted(reference_ids.items())},
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return perf_counter() - started


def validate_index(cli: OlafCLI, references: Mapping[str, Path]) -> dict[str, float | int]:
    reference_ids = numeric_reference_map(references)
    stats = cli.stats()
    if int(stats["reference_songs"]) != len(references):
        raise OlafAdapterError(
            f"OLAF index/reference mismatch: expected {len(references)}, "
            f"found {stats['reference_songs']}; rerun with --build-index"
        )
    metadata_path = cli.index / INDEX_METADATA_FILENAME
    if not metadata_path.is_file():
        raise OlafAdapterError(
            f"OLAF index metadata is missing: {metadata_path}; rerun with --build-index"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_map = {str(key): value for key, value in sorted(reference_ids.items())}
    if metadata.get("reference_ids") != expected_map:
        raise OlafAdapterError("OLAF index metadata does not match this reference collection")
    if metadata.get("commit") != OLAF_COMMIT:
        raise OlafAdapterError("OLAF index was not built with the pinned v2.0.10 commit")
    if metadata.get("profile") != cli.profile_name:
        raise OlafAdapterError("OLAF index was built with a different retrieval profile")
    if metadata.get("profile_overrides", {}) != cli.profile_overrides:
        raise OlafAdapterError("OLAF index configuration does not match this run")
    return stats


def ffmpeg_version() -> str:
    result = subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True, text=True)
    return result.stdout.splitlines()[0]
