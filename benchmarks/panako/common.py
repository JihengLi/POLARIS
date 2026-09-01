"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

#!/usr/bin/env python3

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import math
import os
import platform
import re
import shutil
import subprocess
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from pydub import AudioSegment

from polaris_eval.io import EvaluationError

PANAKO_VERSION = "2.1"
DEFAULT_JAR = Path(__file__).parent / ".cache" / "Panako" / "build" / "libs" / "panako-2.1-all.jar"


def _finite_or_negative_infinity(value: float) -> float:
    """Map non-finite native metrics to a deterministic lowest rank."""

    return value if math.isfinite(value) else -math.inf


def _match_rank_key(reference_id: str, match: PanakoWindowMatch) -> tuple[float, float, float, str]:
    """Return a total ordering for Panako candidates, including weak-score ties."""

    return (
        _finite_or_negative_infinity(match.score),
        _finite_or_negative_infinity(match.seconds_with_match),
        _finite_or_negative_infinity(match.reference_stop - match.reference_start),
        reference_id,
    )


@dataclass(frozen=True)
class PanakoWindowMatch:
    query_path: Path
    query_start: float
    reference_path: Path
    reference_start: float
    reference_stop: float
    score: float
    seconds_with_match: float


def window_starts(duration_ms: int, window_ms: int, hop_ms: int) -> list[int]:
    if duration_ms <= window_ms:
        return [0]
    starts = list(range(0, duration_ms - window_ms + 1, hop_ms))
    final_start = duration_ms - window_ms
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def parse_query_output(output: str) -> list[PanakoWindowMatch]:
    matches = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(";")]
        if len(fields) != 13 or fields[0] == "Index":
            continue
        if fields[5] in {"", "null"} or fields[6] in {"", "null"}:
            continue
        try:
            matches.append(
                PanakoWindowMatch(
                    query_path=Path(fields[2]).resolve(),
                    query_start=float(fields[3]),
                    reference_path=Path(fields[5]).resolve(),
                    reference_start=float(fields[7]),
                    reference_stop=float(fields[8]),
                    score=float(fields[9]),
                    seconds_with_match=float(fields[12]),
                )
            )
        except ValueError as exc:
            raise EvaluationError(f"Could not parse Panako query row: {line}") from exc
    return matches


def parse_stats_output(output: str) -> dict[str, float | int]:
    patterns = {
        "reference_songs": r">\s+(\d+) audio files",
        "audio_seconds": r">\s+([0-9.]+) seconds of audio",
        "stored_fingerprints": r"> Number of items in databases:\s+(\d+)",
    }
    parsed: dict[str, float | int] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match is None:
            raise EvaluationError(f"Panako stats output is missing {key!r}")
        parsed[key] = int(match.group(1)) if key != "audio_seconds" else float(match.group(1))
    return parsed


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_java(requested: Path | None) -> Path:
    if requested is not None:
        java = requested.expanduser().resolve()
        if not java.is_file():
            raise EvaluationError(f"Java executable does not exist: {java}")
        return java
    configured = os.environ.get("PANAKO_JAVA")
    if configured:
        return resolve_java(Path(configured))
    java_home = Path("/usr/libexec/java_home")
    if platform.system() == "Darwin" and java_home.is_file():
        result = subprocess.run(
            [str(java_home), "-v", "17"], check=False, capture_output=True, text=True
        )
        candidate = Path(result.stdout.strip()) / "bin" / "java"
        if result.returncode == 0 and candidate.is_file():
            return candidate.resolve()
    executable = shutil.which("java")
    if executable is None:
        raise EvaluationError("Panako 2.1 requires JDK 17 or later")
    return Path(executable).resolve()


def resolve_native_library_path(requested: Path | None) -> Path | None:
    if requested is not None:
        path = requested.expanduser().resolve()
        if not path.is_dir():
            raise EvaluationError(f"Native library directory does not exist: {path}")
        return path
    if platform.system() != "Darwin":
        return None
    for candidate in (Path("/opt/homebrew/opt/lmdb/lib"), Path("/usr/local/opt/lmdb/lib")):
        if any(candidate.glob("liblmdb*.dylib")):
            return candidate.resolve()
    raise EvaluationError("Panako 2.1 needs LMDB on macOS; install it with `brew install lmdb`")


def _resolve_lmdb_library(native_library_path: Path | None) -> Path | str | None:
    if native_library_path is not None:
        for pattern in ("liblmdb*.dylib", "liblmdb*.so", "liblmdb*.dll"):
            candidates = sorted(native_library_path.glob(pattern))
            if candidates:
                return candidates[0]
    return ctypes.util.find_library("lmdb")


def clear_stale_lmdb_readers(database: Path, native_library_path: Path | None) -> int:
    database = database.resolve()
    if not (database / "data.mdb").is_file() or not (database / "lock.mdb").is_file():
        return 0
    library_path = _resolve_lmdb_library(native_library_path)
    if library_path is None:
        raise EvaluationError("Could not locate liblmdb for Panako index recovery")
    lmdb = ctypes.CDLL(str(library_path))
    environment = ctypes.c_void_p()
    lmdb.mdb_env_create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lmdb.mdb_env_create.restype = ctypes.c_int
    lmdb.mdb_env_open.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint, ctypes.c_uint]
    lmdb.mdb_env_open.restype = ctypes.c_int
    lmdb.mdb_reader_check.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    lmdb.mdb_reader_check.restype = ctypes.c_int
    lmdb.mdb_env_close.argtypes = [ctypes.c_void_p]
    lmdb.mdb_strerror.argtypes = [ctypes.c_int]
    lmdb.mdb_strerror.restype = ctypes.c_char_p

    def checked(code: int, operation: str) -> None:
        if code:
            detail = lmdb.mdb_strerror(code)
            message = detail.decode(errors="replace") if detail else f"error {code}"
            raise EvaluationError(f"LMDB {operation} failed for {database}: {message}")

    checked(lmdb.mdb_env_create(ctypes.byref(environment)), "environment creation")
    try:
        checked(lmdb.mdb_env_open(environment, os.fsencode(database), 0, 0o644), "open")
        dead_readers = ctypes.c_int()
        checked(lmdb.mdb_reader_check(environment, ctypes.byref(dead_readers)), "reader check")
        return dead_readers.value
    finally:
        lmdb.mdb_env_close(environment)


class PanakoCLI:
    def __init__(
        self,
        *,
        jar: Path,
        java: Path,
        database_directory: Path,
        native_library_path: Path | None,
        workers: int,
        number_of_results: int,
        strategy: str = "PANAKO",
    ) -> None:
        self.jar = jar.expanduser().resolve()
        self.java = java
        self.database_directory = database_directory.resolve()
        self.cache_directory = self.database_directory.parent / "cache-disabled"
        self.native_library_path = native_library_path
        self.workers = workers
        self.number_of_results = number_of_results
        self.strategy = strategy.upper()
        if self.strategy != "PANAKO":
            raise EvaluationError(f"Unsupported strategy: {strategy}")
        self.java_max_heap = os.environ.get("PANAKO_JAVA_MAX_HEAP")
        if self.java_max_heap and not re.fullmatch(r"[1-9][0-9]*[kKmMgG]", self.java_max_heap):
            raise EvaluationError("PANAKO_JAVA_MAX_HEAP must look like 2g or 1536m")

    def command(self, application: str, *arguments: str) -> list[str]:
        prefix = self.strategy
        command = [
            str(self.java),
            "--add-opens=java.base/java.nio=ALL-UNNAMED",
            "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
            "-server",
        ]
        if self.java_max_heap:
            command.append(f"-Xmx{self.java_max_heap}")
        if self.native_library_path is not None:
            command.append(f"-Djava.library.path={self.native_library_path}")
        command.extend(
            [
                "-jar",
                str(self.jar),
                application,
                f"STRATEGY={prefix}",
                f"{prefix}_LMDB_FOLDER={self.database_directory}",
                f"{prefix}_CACHE_FOLDER={self.cache_directory}",
                f"{prefix}_CACHE_TO_FILE=FALSE",
                f"{prefix}_USE_CACHED_PRINTS=FALSE",
                f"AVAILABLE_PROCESSORS={self.workers}",
                f"NUMBER_OF_QUERY_RESULTS={self.number_of_results}",
                *arguments,
            ]
        )
        return command

    def _prepare_database(self) -> None:
        cleared = clear_stale_lmdb_readers(self.database_directory, self.native_library_path)
        if cleared:
            print(f"Panako LMDB recovery: cleared {cleared} stale reader slots", flush=True)

    def run(self, application: str, *arguments: str) -> subprocess.CompletedProcess[str]:
        self._prepare_database()
        result = subprocess.run(
            self.command(application, *arguments), check=False, capture_output=True, text=True
        )
        if result.returncode:
            detail = result.stderr.strip() or result.stdout.strip()
            raise EvaluationError(f"Panako {application} failed ({result.returncode}):\n{detail}")
        return result

    def run_streaming(self, application: str, *arguments: str) -> None:
        self._prepare_database()
        command = self.command(application, *arguments)
        tail: deque[str] = deque(maxlen=200)
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            tail.append(line)
        if process.wait():
            raise EvaluationError(f"Panako {application} failed:\n{''.join(tail).strip()}")


def _write_path_list(paths: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(f"{path.resolve()}\n" for path in paths), encoding="utf-8")


def build_reference_database(
    cli: PanakoCLI,
    reference_paths: dict[str, Path],
    reference_list_path: Path,
    *,
    batch_size: int | None = None,
) -> float:
    paths = [path for _, path in sorted(reference_paths.items())]
    _write_path_list(paths, reference_list_path)
    started = perf_counter()
    if batch_size is None:
        cli.run_streaming("store", "CHECK_DUPLICATE_FILE_NAMES=TRUE", str(reference_list_path))
    else:
        if batch_size <= 0:
            raise EvaluationError("reference store batch size must be positive")
        for start in range(0, len(paths), batch_size):
            cli.run_streaming(
                "store",
                "CHECK_DUPLICATE_FILE_NAMES=TRUE",
                *(str(path) for path in paths[start : start + batch_size]),
            )
    return perf_counter() - started


def query_one_file(
    cli: PanakoCLI,
    query_id: str,
    query_path: Path,
    *,
    window_seconds: float,
    hop_seconds: float,
    query_configuration: tuple[str, ...] = (),
) -> tuple[list[tuple[str, PanakoWindowMatch, float]], float]:
    return query_many_files(
        cli,
        [(query_id, query_path)],
        window_seconds=window_seconds,
        hop_seconds=hop_seconds,
        query_configuration=query_configuration,
    )[query_id]


def query_many_files(
    cli: PanakoCLI,
    queries: list[tuple[str, Path]],
    *,
    window_seconds: float,
    hop_seconds: float,
    query_configuration: tuple[str, ...] = (),
) -> dict[str, tuple[list[tuple[str, PanakoWindowMatch, float]], float]]:
    if not queries:
        return {}
    if len(queries) != len({query_id for query_id, _ in queries}):
        raise EvaluationError("Panako batch query IDs must be unique")
    started = perf_counter()
    window_ms = round(window_seconds * 1000)
    hop_ms = round(hop_seconds * 1000)
    best: dict[str, dict[str, tuple[PanakoWindowMatch, float]]] = {
        query_id: {} for query_id, _ in queries
    }
    with tempfile.TemporaryDirectory(prefix="panako-query-") as directory:
        root = Path(directory)
        metadata: dict[Path, tuple[str, float]] = {}
        windows = []
        for number, (query_id, query_path) in enumerate(queries):
            audio = (
                AudioSegment.from_file(query_path)
                .set_channels(1)
                .set_frame_rate(16000)
                .set_sample_width(2)
            )
            starts = window_starts(len(audio), window_ms, hop_ms)
            query_directory = root / f"q{number:04d}"
            query_directory.mkdir()
            for start_ms in starts:
                path = (query_directory / f"w{start_ms:09d}.wav").resolve()
                audio[start_ms : min(start_ms + window_ms, len(audio))].export(path, format="wav")
                metadata[path] = (query_id, start_ms / 1000)
                windows.append(path)
        output = cli.run("query", *query_configuration, *(str(path) for path in windows)).stdout
        for match in parse_query_output(output):
            if match.query_path not in metadata:
                raise EvaluationError(f"Panako returned unknown query window: {match.query_path}")
            query_id, start = metadata[match.query_path]
            reference_id = match.reference_path.stem
            previous = best[query_id].get(reference_id)
            key = _match_rank_key(reference_id, match)
            previous_key = _match_rank_key(reference_id, previous[0]) if previous else None
            if previous_key is None or key > previous_key:
                best[query_id][reference_id] = (match, start)
    per_query = (perf_counter() - started) / len(queries)
    results = {}
    for query_id, _ in queries:
        ranked = sorted(
            (
                (reference_id, match, start)
                for reference_id, (match, start) in best[query_id].items()
            ),
            key=lambda item: _match_rank_key(item[0], item[1]),
            reverse=True,
        )
        results[query_id] = (ranked, per_query)
    return results


def git_commit_for_jar(jar: Path) -> str | None:
    for parent in jar.parents:
        if (parent / ".git").exists():
            result = subprocess.run(
                ["git", "-C", str(parent), "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return result.stdout.strip()
    return None
