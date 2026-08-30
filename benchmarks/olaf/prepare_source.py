"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from common import (
    DEFAULT_BINARY,
    DEFAULT_SOURCE,
    OLAF_COMMIT,
    OLAF_REPOSITORY,
    OLAF_TAG,
)

QUERY_PROBE_FUNCTIONS = ("olaf_query", "olaf_query_json", "olaf_query_collect")


def apply_readonly_query_probe_patch(source: Path) -> None:
    """Make the redundant query preflight read-only without touching matching."""

    bridge = source / "cli" / "olaf_cli_bridge.c"
    text = bridge.read_text(encoding="utf-8")
    for function in QUERY_PROBE_FUNCTIONS:
        signature = f"{function}(Olaf_Config* config"
        function_start = text.find(signature)
        if function_start < 0:
            raise SystemExit(f"Could not locate official {function} in {bridge}")
        probe_start = text.find("olaf_db_new(config->dbFolder,", function_start)
        runner_start = text.find("olaf_runner_new", function_start)
        if probe_start < 0 or runner_start < 0 or probe_start > runner_start:
            raise SystemExit(f"Could not locate the {function} database preflight")
        writable = "olaf_db_new(config->dbFolder,false)"
        readonly = "olaf_db_new(config->dbFolder,true)"
        probe = text[probe_start:runner_start]
        if writable in probe:
            probe = probe.replace(writable, readonly, 1)
            text = text[:probe_start] + probe + text[runner_start:]
        elif readonly not in probe:
            raise SystemExit(f"Unexpected {function} database preflight in {bridge}")
    bridge.write_text(text, encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--zig", type=Path, help="zig or python-zig executable")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def command_for_zig(explicit: Path | None) -> list[str]:
    if explicit is not None:
        return [str(explicit.expanduser().resolve())]
    for executable in ("zig", "python-zig"):
        resolved = shutil.which(executable)
        if resolved:
            return [resolved]
    uvx = shutil.which("uvx")
    if uvx:
        return [uvx, "--from", "ziglang", "python-zig"]
    raise SystemExit("Zig 0.16 is unavailable. Install the `ziglang` package or pass --zig.")


def main() -> int:
    args = parse_arguments()
    source = args.source.expanduser().resolve()
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--branch",
                OLAF_TAG,
                "--depth",
                "1",
                OLAF_REPOSITORY,
                str(source),
            ],
            check=True,
        )
    if not (source / ".git").is_dir():
        raise SystemExit(f"OLAF source is not a Git checkout: {source}")
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != OLAF_COMMIT:
        raise SystemExit(f"Expected OLAF {OLAF_TAG} commit {OLAF_COMMIT}, found {commit}")
    if args.check_only:
        print(f"OLAF {OLAF_TAG} source is pinned at {commit}")
        return 0

    apply_readonly_query_probe_patch(source)
    zig = command_for_zig(args.zig)
    version = subprocess.run(
        [*zig, "version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if tuple(int(part) for part in version.split(".")[:2]) < (0, 16):
        raise SystemExit(f"OLAF v2.0.10 requires Zig >=0.16; found {version}")
    subprocess.run([*zig, "build", "-Doptimize=ReleaseFast"], cwd=source, check=True)
    binary = source / DEFAULT_BINARY.relative_to(DEFAULT_SOURCE)
    if not binary.is_file():
        raise SystemExit(f"OLAF build completed without the expected binary: {binary}")
    print(f"Built OLAF {OLAF_TAG} ({commit}) with Zig {version}: {binary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
