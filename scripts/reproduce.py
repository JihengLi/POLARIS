"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
POLARIS_SYSTEMS = (
    "polaris_o",
    "polaris_a",
    "polaris_f",
)
MAIN_SYSTEMS = (
    *POLARIS_SYSTEMS,
    "audfp_m",
    "audfp_q",
    "panako",
    "olaf",
    "nmfp",
)
ABLATIONS = (
    "magnitude_maxima",
    "target_region",
)
SYSTEMS = (*MAIN_SYSTEMS, *ABLATIONS)
PROTOCOLS = {
    "pex": "pex_oracle_segment_exact_scale_v1",
    "sdrr": "sdrr_closed_set_retrieval_and_offset_v1",
}


def run(command: list[str], *, env: dict[str, str]) -> None:
    print("\n+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def complete(path: Path, expected: int, *, dataset: str, system: str) -> bool:
    if not path.is_file():
        return False
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if summary.get("schema") != "polaris-paper-results-v1":
        return False
    if summary.get("protocol") != PROTOCOLS[dataset] or summary.get("system") != system:
        return False
    count = summary.get("trials")
    errors = summary.get("errors", 0)
    return int(count or 0) == expected and int(errors or 0) == 0


def polaris(dataset: str, system: str, args: argparse.Namespace, env: dict[str, str]) -> None:
    summary = args.output / dataset / system / "summary.json"
    expected = 1_488 if dataset == "sdrr" else 791
    if not args.force and complete(summary, expected, dataset=dataset, system=system):
        print(f"reuse complete {dataset}/{system}")
        return
    command = [
        sys.executable,
        "-m",
        "polaris_eval.run",
        dataset,
        "--system",
        system,
        "--data-root",
        str(args.sdrr if dataset == "sdrr" else args.pex),
        "--output-root",
        str(args.output),
        "--workers",
        str(args.workers),
    ]
    if args.force:
        command.append("--force")
    run(command, env=env)


def audfprint(dataset: str, label: str, args: argparse.Namespace, env: dict[str, str]) -> None:
    python = ROOT / "benchmarks" / "audfprint" / ".venv" / "bin" / "python"
    source = ROOT / "benchmarks" / "audfprint" / ".cache" / "audfprint"
    expected = 1_488 if dataset == "sdrr" else 791
    if dataset == "sdrr":
        output = args.output / "sdrr" / label
        summary = output / "summary.json"
        if not args.force and complete(summary, expected, dataset=dataset, system=label):
            print(f"reuse complete sdrr/{label}")
            return
        run(
            [
                str(python),
                "benchmarks/audfprint/run_sdrr.py",
                str(args.sdrr),
                "--output",
                str(output),
                "--source-dir",
                str(source),
                "--profile",
                label,
                "--index-root",
                str(args.output / "shared" / "sdrr" / "audfp"),
                "--query-workers",
                str(args.workers),
            ],
            env=env,
        )
        return
    output = args.output / "pex" / label
    summary = output / "summary.json"
    if not args.force and complete(summary, expected, dataset=dataset, system=label):
        print(f"reuse complete pex/{label}")
        return
    command = [
        str(python),
        "benchmarks/audfprint/run_pex.py",
        str(args.pex),
        "--output",
        str(output),
        "--source-dir",
        str(source),
        "--index",
        str(args.output / "shared" / "pex" / "audfp.pklz"),
        "--profile",
        label,
        "--query-workers",
        str(args.workers),
    ]
    run(command, env=env)


def panako(dataset: str, args: argparse.Namespace, env: dict[str, str]) -> None:
    expected = 1_488 if dataset == "sdrr" else 791
    output = args.output / dataset / "panako"
    summary = output / "summary.json"
    if not args.force and complete(summary, expected, dataset=dataset, system="panako"):
        print(f"reuse complete {dataset}/panako")
        return
    jar = (
        ROOT
        / "benchmarks"
        / "panako"
        / ".cache"
        / "Panako"
        / "build"
        / "libs"
        / "panako-2.1-all.jar"
    )
    index = args.output / "shared" / dataset / "panako"
    if dataset == "sdrr":
        command = [
            sys.executable,
            "benchmarks/panako/run_sdrr.py",
            str(args.sdrr),
            "--output",
            str(output),
            "--index",
            str(index),
            "--jar",
            str(jar),
        ]
        if not (index / "lmdb").is_dir():
            command.append("--build-index")
    else:
        database = index / "lmdb"
        command = [
            sys.executable,
            "benchmarks/panako/run_pex.py",
            str(args.pex),
            "--output",
            str(output),
            "--index",
            str(database),
            "--jar",
            str(jar),
        ]
        if not database.is_dir():
            command.append("--build-index")
    run(command, env=env)


def olaf(dataset: str, args: argparse.Namespace, env: dict[str, str]) -> None:
    expected = 1_488 if dataset == "sdrr" else 791
    output = args.output / dataset / "olaf"
    summary = output / "summary.json"
    if not args.force and complete(summary, expected, dataset=dataset, system="olaf"):
        print(f"reuse complete {dataset}/olaf")
        return
    binary = ROOT / "benchmarks" / "olaf" / ".cache" / "Olaf" / "zig-out" / "bin" / "olaf"
    index = args.output / "shared" / dataset / "olaf"
    script = "benchmarks/olaf/run_sdrr.py" if dataset == "sdrr" else "benchmarks/olaf/run_pex.py"
    command = [
        sys.executable,
        script,
        str(args.sdrr if dataset == "sdrr" else args.pex),
        "--output",
        str(output),
        "--index",
        str(index),
        "--binary",
        str(binary),
        "--workers",
        str(args.workers),
    ]
    # The LMDB directory is created before the first store transaction.  Only
    # the adapter metadata is evidence that the complete reference collection
    # was committed; a directory or lock.mdb alone may be left by an interrupted run.
    if not (index / "olaf_index.json").is_file():
        command.append("--build-index")
    run(command, env=env)


def nmfp(dataset: str, args: argparse.Namespace, env: dict[str, str]) -> None:
    expected = 1_488 if dataset == "sdrr" else 791
    output = args.output / dataset / "nmfp"
    summary = output / "summary.json"
    if not args.force and complete(summary, expected, dataset=dataset, system="nmfp"):
        print(f"reuse complete {dataset}/nmfp")
        return
    python = ROOT / "benchmarks" / "nmfp" / ".venv" / "bin" / "python"
    source = ROOT / "benchmarks" / "nmfp" / ".cache" / "neural-music-fp"
    model = source / "logs" / "nmfp" / "fma-nmfp_deg" / "checkpoint" / "nmfp-triplet"
    script = "benchmarks/nmfp/run_sdrr.py" if dataset == "sdrr" else "benchmarks/nmfp/run_pex.py"
    command = [
        str(python),
        script,
        str(args.sdrr if dataset == "sdrr" else args.pex),
        "--output",
        str(output),
        "--source-dir",
        str(source),
        "--model-dir",
        str(model),
        "--reference-cache",
        str(args.output / "shared" / dataset / "nmfp_embeddings"),
    ]
    run(command, env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the complete POLARIS paper matrix.")
    parser.add_argument("--dataset", choices=("sdrr", "pex", "both"), default="both")
    parser.add_argument("--only", action="append", choices=SYSTEMS)
    parser.add_argument("--sdrr", type=Path, default=ROOT / "data" / "sdrr")
    parser.add_argument("--pex", type=Path, default=ROOT / "data" / "pex_hard_medium")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.sdrr = args.sdrr.resolve()
    args.pex = args.pex.resolve()
    args.output = args.output.resolve()
    datasets = ("sdrr", "pex") if args.dataset == "both" else (args.dataset,)
    selected = tuple(dict.fromkeys(args.only or SYSTEMS))
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(SRC) + os.pathsep + environment.get("PYTHONPATH", "")
    for dataset in datasets:
        for system in selected:
            if dataset == "pex" and system in ABLATIONS:
                continue
            if system in {*POLARIS_SYSTEMS, *ABLATIONS}:
                polaris(dataset, system, args, environment)
            elif system == "audfp_m":
                audfprint(dataset, "audfp_m", args, environment)
            elif system == "audfp_q":
                audfprint(dataset, "audfp_q", args, environment)
            elif system == "panako":
                panako(dataset, args, environment)
            elif system == "olaf":
                olaf(dataset, args, environment)
            elif system == "nmfp":
                nmfp(dataset, args, environment)
    if args.dataset == "both" and args.only is None:
        run(
            [sys.executable, "-m", "polaris_eval.report", "--outputs", str(args.output)],
            env=environment,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
