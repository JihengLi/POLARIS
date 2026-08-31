"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from polaris_eval.datasets import load_sdrr
from polaris_eval.io import EvaluationError, sha256

RECORD = "22169646"
BASE_URL = f"https://zenodo.org/records/{RECORD}/files"
FILES = {
    "CITATION.cff": "45e3f61ac8366ada7b133241b9762aab75b3eda0325038d206598c3d3e6c6940",
    "DATASET_CARD.md": "daf7831de1e351c74f73ad3d3e1614fc4ea9732696caaad2eb78d80b79b7d041",
    "LICENSES.md": "e29aae134ba2ce507a4602a8d642035d4f099e8b33cd0083f506a10454b17c95",
    "README.md": "d7074fcdca9edf59237d6e73f769ba0d6ffc9be0a2cd4e64ae3bfd1a7ca5c1ad",
    "SHA256SUMS": "4fe052f16e7e57855a62493362c0a14c5db4efb2517f5b774b085d5edf5b5fb1",
    "attribution.csv": "ff9fa4d0bc270589342ca1de0c10fc1fd4c9c62e7e4f224c31c3b71b11f4d5f5",
    "dataset_config.json": "aec6eb31a9f09212f024489d1953af722f4fb529ff293bcb64206a30c1a9dd2b",
    "dataset_summary.json": "6a6e84b1930043e3582d57282edcb240837ff8577a380f4f7ef1bd0db43ae6a9",
    "exclusions.csv": "665d6db51ce4407a660378d9169e30433635fd10e3e16fef80206433d08819a6",
    "manifest.csv": "5e6c66cdfcf17aa98e085a0de9a545096f44183c65787608270644f82556e0e5",
    "reference_manifest.csv": "1843cf0c1d9aa658b5b1009f2eb21107c08f6bfd8f4ac75c552162eab78c1f27",
    "sdrr-v1.0-queries.zip": "48c27ce1dbc0244d8461dd8ab17256d17afb3345041614c7f94a2e527776dbb5",
    "sdrr-v1.0-references.zip": "e3c33f1ab71080dfbac5f850324888fd4edba3d3de55cb223da8713233d7efdb",
}


def _download(name: str, destination: Path) -> None:
    request = urllib.request.Request(
        f"{BASE_URL}/{name}?download=1",
        headers={"User-Agent": "POLARIS-reproduction/1.0"},
    )
    with urllib.request.urlopen(request) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    observed = sha256(destination)
    if observed != FILES[name]:
        raise EvaluationError(f"SHA-256 mismatch for {name}: {observed}")


def _verify_audio(destination: Path) -> None:
    expected: dict[Path, str] = {}
    for manifest, path_field, digest_field in (
        ("reference_manifest.csv", "reference_file_name", "reference_sha256"),
        ("manifest.csv", "query_path", "query_sha256"),
    ):
        with (destination / manifest).open(encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                relative = (
                    Path("references") / row[path_field]
                    if manifest == "reference_manifest.csv"
                    else Path(row[path_field])
                )
                previous = expected.setdefault(relative, row[digest_field])
                if previous != row[digest_field]:
                    raise EvaluationError(f"conflicting SHA-256 values for {relative}")
    for number, (relative, digest) in enumerate(sorted(expected.items()), start=1):
        path = destination / relative
        if not path.is_file():
            raise EvaluationError(f"SD-RR file is missing: {relative}")
        observed = sha256(path)
        if observed != digest:
            raise EvaluationError(f"SHA-256 mismatch for {relative}: {observed}")
        if number % 250 == 0 or number == len(expected):
            print(f"verified {number}/{len(expected)} audio files", flush=True)


def download_sdrr(destination: Path) -> None:
    destination = destination.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        # Treat a fully valid existing release as success; never overwrite data.
        load_sdrr(destination)
        _verify_audio(destination)
        print(f"SD-RR is already complete: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sdrr-download-", dir=destination.parent) as tmp:
        temporary = Path(tmp)
        for name in FILES:
            print(f"download {name}", flush=True)
            _download(name, temporary / name)
        extracted = temporary / "extracted"
        extracted.mkdir()
        for name in ("sdrr-v1.0-references.zip", "sdrr-v1.0-queries.zip"):
            with zipfile.ZipFile(temporary / name) as archive:
                archive.extractall(extracted)
        for name in FILES:
            if not name.endswith(".zip"):
                shutil.move(temporary / name, extracted / name)
        load_sdrr(extracted)
        _verify_audio(extracted)
        if destination.exists():
            destination.rmdir()
        shutil.move(extracted, destination)
    print(f"SD-RR v1.0 ready: {destination}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and verify paper datasets.")
    parser.add_argument("dataset", choices=("sdrr",))
    parser.add_argument("--destination", type=Path, default=Path("data/sdrr"))
    args = parser.parse_args()
    download_sdrr(args.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
