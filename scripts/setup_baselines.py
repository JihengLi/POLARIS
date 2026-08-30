"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ("audfprint", "panako", "olaf", "nmfp")
REVISIONS = {
    "audfprint": "cb03ba99feafd41b8874307f0f4e808a6ce34362",
    "panako": "e4b0e1dbb55e340bc66c90bac0ceb82b2cf84211",
    "olaf": "532f1991ba170b39d2156b43935428814e833614",
    "nmfp": "e95e2b4009751274b060a6b74c26ae1323daae59",
}


def run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def clone(repository: str, destination: Path, commit: str) -> None:
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", repository, str(destination)])
    observed = subprocess.run(
        ["git", "-C", str(destination), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed != commit:
        run(["git", "-C", str(destination), "fetch", "origin", commit])
        run(["git", "-C", str(destination), "checkout", "--detach", commit])
    observed = subprocess.run(
        ["git", "-C", str(destination), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed != commit:
        raise SystemExit(f"expected {commit}, found {observed} in {destination}")


def setup_audfprint() -> None:
    directory = ROOT / "benchmarks" / "audfprint"
    clone(
        "https://github.com/dpwe/audfprint.git",
        directory / ".cache" / "audfprint",
        REVISIONS["audfprint"],
    )
    run(["uv", "sync", "--project", str(directory), "--python", "3.11"])


def setup_panako() -> None:
    directory = ROOT / "benchmarks" / "panako"
    source = directory / ".cache" / "Panako"
    clone("https://github.com/JorenSix/Panako.git", source, REVISIONS["panako"])
    environment = dict(os.environ)
    if platform.system() == "Darwin" and Path("/usr/libexec/java_home").is_file():
        java_home = subprocess.run(
            ["/usr/libexec/java_home", "-v", "17"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        environment["JAVA_HOME"] = java_home
    run([str(source / "gradlew"), "shadowJar"], cwd=source, env=environment)


def setup_olaf() -> None:
    script = ROOT / "benchmarks" / "olaf" / "prepare_source.py"
    run([sys.executable, str(script)])


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - upstream release publishes MD5
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def setup_nmfp() -> None:
    directory = ROOT / "benchmarks" / "nmfp"
    source = directory / ".cache" / "neural-music-fp"
    clone("https://github.com/raraz15/neural-music-fp.git", source, REVISIONS["nmfp"])
    checkpoint = source / "logs" / "nmfp" / "fma-nmfp_deg" / "checkpoint" / "nmfp-triplet"
    if not (checkpoint / "ckpt-100.index").is_file():
        cache = directory / ".cache"
        cache.mkdir(parents=True, exist_ok=True)
        archive = cache / "nmfp-triplet.zip"
        urllib.request.urlretrieve(
            "https://zenodo.org/records/15719945/files/nmfp-triplet.zip?download=1",
            archive,
        )
        if _md5(archive) != "ee8a3358fc5e5cdd09d6d2245d395021":
            raise SystemExit("NMFP checkpoint MD5 mismatch")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as compressed:
            compressed.extractall(checkpoint.parent)
        archive.unlink()
    run(["uv", "sync", "--project", str(directory), "--python", "3.11"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", action="append", choices=BASELINES)
    args = parser.parse_args()
    selected = tuple(dict.fromkeys(args.only or BASELINES))
    missing = [name for name in ("git", "uv", "ffmpeg") if shutil.which(name) is None]
    if missing:
        raise SystemExit("missing required executables: " + ", ".join(missing))
    actions = {
        "audfprint": setup_audfprint,
        "panako": setup_panako,
        "olaf": setup_olaf,
        "nmfp": setup_nmfp,
    }
    for name in selected:
        print(f"\n== {name} ==", flush=True)
        actions[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
