"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

#!/usr/bin/env python3

from pathlib import Path

import pytest

from polaris_eval.datasets import load_pex, load_sdrr
from polaris_eval.download import _verify_audio
from polaris_eval.io import EvaluationError, sha256


def test_missing_sdrr_has_actionable_error(tmp_path: Path):
    with pytest.raises(EvaluationError, match="manifest is missing"):
        load_sdrr(tmp_path)


def test_missing_pex_has_actionable_error(tmp_path: Path):
    with pytest.raises(EvaluationError, match="must contain"):
        load_pex(tmp_path)


def test_sdrr_audio_hash_verification(tmp_path: Path):
    reference = tmp_path / "references" / "001.mp3"
    query = tmp_path / "queries" / "001-q1.wav"
    reference.parent.mkdir()
    query.parent.mkdir()
    reference.write_bytes(b"reference")
    query.write_bytes(b"query")
    (tmp_path / "reference_manifest.csv").write_text(
        f"reference_file_name,reference_sha256\n001.mp3,{sha256(reference)}\n",
        encoding="utf-8",
    )
    (tmp_path / "manifest.csv").write_text(
        f"query_path,query_sha256\nqueries/001-q1.wav,{sha256(query)}\n",
        encoding="utf-8",
    )
    _verify_audio(tmp_path)
    query.write_bytes(b"corrupt")
    with pytest.raises(EvaluationError, match="SHA-256 mismatch"):
        _verify_audio(tmp_path)
