"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

#!/usr/bin/env python3

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from polaris import DEFAULT_CONFIG, Polaris, fingerprint_reference, prepare_query
from polaris.engine import build_reference_index
from polaris.fingerprint import pack_descriptor, pack_hash
from polaris.index import build_packed_index


def _signal(seconds: float = 3.0) -> np.ndarray:
    sample_rate = DEFAULT_CONFIG.audio.sample_rate
    time = np.arange(round(sample_rate * seconds)) / sample_rate
    wave = (
        0.45 * np.sin(2 * np.pi * (550 + 120 * time) * time)
        + 0.30 * np.sin(2 * np.pi * 1_100 * time)
        + 0.20 * np.sin(2 * np.pi * (1_900 - 80 * time) * time)
    )
    return np.asarray(np.clip(wave, -1, 1) * 30_000, dtype=np.int16)


def test_frozen_configuration_identity():
    config = DEFAULT_CONFIG.to_dict()
    assert config["name"] == "polaris"
    assert config["audio"]["sample_rate"] == 40_000
    assert config["landmarks"]["saliency_threshold"] == 5.0
    assert config["landmarks"]["landmarks_per_second"] == 22.0
    assert config["query"]["neighborhood_hops"] == 2


def test_packed_descriptor_round_trip_identity():
    descriptor = (25, -3, 4, 7, 12)
    assert pack_hash("tri:25:-3:4:7:12") == pack_descriptor(descriptor)


def test_query_stages_are_nested_and_deterministic():
    samples = _signal()
    first = fingerprint_reference(samples, 40_000)
    second = fingerprint_reference(samples, 40_000)
    prepared = prepare_query(samples, 40_000)
    original = prepared.hashes("original")
    two_hop = prepared.hashes("two_hop")
    assert first == second
    assert first
    assert original <= two_hop
    digest = hashlib.sha256(repr(sorted(first)).encode()).hexdigest()
    assert len(digest) == 64


def test_end_to_end_file_and_packed_index_agree(tmp_path: Path):
    sample_rate = DEFAULT_CONFIG.audio.sample_rate
    reference = tmp_path / "reference-a.wav"
    wavfile.write(reference, sample_rate, _signal())
    index = tmp_path / "index"
    assert build_reference_index([reference], index, workers=1) == 1

    with Polaris(index, index_backend="file") as recognizer:
        file_results = {
            mode: recognizer.recognize_file(reference, mode=mode, topn=1)
            for mode in ("o", "a", "f")
        }
    build_packed_index(index)
    with Polaris(index, index_backend="packed") as recognizer:
        packed_results = {
            mode: recognizer.recognize_file(reference, mode=mode, topn=1)
            for mode in ("o", "a", "f")
        }

    assert all(
        result["results"][0]["song_name"] == "reference-a"
        for result in file_results.values()
    )
    assert {
        mode: result["results"] for mode, result in packed_results.items()
    } == {mode: result["results"] for mode, result in file_results.items()}
