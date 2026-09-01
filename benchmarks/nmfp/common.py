"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
import yaml  # noqa: E402

ADAPTER_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = ADAPTER_DIR / ".cache" / "neural-music-fp"
DEFAULT_MODEL_DIR = (
    DEFAULT_SOURCE_DIR / "logs" / "nmfp" / "fma-nmfp_deg" / "checkpoint" / "nmfp-triplet"
)
SOURCE_COMMIT = "e95e2b4009751274b060a6b74c26ae1323daae59"

def _official_modules(source_dir: Path) -> tuple[Any, Any, Any]:
    if not (source_dir / "nmfp").is_dir():
        raise SystemExit(f"NMFP source tree is missing: {source_dir}")
    sys.path.insert(0, str(source_dir))
    import essentia.standard as es
    from nmfp.audio_processing import Melspec_layer, segment_audio
    from nmfp.model.utils import get_checkpoint_index_and_restore_model, get_fingerprinter

    return (
        es,
        (Melspec_layer, segment_audio),
        (get_fingerprinter, get_checkpoint_index_and_restore_model),
    )


def load_model(model_dir: Path, source_dir: Path) -> tuple[Any, Any, dict[str, Any], Any, Any]:
    tf.config.set_visible_devices([], "GPU")
    es, audio_modules, model_modules = _official_modules(source_dir)
    melspec_class, segment_audio = audio_modules
    get_fingerprinter, restore_model = model_modules

    config_path = model_dir / "config.yaml"
    if not config_path.is_file():
        raise SystemExit(f"NMFP checkpoint config is missing: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["TRAIN"]["MIXED_PRECISION"] = False
    frontend = melspec_class(
        segment_duration=config["MODEL"]["AUDIO"]["SEGMENT_DUR"],
        fs=config["MODEL"]["AUDIO"]["FS"],
        n_fft=config["MODEL"]["INPUT"]["STFT_WIN"],
        stft_hop=config["MODEL"]["INPUT"]["STFT_HOP"],
        n_mels=config["MODEL"]["INPUT"]["N_MELS"],
        f_min=config["MODEL"]["INPUT"]["F_MIN"],
        f_max=config["MODEL"]["INPUT"]["F_MAX"],
        dynamic_range=config["MODEL"]["INPUT"]["DYNAMIC_RANGE"],
        scale=config["MODEL"]["INPUT"]["SCALE"],
    )
    dummy_audio = np.zeros(
        int(config["MODEL"]["AUDIO"]["SEGMENT_DUR"] * config["MODEL"]["AUDIO"]["FS"]),
        dtype=np.float32,
    )
    dummy_mel = frontend.compute(dummy_audio)[None, :, :, None].astype(np.float32)
    model = get_fingerprinter(config, trainable=False)
    model(tf.convert_to_tensor(dummy_mel))
    restore_model(model, str(model_dir))
    return model, frontend, config, es, segment_audio


def infer_embeddings(
    audio: np.ndarray,
    model: Any,
    frontend: Any,
    segment_audio: Any,
    *,
    sample_rate: int,
    segment_seconds: float,
    hop_seconds: float,
    batch_size: int,
) -> np.ndarray:
    segment_length = int(round(sample_rate * segment_seconds))
    hop_length = int(round(sample_rate * hop_seconds))
    if len(audio) < segment_length:
        audio = np.pad(audio, (0, segment_length - len(audio)))
    segments, _ = segment_audio(
        np.asarray(audio, dtype=np.float32),
        L=segment_length,
        H=hop_length,
        discard_remainder=True,
    )
    embeddings = []
    for start in range(0, len(segments), batch_size):
        mel = frontend.compute_batch(segments[start : start + batch_size])
        mel = np.expand_dims(mel, axis=3).astype(np.float32)
        embeddings.append(np.asarray(model(tf.convert_to_tensor(mel))))
    return np.concatenate(embeddings, axis=0).astype(np.float32, copy=False)


def file_md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - upstream checkpoint publishes MD5
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_commit(source_dir: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
