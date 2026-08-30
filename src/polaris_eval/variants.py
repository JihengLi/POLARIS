"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

#!/usr/bin/env python3

from __future__ import annotations

import multiprocessing
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from polaris.config import DEFAULT_CONFIG, DelaunayConfig
from polaris.engine import read_audio
from polaris.fingerprint import (
    DelaunayGeometry,
    FingerprintLike,
    Landmark,
    SaliencyAnalysis,
    _descriptor_probes,
    _local_maximum_coordinates,
    compute_spectrogram,
    pack_descriptor,
)
from polaris.index import FileIndex, merge_file_index

ABLATION_SYSTEMS = (
    "magnitude_maxima",
    "two_hop_only",
    "dense_only",
    "canonical_only",
    "target_region",
)


class MagnitudeAnalysis(SaliencyAnalysis):
    """Change only the field defining candidate local maxima: S -> magnitude."""

    def _detect_candidates(self) -> list[Landmark]:
        config = self.config
        coordinates = _local_maximum_coordinates(
            self.band,
            frequency_radius=config.candidate_frequency_radius,
            time_radius=config.candidate_time_radius,
        )
        if coordinates.size == 0:
            return []
        amplitudes = self.band[coordinates[:, 0], coordinates[:, 1]].astype(np.float64)
        valid = amplitudes > config.amplitude_min
        coordinates = coordinates[valid]
        amplitudes = amplitudes[valid]
        points = [
            Landmark(
                frequency_bin=int(frequency_bin),
                time_bin=int(time_bin),
                score=float(amplitude + self.saliency[int(frequency_bin), int(time_bin)]),
            )
            for (frequency_bin, time_bin), amplitude in zip(coordinates, amplitudes, strict=True)
        ]
        points.sort(key=lambda point: (-point.score, point.time_bin, point.frequency_bin))
        return points


def _olaf_style_hashes(
    landmarks: list[Landmark],
    *,
    options: DelaunayConfig,
    max_landmark_usages: int = 12,
    packed: bool = False,
) -> set[FingerprintLike]:
    """OLAF-style sequential grouping with the unchanged POLARIS descriptor."""

    unique = {point.as_peak(): point for point in landmarks}
    ordered = sorted(unique.values(), key=lambda point: (point.time_bin, point.frequency_bin))
    normalized = np.asarray(
        [
            (point.frequency_bin / options.frequency_scale, point.time_bin / options.time_scale)
            for point in ordered
        ],
        dtype=np.float64,
    )
    usages = [0] * len(ordered)
    selected: list[tuple[int, int, int]] = []

    def valid_step(first: int, second: int) -> bool:
        delta_t = ordered[second].time_bin - ordered[first].time_bin
        delta_f = abs(ordered[second].frequency_bin - ordered[first].frequency_bin)
        return 1 <= delta_t <= 8 and 1 <= delta_f <= 128

    for anchor_index in range(len(ordered) - 2):
        if usages[anchor_index] >= max_landmark_usages:
            continue
        for second_index in range(anchor_index + 1, len(ordered) - 1):
            if ordered[second_index].time_bin - ordered[anchor_index].time_bin > 8:
                break
            if usages[second_index] >= max_landmark_usages or not valid_step(
                anchor_index, second_index
            ):
                continue
            for third_index in range(second_index + 1, len(ordered)):
                if ordered[third_index].time_bin - ordered[second_index].time_bin > 8:
                    break
                if (
                    usages[anchor_index] >= max_landmark_usages
                    or usages[second_index] >= max_landmark_usages
                ):
                    break
                if usages[third_index] >= max_landmark_usages or not valid_step(
                    second_index, third_index
                ):
                    continue
                indices = (anchor_index, second_index, third_index)
                coordinates = normalized[np.asarray(indices)]
                first_vector = coordinates[1] - coordinates[0]
                second_vector = coordinates[2] - coordinates[0]
                area = (
                    abs(
                        float(
                            first_vector[0] * second_vector[1] - first_vector[1] * second_vector[0]
                        )
                    )
                    / 2
                )
                if area < options.min_normalized_area:
                    continue
                edges = (
                    float(np.linalg.norm(coordinates[1] - coordinates[0])),
                    float(np.linalg.norm(coordinates[2] - coordinates[1])),
                    float(np.linalg.norm(coordinates[0] - coordinates[2])),
                )
                time_span = ordered[third_index].time_bin - ordered[anchor_index].time_bin
                if not options.min_time_span <= time_span <= options.max_time_span:
                    continue
                if max(edges) > options.max_normalized_edge:
                    continue
                if edges[0] * edges[1] * edges[2] / (4 * area) > options.max_circumradius:
                    continue
                selected.append(indices)
                usages[anchor_index] += 1
                usages[second_index] += 1
                usages[third_index] += 1

    hashes: set[FingerprintLike] = set()
    for anchor_index, second_index, third_index in selected:
        anchor, second, third = (
            ordered[anchor_index],
            ordered[second_index],
            ordered[third_index],
        )
        descriptor = [
            round(anchor.frequency_bin / options.anchor_frequency_quantization),
            round(
                (second.frequency_bin - anchor.frequency_bin) / options.delta_frequency_quantization
            ),
            round((second.time_bin - anchor.time_bin) / options.time_quantization),
            round(
                (third.frequency_bin - anchor.frequency_bin) / options.delta_frequency_quantization
            ),
            round((third.time_bin - anchor.time_bin) / options.time_quantization),
        ]
        for values in _descriptor_probes(descriptor, radius=0):
            key = pack_descriptor(values) if packed else "tri:" + ":".join(map(str, values))
            hashes.add((key, anchor.time_bin))
    return hashes


def _shift_hashes(
    samples: np.ndarray,
    sample_rate: int,
    *,
    shift_index: int,
    system: str,
    for_query: bool,
    packed: bool,
) -> set[FingerprintLike]:
    config = DEFAULT_CONFIG
    sample_shift = int(shift_index * config.audio.hop_length / config.audio.query_stft_shifts)
    spectrogram = compute_spectrogram(
        samples[sample_shift:],
        window_size=config.audio.window_size,
        hop_length=config.audio.hop_length,
    )
    analysis_type = MagnitudeAnalysis if system == "magnitude_maxima" else SaliencyAnalysis
    analysis = analysis_type(
        spectrogram,
        sample_rate=sample_rate,
        hop_length=config.audio.hop_length,
        config=config.reference_landmarks,
    )
    sparse = analysis.landmarks(config.reference_landmarks)
    if system == "target_region":
        return _olaf_style_hashes(
            sparse,
            options=config.reference_pairing,
            packed=packed,
        )
    geometry = DelaunayGeometry.prepare(sparse, config.reference_pairing)
    hashes = set(geometry.hashes(config.reference_pairing, packed=packed))
    if not for_query or system == "canonical_only":
        return hashes
    if system in {"magnitude_maxima", "two_hop_only", "polaris"}:
        hashes.update(geometry.hashes(config.same_density_query_pairing, packed=packed))
    if system in {"magnitude_maxima", "dense_only", "polaris"}:
        dense = analysis.landmarks(config.dense_query_landmarks)
        dense_geometry = DelaunayGeometry.prepare(dense, config.dense_query_pairing)
        dense_options = config.dense_query_pairing
        if system == "dense_only":
            from dataclasses import replace

            dense_options = replace(dense_options, neighborhood_hops=1)
        hashes.update(dense_geometry.hashes(dense_options, packed=packed))
    return hashes


def reference_key(system: str) -> str:
    if system == "magnitude_maxima":
        return "magnitude_maxima"
    if system == "target_region":
        return "target_region"
    return "polaris"


def variant_hashes(
    samples: np.ndarray,
    sample_rate: int,
    *,
    system: str,
    for_query: bool,
    packed: bool = False,
) -> set[FingerprintLike]:
    if system not in {*ABLATION_SYSTEMS, "polaris"}:
        raise ValueError(f"unsupported variant: {system}")
    shifts = DEFAULT_CONFIG.audio.query_stft_shifts if for_query else 1
    return {
        item
        for shift_index in range(shifts)
        for item in _shift_hashes(
            np.asarray(samples),
            sample_rate,
            shift_index=shift_index,
            system=system,
            for_query=for_query,
            packed=packed,
        )
    }


def _index_worker(arguments: tuple[str, str]) -> tuple[str, set[tuple[str, int]]]:
    filename, system = arguments
    samples = read_audio(filename, sample_rate=DEFAULT_CONFIG.audio.sample_rate)
    hashes = variant_hashes(
        samples,
        DEFAULT_CONFIG.audio.sample_rate,
        system=system,
        for_query=False,
        packed=False,
    )
    return Path(filename).stem, hashes  # type: ignore[return-value]


def build_variant_index(
    files: Sequence[Path],
    directory: Path,
    *,
    system: str,
    workers: int,
) -> int:
    index = FileIndex(directory, writable=True)
    missing = [path.resolve() for path in files if not index.is_song_fingerprinted(path.stem)]
    inputs = [(str(path), system) for path in missing]
    inserted = 0
    iterator = map(_index_worker, inputs)
    pool = None
    if workers > 1:
        pool = multiprocessing.Pool(workers)
        iterator = pool.imap_unordered(_index_worker, inputs)
    try:
        for song_name, hashes in iterator:
            song_id = index.insert_song(song_name, len(hashes))
            index.insert_hashes(song_id, hashes)
            inserted += 1
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        index.close()
    merge_file_index(directory)
    return inserted
