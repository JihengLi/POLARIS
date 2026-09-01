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
from scipy.ndimage import gaussian_filter

from polaris.config import DEFAULT_CONFIG
from polaris.engine import read_audio
from polaris.fingerprint import (
    DelaunayGeometry,
    FingerprintLike,
    Landmark,
    SaliencyAnalysis,
    analyze_shift,
    descriptor_probes,
    local_maxima,
    pack_descriptor,
)
from polaris.index import FileIndex, merge_file_index

CONTROL_SYSTEMS = ("magnitude_maxima", "target_region")


class MagnitudeAnalysis(SaliencyAnalysis):
    """Replace saliency maxima and ranking with smoothed-magnitude maxima."""

    def detect_candidates(self) -> list[Landmark]:
        config = self.config
        smoothed = gaussian_filter(
            self.band,
            sigma=(config.fine_frequency_sigma, config.fine_time_sigma),
            mode="nearest",
        )
        coordinates = local_maxima(
            smoothed,
            frequency_radius=config.maximum_frequency_radius,
            time_radius=config.maximum_time_radius,
        )
        if coordinates.size == 0:
            return []
        scores = smoothed[coordinates[:, 0], coordinates[:, 1]].astype(np.float64)
        coordinates = coordinates[scores > config.magnitude_threshold]
        scores = scores[scores > config.magnitude_threshold]
        points = [
            Landmark(int(frequency), int(time), float(score))
            for (frequency, time), score in zip(coordinates, scores, strict=True)
        ]
        points.sort(key=lambda point: (-point.score, point.time_bin, point.frequency_bin))
        return points


def _target_region_hashes(
    landmarks: list[Landmark],
    *,
    max_landmark_usages: int,
    multiprobe_radius: int,
    packed: bool,
) -> set[FingerprintLike]:
    config = DEFAULT_CONFIG.delaunay
    unique: dict[tuple[int, int], Landmark] = {}
    for point in landmarks:
        previous = unique.get(point.coordinate())
        if previous is None or point.score > previous.score:
            unique[point.coordinate()] = point
    ordered = sorted(
        unique.values(),
        key=lambda point: (point.time_bin, point.frequency_bin, -point.score),
    )
    normalized = np.asarray(
        [
            (point.frequency_bin / config.frequency_scale, point.time_bin / config.time_scale)
            for point in ordered
        ],
        dtype=np.float64,
    )
    usages = [0] * len(ordered)
    selected: list[tuple[int, int, int]] = []

    def valid_step(first: int, second: int) -> bool:
        time_delta = ordered[second].time_bin - ordered[first].time_bin
        frequency_delta = abs(ordered[second].frequency_bin - ordered[first].frequency_bin)
        return 1 <= time_delta <= 12 and 1 <= frequency_delta <= 128

    for anchor_index in range(len(ordered) - 2):
        if usages[anchor_index] >= max_landmark_usages:
            continue
        for second_index in range(anchor_index + 1, len(ordered) - 1):
            if ordered[second_index].time_bin - ordered[anchor_index].time_bin > 12:
                break
            if usages[second_index] >= max_landmark_usages:
                continue
            if not valid_step(anchor_index, second_index):
                continue
            for third_index in range(second_index + 1, len(ordered)):
                if ordered[third_index].time_bin - ordered[second_index].time_bin > 12:
                    break
                if usages[anchor_index] >= max_landmark_usages:
                    break
                if usages[second_index] >= max_landmark_usages:
                    break
                if usages[third_index] >= max_landmark_usages:
                    continue
                if not valid_step(second_index, third_index):
                    continue

                time_span = (
                    ordered[third_index].time_bin - ordered[anchor_index].time_bin
                )
                if not config.min_time_span <= time_span <= config.max_time_span:
                    continue
                indices = (anchor_index, second_index, third_index)
                coordinates = normalized[np.asarray(indices)]
                first = coordinates[1] - coordinates[0]
                second = coordinates[2] - coordinates[0]
                area = abs(float(first[0] * second[1] - first[1] * second[0])) / 2.0
                if area < config.min_normalized_area:
                    continue
                edges = (
                    float(np.linalg.norm(coordinates[1] - coordinates[0])),
                    float(np.linalg.norm(coordinates[2] - coordinates[1])),
                    float(np.linalg.norm(coordinates[0] - coordinates[2])),
                )
                if max(edges) > config.max_normalized_edge:
                    continue
                if edges[0] * edges[1] * edges[2] / (4.0 * area) > config.max_circumradius:
                    continue
                selected.append(indices)
                usages[anchor_index] += 1
                usages[second_index] += 1
                usages[third_index] += 1

    result: set[FingerprintLike] = set()
    for anchor_index, second_index, third_index in selected:
        anchor = ordered[anchor_index]
        second = ordered[second_index]
        third = ordered[third_index]
        descriptor = [
            round(anchor.frequency_bin / config.anchor_frequency_quantization),
            round(
                (second.frequency_bin - anchor.frequency_bin)
                / config.delta_frequency_quantization
            ),
            round((second.time_bin - anchor.time_bin) / config.time_quantization),
            round(
                (third.frequency_bin - anchor.frequency_bin)
                / config.delta_frequency_quantization
            ),
            round((third.time_bin - anchor.time_bin) / config.time_quantization),
        ]
        for values in descriptor_probes(descriptor, multiprobe_radius):
            key = (
                pack_descriptor(values)
                if packed
                else "tri:" + ":".join(str(value) for value in values)
            )
            result.add((key, anchor.time_bin))
    return result


def control_fingerprints(
    samples: np.ndarray,
    sample_rate: int,
    *,
    system: str,
    query: bool,
    packed: bool,
) -> set[FingerprintLike]:
    if system not in CONTROL_SYSTEMS:
        raise ValueError(f"unsupported control: {system}")
    config = DEFAULT_CONFIG
    shifts = config.audio.query_stft_shifts if query else 1
    result: set[FingerprintLike] = set()
    for shift_index in range(shifts):
        analysis_type = MagnitudeAnalysis if system == "magnitude_maxima" else SaliencyAnalysis
        analysis = analyze_shift(
            np.asarray(samples),
            sample_rate,
            shift_index,
            config,
            analysis_type,
        )
        landmarks = analysis.landmarks()
        if system == "target_region":
            result.update(
                _target_region_hashes(
                    landmarks,
                    max_landmark_usages=24,
                    multiprobe_radius=0,
                    packed=packed,
                )
            )
            if query:
                result.update(
                    _target_region_hashes(
                        landmarks,
                        max_landmark_usages=96,
                        multiprobe_radius=1,
                        packed=packed,
                    )
                )
            continue

        geometry = DelaunayGeometry(landmarks, config.delaunay)
        result.update(
            geometry.face_hashes(
                multiprobe_radius=(config.query.multiprobe_radius if query else 0),
                packed=packed,
            )
        )
        if query:
            result.update(
                geometry.two_hop_hashes(
                    neighborhood_hops=config.query.neighborhood_hops,
                    max_neighbors_per_anchor=config.query.max_neighbors_per_anchor,
                    max_triangles_per_anchor=config.query.max_triangles_per_anchor,
                    multiprobe_radius=config.query.multiprobe_radius,
                    packed=packed,
                )
            )
    return result


def reference_key(system: str) -> str:
    return system if system in CONTROL_SYSTEMS else "polaris"


def _index_worker(arguments: tuple[str, str]) -> tuple[str, set[tuple[str, int]]]:
    filename, system = arguments
    samples = read_audio(filename, sample_rate=DEFAULT_CONFIG.audio.sample_rate)
    hashes = control_fingerprints(
        samples,
        DEFAULT_CONFIG.audio.sample_rate,
        system=system,
        query=False,
        packed=False,
    )
    return Path(filename).stem, hashes  # type: ignore[return-value]


def build_control_index(
    files: Sequence[Path],
    directory: Path,
    *,
    system: str,
    workers: int,
) -> int:
    index = FileIndex(directory, writable=True)
    missing = sorted(
        (path.resolve() for path in files if not index.is_song_fingerprinted(path.stem)),
        key=lambda path: (path.stem, str(path)),
    )
    inputs = [(str(path), system) for path in missing]
    inserted = 0
    if workers == 1:
        iterator = map(_index_worker, inputs)
        for track_name, hashes in iterator:
            track_id = index.insert_song(track_name, len(hashes))
            index.insert_hashes(track_id, hashes)
            inserted += 1
    else:
        with multiprocessing.Pool(workers) as pool:
            for track_name, hashes in pool.imap(_index_worker, inputs):
                track_id = index.insert_song(track_name, len(hashes))
                index.insert_hashes(track_id, hashes)
                inserted += 1
    index.close()
    merge_file_index(directory)
    return inserted


def resolved_control(system: str) -> dict[str, object]:
    if system == "magnitude_maxima":
        return {
            "name": system,
            "change": "smoothed-magnitude maxima replace saliency maxima",
            "base": DEFAULT_CONFIG.to_dict(),
            "query_mode": "two_hop",
        }
    if system == "target_region":
        return {
            "name": system,
            "change": "OLAF-style target-region triplets replace Delaunay faces",
            "base": DEFAULT_CONFIG.to_dict(),
            "forward_time_frames": [1, 12],
            "forward_frequency_bins": [1, 128],
            "reference_max_landmark_usages": 24,
            "query_max_landmark_usages": 96,
            "query_multiprobe_radius": 1,
        }
    raise ValueError(f"unsupported control: {system}")
