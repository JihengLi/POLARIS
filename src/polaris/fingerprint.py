"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

#!/usr/bin/env python3

from __future__ import annotations

from bisect import bisect_left, bisect_right, insort
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

import librosa
import numpy as np
from scipy.ndimage import (
    gaussian_filter,
    gaussian_filter1d,
    generate_binary_structure,
    label,
    maximum_filter,
)
from scipy.spatial import Delaunay, QhullError

from polaris.config import DEFAULT_CONFIG, DelaunayConfig, LandmarkConfig, PolarisConfig

Fingerprint = tuple[str, int]
PackedFingerprint = tuple[int, int]
FingerprintLike = Fingerprint | PackedFingerprint
QUERY_STAGES = ("original", "two_hop")


def pack_descriptor(values: tuple[int, ...] | list[int]) -> int:
    if len(values) != 5:
        raise ValueError("POLARIS descriptors have five coordinates")
    anchor, *signed_values = values
    if not 0 <= anchor <= 255:
        raise ValueError(f"anchor-frequency coordinate is out of range: {anchor}")
    packed = int(anchor)
    for value in signed_values:
        if not -128 <= value <= 127:
            raise ValueError(f"signed descriptor coordinate is out of range: {value}")
        packed = (packed << 8) | (int(value) + 128)
    return packed


def pack_hash(fingerprint_hash: str) -> int:
    fields = fingerprint_hash.split(":")
    if len(fields) != 6 or fields[0] != "tri":
        raise ValueError(f"not a POLARIS triplet descriptor: {fingerprint_hash!r}")
    return pack_descriptor([int(value) for value in fields[1:]])


@dataclass(frozen=True)
class Landmark:
    frequency_bin: int
    time_bin: int
    score: float

    def coordinate(self) -> tuple[int, int]:
        return self.frequency_bin, self.time_bin


def compute_spectrogram(
    samples: np.ndarray | list[int],
    *,
    window_size: int,
    hop_size: int,
) -> np.ndarray:
    signal = np.asarray(samples, dtype=float) / 32767.0
    spectrum = librosa.stft(y=signal, n_fft=window_size, hop_length=hop_size)
    return librosa.amplitude_to_db(np.abs(spectrum), ref=np.max) + 80


def local_maxima(
    values: np.ndarray,
    *,
    frequency_radius: int,
    time_radius: int,
) -> np.ndarray:
    """Return one deterministic representative from every maximum plateau."""

    maxima = maximum_filter(
        values,
        size=(2 * frequency_radius + 1, 2 * time_radius + 1),
        mode="nearest",
    )
    labels, component_count = label(
        values == maxima,
        structure=generate_binary_structure(2, 2),
    )
    flat_labels = labels.ravel()
    flat_values = values.ravel()
    candidate_indices = np.flatnonzero(flat_labels)
    if candidate_indices.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    component_scores = np.full(component_count + 1, -np.inf)
    np.maximum.at(
        component_scores,
        flat_labels[candidate_indices],
        flat_values[candidate_indices],
    )
    best = candidate_indices[
        flat_values[candidate_indices] == component_scores[flat_labels[candidate_indices]]
    ]
    _, first = np.unique(flat_labels[best], return_index=True)
    frequencies, times = np.unravel_index(best[first], values.shape)
    return np.column_stack((frequencies, times)).astype(np.int64, copy=False)


class SaliencyAnalysis:
    """Compute the paper saliency field and apply its temporal rate controller."""

    def __init__(
        self,
        spectrogram: np.ndarray,
        *,
        sample_rate: int,
        hop_size: int,
        config: LandmarkConfig,
    ) -> None:
        if spectrogram.ndim != 2:
            raise ValueError("spectrogram must be two-dimensional")
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.config = config
        self.lower_frequency_bin = max(0, config.min_frequency_bin)
        upper = min(spectrogram.shape[0], config.max_frequency_bin)
        self.band = np.asarray(
            spectrogram[self.lower_frequency_bin : upper],
            dtype=np.float32,
        )
        if self.lower_frequency_bin >= upper or spectrogram.shape[1] == 0:
            self.saliency = np.empty_like(self.band)
            self.ordered_candidates: list[Landmark] = []
            self.relative_temporal_energy = np.empty(0, dtype=np.float32)
            return
        self.saliency = self.compute_saliency(self.band, config)
        self.ordered_candidates = self.detect_candidates()
        self.relative_temporal_energy = self._relative_temporal_energy()

    @staticmethod
    def compute_saliency(band: np.ndarray, config: LandmarkConfig) -> np.ndarray:
        smoothed = gaussian_filter(
            band,
            sigma=(config.fine_frequency_sigma, config.fine_time_sigma),
            mode="nearest",
        )
        background = gaussian_filter(
            smoothed,
            sigma=(config.background_frequency_sigma, config.background_time_sigma),
            mode="nearest",
        )
        contrast = smoothed - background
        contrast_energy = gaussian_filter(
            contrast * contrast,
            sigma=(
                config.normalization_frequency_sigma,
                config.normalization_time_sigma,
            ),
            mode="nearest",
        )
        normalized_contrast = contrast / np.sqrt(
            contrast_energy + config.contrast_floor**2
        )
        saliency = config.contrast_weight * normalized_contrast
        saliency += config.magnitude_weight * np.maximum(
            smoothed - config.magnitude_threshold,
            0.0,
        )
        return np.clip(saliency, -12.0, 12.0)

    def detect_candidates(self) -> list[Landmark]:
        coordinates = local_maxima(
            self.saliency,
            frequency_radius=self.config.maximum_frequency_radius,
            time_radius=self.config.maximum_time_radius,
        )
        if coordinates.size == 0:
            return []
        scores = self.saliency[coordinates[:, 0], coordinates[:, 1]].astype(np.float64)
        coordinates = coordinates[scores > self.config.saliency_threshold]
        scores = scores[scores > self.config.saliency_threshold]
        points = [
            Landmark(int(frequency), int(time), float(score))
            for (frequency, time), score in zip(coordinates, scores, strict=True)
        ]
        points.sort(key=lambda point: (-point.score, point.time_bin, point.frequency_bin))
        return points

    def _relative_temporal_energy(self) -> np.ndarray:
        if self.saliency.shape[1] == 0:
            return np.empty(0, dtype=np.float32)
        positive = np.maximum(self.saliency, 0.0)
        top_count = max(1, positive.shape[0] // 20)
        strongest = np.partition(
            positive,
            positive.shape[0] - top_count,
            axis=0,
        )[-top_count:]
        temporal_energy = np.mean(strongest, axis=0)
        frames_per_second = self.sample_rate / self.hop_size
        foreground_sigma = max(
            1.0,
            self.config.density_window_seconds * frames_per_second / 4,
        )
        background_sigma = max(
            foreground_sigma,
            self.config.density_background_seconds * frames_per_second / 4,
        )
        foreground = gaussian_filter1d(
            temporal_energy,
            sigma=foreground_sigma,
            mode="nearest",
        )
        background = gaussian_filter1d(
            temporal_energy,
            sigma=background_sigma,
            mode="nearest",
        )
        return (foreground + 0.25) / (background + 0.25)

    def landmarks(self) -> list[Landmark]:
        if not self.ordered_candidates:
            return []
        config = self.config
        rates = config.landmarks_per_second * np.power(
            self.relative_temporal_energy,
            config.density_adaptation_exponent,
        )
        rates = np.clip(
            rates,
            config.min_landmarks_per_second,
            config.max_landmarks_per_second,
        )
        window_frames = config.density_window_seconds * self.sample_rate / self.hop_size
        selected: list[Landmark] = []
        selected_times: list[int] = []
        for point in self.ordered_candidates:
            capacity = max(
                1,
                round(rates[point.time_bin] * config.density_window_seconds),
            )
            begin = bisect_left(selected_times, point.time_bin - window_frames)
            insertion = bisect_right(selected_times, point.time_bin)
            starts = selected_times[begin:insertion]
            starts.append(point.time_bin)
            if any(
                bisect_right(selected_times, start + window_frames)
                - bisect_left(selected_times, start)
                >= capacity
                for start in starts
            ):
                continue
            selected.append(point)
            insort(selected_times, point.time_bin)

        lower = self.lower_frequency_bin
        return [
            Landmark(point.frequency_bin + lower, point.time_bin, point.score)
            for point in selected
        ]


class DelaunayGeometry:
    """Reference faces and optional two-hop triples from one landmark set."""

    def __init__(self, landmarks: list[Landmark], config: DelaunayConfig) -> None:
        unique: dict[tuple[int, int], Landmark] = {}
        for point in landmarks:
            previous = unique.get(point.coordinate())
            if previous is None or point.score > previous.score:
                unique[point.coordinate()] = point
        self.landmarks = sorted(
            unique.values(),
            key=lambda point: (point.time_bin, point.frequency_bin, -point.score),
        )
        self.config = config
        self.coordinates = np.asarray(
            [
                (
                    point.frequency_bin / config.frequency_scale,
                    point.time_bin / config.time_scale,
                )
                for point in self.landmarks
            ],
            dtype=np.float64,
        )
        self.faces: set[tuple[int, int, int]] = set()
        self.adjacency: tuple[frozenset[int], ...] = tuple()
        self._rank_cache: dict[tuple[int, int, int], tuple[object, ...] | None] = {}
        if len(self.landmarks) < 3:
            return
        try:
            triangulation = Delaunay(self.coordinates)
        except QhullError:
            return
        self.faces = {
            tuple(sorted(int(index) for index in face))
            for face in triangulation.simplices
        }
        adjacency: list[set[int]] = [set() for _ in self.landmarks]
        for first, second, third in self.faces:
            adjacency[first].update((second, third))
            adjacency[second].update((first, third))
            adjacency[third].update((first, second))
        self.adjacency = tuple(frozenset(neighbors) for neighbors in adjacency)

    def face_hashes(
        self,
        *,
        multiprobe_radius: int,
        packed: bool,
    ) -> set[FingerprintLike]:
        if not self.faces:
            return set()
        selected = [indices for indices in self.faces if self._rank(indices) is not None]
        return self._hash_triangles(selected, multiprobe_radius, packed)

    def two_hop_hashes(
        self,
        *,
        neighborhood_hops: int,
        max_neighbors_per_anchor: int,
        max_triangles_per_anchor: int,
        multiprobe_radius: int,
        packed: bool,
    ) -> set[FingerprintLike]:
        """Hash capped additional triples; canonical Delaunay faces are excluded."""

        if not self.faces:
            return set()
        by_anchor: dict[int, list[tuple[tuple[object, ...], tuple[int, int, int]]]] = (
            defaultdict(list)
        )
        for anchor_index, anchor in enumerate(self.landmarks):
            visited = {anchor_index}
            frontier = {anchor_index}
            for _ in range(neighborhood_hops):
                frontier = {
                    neighbor
                    for index in frontier
                    for neighbor in self.adjacency[index]
                    if neighbor not in visited
                }
                if not frontier:
                    break
                visited.update(frontier)
            neighbors = [
                index
                for index in visited
                if index > anchor_index
                and self.landmarks[index].time_bin - anchor.time_bin
                <= self.config.max_time_span
            ]
            neighbors.sort(
                key=lambda index: (
                    float(np.linalg.norm(self.coordinates[index] - self.coordinates[anchor_index])),
                    -self.landmarks[index].score,
                    self.landmarks[index].time_bin,
                    self.landmarks[index].frequency_bin,
                )
            )
            neighbors = neighbors[:max_neighbors_per_anchor]
            for first, second in combinations(neighbors, 2):
                indices = tuple(sorted((anchor_index, first, second)))
                if indices in self.faces:
                    continue
                rank = self._rank(indices)
                if rank is not None:
                    by_anchor[anchor_index].append((rank, indices))

        selected = [
            indices
            for anchor_index in sorted(by_anchor)
            for _, indices in sorted(by_anchor[anchor_index])[:max_triangles_per_anchor]
        ]
        return self._hash_triangles(selected, multiprobe_radius, packed)

    def _hash_triangles(
        self,
        triangles: list[tuple[int, int, int]],
        multiprobe_radius: int,
        packed: bool,
    ) -> set[FingerprintLike]:

        result: set[FingerprintLike] = set()
        for indices in triangles:
            anchor, second, third = (self.landmarks[index] for index in indices)
            descriptor = [
                round(anchor.frequency_bin / self.config.anchor_frequency_quantization),
                round(
                    (second.frequency_bin - anchor.frequency_bin)
                    / self.config.delta_frequency_quantization
                ),
                round(
                    (second.time_bin - anchor.time_bin) / self.config.time_quantization
                ),
                round(
                    (third.frequency_bin - anchor.frequency_bin)
                    / self.config.delta_frequency_quantization
                ),
                round((third.time_bin - anchor.time_bin) / self.config.time_quantization),
            ]
            for values in descriptor_probes(descriptor, multiprobe_radius):
                key = (
                    pack_descriptor(values)
                    if packed
                    else "tri:" + ":".join(str(value) for value in values)
                )
                result.add((key, anchor.time_bin))
        return result

    def _rank(
        self,
        indices: tuple[int, int, int],
    ) -> tuple[object, ...] | None:
        if indices not in self._rank_cache:
            self._rank_cache[indices] = self._triangle_rank(indices)
        return self._rank_cache[indices]

    def _triangle_rank(
        self,
        indices: tuple[int, int, int],
    ) -> tuple[object, ...] | None:
        config = self.config
        time_span = (
            self.landmarks[indices[2]].time_bin - self.landmarks[indices[0]].time_bin
        )
        if not config.min_time_span <= time_span <= config.max_time_span:
            return None
        coordinates = self.coordinates[np.asarray(indices)]
        first = coordinates[1] - coordinates[0]
        second = coordinates[2] - coordinates[0]
        area = abs(float(first[0] * second[1] - first[1] * second[0])) / 2.0
        if area < config.min_normalized_area:
            return None
        edges = (
            float(np.linalg.norm(coordinates[1] - coordinates[0])),
            float(np.linalg.norm(coordinates[2] - coordinates[1])),
            float(np.linalg.norm(coordinates[0] - coordinates[2])),
        )
        if max(edges) > config.max_normalized_edge:
            return None
        circumradius = edges[0] * edges[1] * edges[2] / (4.0 * area)
        if circumradius > config.max_circumradius:
            return None
        quality = 4.0 * np.sqrt(3.0) * area / sum(edge * edge for edge in edges)
        minimum_saliency = min(self.landmarks[index].score for index in indices)
        return -quality, -minimum_saliency, circumradius, indices


def descriptor_probes(descriptor: list[int], radius: int) -> set[tuple[int, ...]]:
    probes = {tuple(descriptor)}
    for dimension in range(len(descriptor)):
        for distance in range(1, radius + 1):
            for direction in (-1, 1):
                shifted = descriptor.copy()
                shifted[dimension] += direction * distance
                probes.add(tuple(shifted))
    return probes


def analyze_shift(
    samples: np.ndarray,
    sample_rate: int,
    shift_index: int,
    config: PolarisConfig,
    analysis_type: type[SaliencyAnalysis] = SaliencyAnalysis,
) -> SaliencyAnalysis:
    shifted = samples[shift_index * config.audio.query_shift_spacing :]
    spectrogram = compute_spectrogram(
        shifted,
        window_size=config.audio.window_size,
        hop_size=config.audio.hop_size,
    )
    return analysis_type(
        spectrogram,
        sample_rate=sample_rate,
        hop_size=config.audio.hop_size,
        config=config.landmarks,
    )


class PreparedQuery:
    def __init__(self, geometries: list[DelaunayGeometry], config: PolarisConfig, packed: bool):
        self.geometries = geometries
        self.config = config
        self.packed = packed
        self._cache: dict[str, set[FingerprintLike]] = {}

    def hashes(self, stage: str) -> set[FingerprintLike]:
        if stage not in QUERY_STAGES:
            raise ValueError(f"unknown query stage: {stage!r}")
        if "original" not in self._cache:
            self._cache["original"] = {
                item
                for geometry in self.geometries
                for item in geometry.face_hashes(
                    multiprobe_radius=self.config.query.multiprobe_radius,
                    packed=self.packed,
                )
            }
        if stage == "two_hop" and "two_hop" not in self._cache:
            query = self.config.query
            self._cache[stage] = self._cache["original"] | {
                item
                for geometry in self.geometries
                for item in geometry.two_hop_hashes(
                    neighborhood_hops=query.neighborhood_hops,
                    max_neighbors_per_anchor=query.max_neighbors_per_anchor,
                    max_triangles_per_anchor=query.max_triangles_per_anchor,
                    multiprobe_radius=query.multiprobe_radius,
                    packed=self.packed,
                )
            }
        return self._cache[stage]


def prepare_query(
    samples: np.ndarray | list[int],
    sample_rate: int,
    *,
    config: PolarisConfig = DEFAULT_CONFIG,
    packed: bool = False,
) -> PreparedQuery:
    signal = np.asarray(samples)
    geometries = [
        DelaunayGeometry(
            analyze_shift(signal, sample_rate, shift_index, config).landmarks(),
            config.delaunay,
        )
        for shift_index in range(config.audio.query_stft_shifts)
    ]
    return PreparedQuery(geometries, config, packed)


def fingerprint_reference(
    samples: np.ndarray | list[int],
    sample_rate: int,
    *,
    config: PolarisConfig = DEFAULT_CONFIG,
    packed: bool = False,
) -> set[FingerprintLike]:
    analysis = analyze_shift(np.asarray(samples), sample_rate, 0, config)
    geometry = DelaunayGeometry(analysis.landmarks(), config.delaunay)
    return geometry.face_hashes(
        multiprobe_radius=0,
        packed=packed,
    )
