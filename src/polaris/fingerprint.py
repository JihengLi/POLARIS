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

from polaris.config import (
    DEFAULT_CONFIG,
    DelaunayConfig,
    PolarisConfig,
    SaliencyConfig,
)

Fingerprint = tuple[str, int]
PackedFingerprint = tuple[int, int]
FingerprintLike = Fingerprint | PackedFingerprint
QUERY_STAGES = ("canonical", "two_hop", "full")


def pack_descriptor(values: tuple[int, ...] | list[int]) -> int:
    """Pack the frozen five-coordinate triplet descriptor into 40 bits."""

    if len(values) != 5:
        raise ValueError("POLARIS descriptors must have five coordinates")
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
    """Convert a compatible ``tri:...`` text key to its packed integer key."""

    fields = fingerprint_hash.split(":")
    if len(fields) != 6 or fields[0] != "tri":
        raise ValueError(f"not a POLARIS triplet descriptor: {fingerprint_hash!r}")
    return pack_descriptor([int(value) for value in fields[1:]])


@dataclass(frozen=True)
class Landmark:
    frequency_bin: int
    time_bin: int
    score: float

    def as_peak(self) -> tuple[int, int]:
        return self.frequency_bin, self.time_bin


def compute_spectrogram(
    channel_samples: np.ndarray | list[int],
    *,
    window_size: int,
    hop_length: int,
) -> np.ndarray:
    """Return the exact dB representation used by the frozen experiments."""

    samples = np.asarray(channel_samples, dtype=float) / 32767.0
    spectrum = librosa.stft(
        y=samples,
        n_fft=window_size,
        hop_length=hop_length,
    )
    return librosa.amplitude_to_db(np.abs(spectrum), ref=np.max) + 80


def _local_maximum_coordinates(
    values: np.ndarray,
    *,
    frequency_radius: int,
    time_radius: int,
) -> np.ndarray:
    """Collapse each local-maximum plateau to its first deterministic cell."""

    neighborhood = (2 * frequency_radius + 1, 2 * time_radius + 1)
    maxima = maximum_filter(values, size=neighborhood, mode="nearest")
    candidates = values == maxima
    labels, component_count = label(
        candidates,
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
    best_indices = candidate_indices[
        flat_values[candidate_indices] == component_scores[flat_labels[candidate_indices]]
    ]
    _, first_per_component = np.unique(
        flat_labels[best_indices],
        return_index=True,
    )
    representatives = best_indices[first_per_component]
    frequencies, times = np.unravel_index(representatives, values.shape)
    return np.column_stack((frequencies, times)).astype(np.int64, copy=False)


class SaliencyAnalysis:
    """One reusable Q field and its complete, score-ordered peak population."""

    def __init__(
        self,
        spectrogram: np.ndarray,
        *,
        sample_rate: int,
        hop_length: int,
        config: SaliencyConfig,
    ) -> None:
        if spectrogram.ndim != 2:
            raise ValueError("spectrogram must be a two-dimensional array")

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.config = config
        lower = max(0, config.min_frequency_bin)
        upper = min(spectrogram.shape[0], config.max_frequency_bin)
        self.lower_frequency_bin = lower
        self.band = np.asarray(spectrogram[lower:upper], dtype=np.float32)
        if lower >= upper or spectrogram.shape[1] == 0:
            self.saliency = np.empty_like(self.band)
            self.ordered_candidates: list[Landmark] = []
            self.relative_temporal_energy = np.empty(0, dtype=np.float32)
            return

        self.saliency = self._compute_saliency(self.band, config)
        self.ordered_candidates = self._detect_candidates()
        self.relative_temporal_energy = self._relative_temporal_energy()

    @staticmethod
    def _compute_saliency(band: np.ndarray, config: SaliencyConfig) -> np.ndarray:
        fine = gaussian_filter(
            band,
            sigma=(config.fine_frequency_sigma, config.fine_time_sigma),
            mode="nearest",
        )
        background = gaussian_filter(
            fine,
            sigma=(
                config.background_frequency_sigma,
                config.background_time_sigma,
            ),
            mode="nearest",
        )
        contrast = fine - background
        local_power = gaussian_filter(
            contrast * contrast,
            sigma=(
                config.normalization_frequency_sigma,
                config.normalization_time_sigma,
            ),
            mode="nearest",
        )
        normalized_contrast = contrast / np.sqrt(local_power + config.contrast_floor**2)
        amplitude_prior = np.maximum(fine - config.amplitude_min, 0.0) / 10.0
        saliency = config.saliency_beta * normalized_contrast
        saliency += config.amplitude_weight * amplitude_prior
        return np.clip(saliency, -12.0, 12.0)

    def _detect_candidates(self) -> list[Landmark]:
        config = self.config
        coordinates = _local_maximum_coordinates(
            self.saliency,
            frequency_radius=config.candidate_frequency_radius,
            time_radius=config.candidate_time_radius,
        )
        if coordinates.size == 0:
            return []
        amplitudes = self.band[coordinates[:, 0], coordinates[:, 1]].astype(np.float64)
        coordinates = coordinates[amplitudes > config.amplitude_min]
        amplitudes = amplitudes[amplitudes > config.amplitude_min]
        points = [
            Landmark(
                frequency_bin=int(frequency_bin),
                time_bin=int(time_bin),
                score=float(amplitude + self.saliency[int(frequency_bin), int(time_bin)]),
            )
            for (frequency_bin, time_bin), amplitude in zip(
                coordinates,
                amplitudes,
                strict=True,
            )
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
        frames_per_second = self.sample_rate / self.hop_length
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

    def landmarks(self, density: SaliencyConfig) -> list[Landmark]:
        """Apply one density budget without recomputing Q or its peak set."""

        if not self.ordered_candidates:
            return []
        if self._field_signature(density) != self._field_signature(self.config):
            raise ValueError("density passes must share the same saliency field")

        rates = density.landmarks_per_second * np.power(
            self.relative_temporal_energy,
            density.density_adaptation_exponent,
        )
        rates = np.clip(
            rates,
            density.min_landmarks_per_second,
            density.max_landmarks_per_second,
        )
        window_frames = density.density_window_seconds * self.sample_rate / self.hop_length
        selected: list[Landmark] = []
        selected_times: list[int] = []
        for point in self.ordered_candidates:
            capacity = max(
                1,
                round(rates[point.time_bin] * density.density_window_seconds),
            )
            first_relevant = bisect_left(
                selected_times,
                point.time_bin - window_frames,
            )
            insertion_index = bisect_right(selected_times, point.time_bin)
            relevant_starts = selected_times[first_relevant:insertion_index]
            relevant_starts.append(point.time_bin)
            full = False
            for start in relevant_starts:
                count = bisect_right(selected_times, start + window_frames)
                count -= bisect_left(selected_times, start)
                if count >= capacity:
                    full = True
                    break
            if full:
                continue
            selected.append(point)
            insort(selected_times, point.time_bin)

        lower = self.lower_frequency_bin
        return [
            Landmark(
                frequency_bin=point.frequency_bin + lower,
                time_bin=point.time_bin,
                score=point.score,
            )
            for point in selected
        ]

    @staticmethod
    def _field_signature(config: SaliencyConfig) -> tuple[object, ...]:
        return (
            config.amplitude_min,
            config.min_frequency_bin,
            config.max_frequency_bin,
            config.candidate_frequency_radius,
            config.candidate_time_radius,
            config.fine_frequency_sigma,
            config.fine_time_sigma,
            config.background_frequency_sigma,
            config.background_time_sigma,
            config.normalization_frequency_sigma,
            config.normalization_time_sigma,
            config.contrast_floor,
            config.saliency_beta,
            config.amplitude_weight,
            config.density_window_seconds,
            config.density_background_seconds,
            config.density_adaptation_exponent,
        )


class DelaunayGeometry:
    """A reusable triangulation of one landmark set."""

    def __init__(
        self,
        landmarks: list[Landmark],
        *,
        frequency_scale: float,
        time_scale: float,
    ) -> None:
        unique: dict[tuple[int, int], Landmark] = {}
        for point in landmarks:
            previous = unique.get(point.as_peak())
            if previous is None or point.score > previous.score:
                unique[point.as_peak()] = point
        self.ordered = sorted(
            unique.values(),
            key=lambda point: (point.time_bin, point.frequency_bin, -point.score),
        )
        self.frequency_scale = frequency_scale
        self.time_scale = time_scale
        self.normalized = np.asarray(
            [
                (
                    point.frequency_bin / frequency_scale,
                    point.time_bin / time_scale,
                )
                for point in self.ordered
            ],
            dtype=np.float64,
        )
        self.simplices: set[tuple[int, int, int]] = set()
        self.adjacency: tuple[frozenset[int], ...] = tuple()
        self._rank_cache: dict[
            tuple[tuple[int, int, int], tuple[object, ...]], tuple[object, ...] | None
        ] = {}
        if len(self.ordered) < 3:
            return
        try:
            triangulation = Delaunay(self.normalized)
        except QhullError:
            return
        self.simplices = {
            tuple(sorted(int(index) for index in simplex)) for simplex in triangulation.simplices
        }
        adjacency: list[set[int]] = [set() for _ in self.ordered]
        for first, second, third in self.simplices:
            adjacency[first].update((second, third))
            adjacency[second].update((first, third))
            adjacency[third].update((first, second))
        self.adjacency = tuple(frozenset(neighbors) for neighbors in adjacency)

    @classmethod
    def prepare(
        cls,
        landmarks: list[Landmark],
        options: DelaunayConfig,
    ) -> DelaunayGeometry:
        return cls(
            landmarks,
            frequency_scale=options.frequency_scale,
            time_scale=options.time_scale,
        )

    def hashes(
        self,
        options: DelaunayConfig,
        *,
        packed: bool = False,
    ) -> list[FingerprintLike]:
        if options.frequency_scale != self.frequency_scale or options.time_scale != self.time_scale:
            raise ValueError("pairing pass cannot change prepared coordinate scales")
        if not self.simplices:
            return []

        candidates = set(self.simplices)
        if options.neighborhood_hops > 1:
            for anchor_index, anchor in enumerate(self.ordered):
                visited = {anchor_index}
                frontier = {anchor_index}
                for _ in range(options.neighborhood_hops):
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
                    and self.ordered[index].time_bin - anchor.time_bin <= options.max_time_span
                ]
                neighbors.sort(
                    key=lambda index: (
                        float(
                            np.linalg.norm(self.normalized[index] - self.normalized[anchor_index])
                        ),
                        -self.ordered[index].score,
                        self.ordered[index].time_bin,
                        self.ordered[index].frequency_bin,
                    )
                )
                if options.max_neighbors_per_anchor:
                    neighbors = neighbors[: options.max_neighbors_per_anchor]
                candidates.update(
                    tuple(sorted((anchor_index, first, second)))
                    for first, second in combinations(neighbors, 2)
                )

        ranked_by_anchor: dict[
            int,
            list[tuple[tuple[object, ...], tuple[int, int, int]]],
        ] = defaultdict(list)
        filter_signature = (
            options.min_time_span,
            options.max_time_span,
            options.min_normalized_area,
            options.max_normalized_edge,
            options.max_circumradius,
        )
        for indices in candidates:
            cache_key = (indices, filter_signature)
            if cache_key not in self._rank_cache:
                self._rank_cache[cache_key] = self._triangle_rank(indices, options)
            rank = self._rank_cache[cache_key]
            if rank is not None:
                ranked_by_anchor[indices[0]].append((rank, indices))

        selected: list[tuple[int, int, int]] = []
        for anchor_index in sorted(ranked_by_anchor):
            ranked = sorted(ranked_by_anchor[anchor_index])
            if options.max_triangles_per_anchor:
                ranked = ranked[: options.max_triangles_per_anchor]
            selected.extend(indices for _, indices in ranked)

        hashes: list[FingerprintLike] = []
        for anchor_index, second_index, third_index in selected:
            anchor = self.ordered[anchor_index]
            second = self.ordered[second_index]
            third = self.ordered[third_index]
            descriptor = [
                round(anchor.frequency_bin / options.anchor_frequency_quantization),
                round(
                    (second.frequency_bin - anchor.frequency_bin)
                    / options.delta_frequency_quantization
                ),
                round((second.time_bin - anchor.time_bin) / options.time_quantization),
                round(
                    (third.frequency_bin - anchor.frequency_bin)
                    / options.delta_frequency_quantization
                ),
                round((third.time_bin - anchor.time_bin) / options.time_quantization),
            ]
            for values in _descriptor_probes(
                descriptor,
                radius=options.multiprobe_radius,
            ):
                key = (
                    pack_descriptor(values)
                    if packed
                    else "tri:" + ":".join(str(value) for value in values)
                )
                hashes.append((key, anchor.time_bin))
        return hashes

    def _triangle_rank(
        self,
        indices: tuple[int, int, int],
        options: DelaunayConfig,
    ) -> tuple[object, ...] | None:
        anchor_index, _, third_index = indices
        time_span = self.ordered[third_index].time_bin - self.ordered[anchor_index].time_bin
        if not options.min_time_span <= time_span <= options.max_time_span:
            return None

        coordinates = self.normalized[np.asarray(indices)]
        first_vector = coordinates[1] - coordinates[0]
        second_vector = coordinates[2] - coordinates[0]
        twice_area = abs(
            float(first_vector[0] * second_vector[1] - first_vector[1] * second_vector[0])
        )
        area = twice_area / 2.0
        if area < options.min_normalized_area:
            return None
        edges = (
            float(np.linalg.norm(coordinates[1] - coordinates[0])),
            float(np.linalg.norm(coordinates[2] - coordinates[1])),
            float(np.linalg.norm(coordinates[0] - coordinates[2])),
        )
        if max(edges) > options.max_normalized_edge:
            return None
        circumradius = edges[0] * edges[1] * edges[2] / (4.0 * area)
        if circumradius > options.max_circumradius:
            return None
        quality = 4.0 * np.sqrt(3.0) * area / sum(edge * edge for edge in edges)
        minimum_score = min(self.ordered[index].score for index in indices)
        return (-quality, -minimum_score, circumradius, indices)


def _descriptor_probes(
    descriptor: list[int],
    *,
    radius: int,
) -> set[tuple[int, ...]]:
    descriptors = {tuple(descriptor)}
    for dimension in range(len(descriptor)):
        for distance in range(1, radius + 1):
            for direction in (-1, 1):
                probe = descriptor.copy()
                probe[dimension] += direction * distance
                descriptors.add(tuple(probe))
    return descriptors


def fingerprint_spectrogram(
    spectrogram: np.ndarray,
    *,
    sample_rate: int,
    config: PolarisConfig = DEFAULT_CONFIG,
    for_query: bool,
    packed: bool = False,
) -> set[FingerprintLike]:
    """Fingerprint one STFT phase while sharing all compatible intermediates."""

    analysis = SaliencyAnalysis(
        spectrogram,
        sample_rate=sample_rate,
        hop_length=config.audio.hop_length,
        config=config.reference_landmarks,
    )
    sparse_landmarks = analysis.landmarks(config.reference_landmarks)
    sparse_geometry = DelaunayGeometry.prepare(
        sparse_landmarks,
        config.reference_pairing,
    )
    hashes = set(sparse_geometry.hashes(config.reference_pairing, packed=packed))
    if not for_query:
        return hashes

    hashes.update(
        sparse_geometry.hashes(
            config.same_density_query_pairing,
            packed=packed,
        )
    )
    dense_landmarks = analysis.landmarks(config.dense_query_landmarks)
    dense_geometry = DelaunayGeometry.prepare(
        dense_landmarks,
        config.dense_query_pairing,
    )
    hashes.update(dense_geometry.hashes(config.dense_query_pairing, packed=packed))
    return hashes


class _PreparedShift:
    def __init__(
        self,
        analysis: SaliencyAnalysis,
        config: PolarisConfig,
        *,
        packed: bool,
    ) -> None:
        self.analysis = analysis
        self.config = config
        self.packed = packed
        sparse_landmarks = analysis.landmarks(config.reference_landmarks)
        self.sparse_geometry = DelaunayGeometry.prepare(
            sparse_landmarks,
            config.reference_pairing,
        )
        self._dense_geometry: DelaunayGeometry | None = None
        self._hashes: dict[str, set[FingerprintLike]] = {}

    def canonical_hashes(self) -> set[FingerprintLike]:
        if "canonical" not in self._hashes:
            self._hashes["canonical"] = set(
                self.sparse_geometry.hashes(
                    self.config.reference_pairing,
                    packed=self.packed,
                )
            )
        return self._hashes["canonical"]

    def two_hop_hashes(self) -> set[FingerprintLike]:
        if "two_hop" not in self._hashes:
            self._hashes["two_hop"] = set(
                self.sparse_geometry.hashes(
                    self.config.same_density_query_pairing,
                    packed=self.packed,
                )
            )
        return self._hashes["two_hop"]

    def dense_hashes(self) -> set[FingerprintLike]:
        if "dense" not in self._hashes:
            if self._dense_geometry is None:
                dense_landmarks = self.analysis.landmarks(self.config.dense_query_landmarks)
                self._dense_geometry = DelaunayGeometry.prepare(
                    dense_landmarks,
                    self.config.dense_query_pairing,
                )
            self._hashes["dense"] = set(
                self._dense_geometry.hashes(
                    self.config.dense_query_pairing,
                    packed=self.packed,
                )
            )
        return self._hashes["dense"]


class PreparedQuery:
    """Lazily expose cumulative query evidence at three retrieval stages."""

    def __init__(
        self,
        shifts: list[_PreparedShift],
    ) -> None:
        self.shifts = shifts
        self._cumulative: dict[str, set[FingerprintLike]] = {}

    def hashes(self, stage: str) -> set[FingerprintLike]:
        if stage not in QUERY_STAGES:
            raise ValueError(f"unknown query stage: {stage!r}")
        if stage not in self._cumulative:
            canonical = {item for shift in self.shifts for item in shift.canonical_hashes()}
            if stage == "canonical":
                self._cumulative[stage] = canonical
            else:
                two_hop = canonical | {
                    item for shift in self.shifts for item in shift.two_hop_hashes()
                }
                self._cumulative["two_hop"] = two_hop
                if stage == "full":
                    self._cumulative[stage] = two_hop | {
                        item for shift in self.shifts for item in shift.dense_hashes()
                    }
        return self._cumulative[stage]


def prepare_query(
    channel_samples: np.ndarray | list[int],
    sample_rate: int,
    *,
    config: PolarisConfig = DEFAULT_CONFIG,
    packed: bool = False,
) -> PreparedQuery:
    """Compute the shared STFT/Q state without eagerly generating all evidence."""

    shift_count = config.audio.query_stft_shifts
    if shift_count < 1:
        raise ValueError("query_stft_shifts must be positive")
    samples = np.asarray(channel_samples)
    shifts = []
    for shift_index in range(shift_count):
        sample_shift = int(shift_index * config.audio.hop_length / shift_count)
        spectrogram = compute_spectrogram(
            samples[sample_shift:],
            window_size=config.audio.window_size,
            hop_length=config.audio.hop_length,
        )
        analysis = SaliencyAnalysis(
            spectrogram,
            sample_rate=sample_rate,
            hop_length=config.audio.hop_length,
            config=config.reference_landmarks,
        )
        shifts.append(_PreparedShift(analysis, config, packed=packed))
    return PreparedQuery(shifts)


def fingerprint(
    channel_samples: np.ndarray | list[int],
    sample_rate: int,
    *,
    config: PolarisConfig = DEFAULT_CONFIG,
    for_query: bool = False,
    packed: bool = False,
) -> list[FingerprintLike]:
    """Generate reference or asymmetric query fingerprints."""

    if for_query:
        return list(
            prepare_query(
                channel_samples,
                sample_rate,
                config=config,
                packed=packed,
            ).hashes("full")
        )

    shift_count = config.audio.query_stft_shifts if for_query else 1
    if shift_count < 1:
        raise ValueError("query_stft_shifts must be positive")
    samples = np.asarray(channel_samples)
    hashes: set[FingerprintLike] = set()
    for shift_index in range(shift_count):
        sample_shift = int(shift_index * config.audio.hop_length / shift_count)
        spectrogram = compute_spectrogram(
            samples[sample_shift:],
            window_size=config.audio.window_size,
            hop_length=config.audio.hop_length,
        )
        hashes.update(
            fingerprint_spectrogram(
                spectrogram,
                sample_rate=sample_rate,
                config=config,
                for_query=for_query,
                packed=packed,
            )
        )
    return list(hashes)
