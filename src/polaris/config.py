"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 40_000
    window_size: int = 2_560
    overlap_ratio: float = 0.5
    query_stft_shifts: int = 4

    @property
    def hop_length(self) -> int:
        return int(self.window_size * self.overlap_ratio)


@dataclass(frozen=True)
class SaliencyConfig:
    amplitude_min: float = 10.0
    min_frequency_bin: int = 50
    max_frequency_bin: int = 350
    candidate_frequency_radius: int = 2
    candidate_time_radius: int = 3
    fine_frequency_sigma: float = 1.0
    fine_time_sigma: float = 1.5
    background_frequency_sigma: float = 8.0
    background_time_sigma: float = 16.0
    normalization_frequency_sigma: float = 6.0
    normalization_time_sigma: float = 12.0
    contrast_floor: float = 0.75
    saliency_beta: float = 3.0
    amplitude_weight: float = 0.15
    landmarks_per_second: float = 22.0
    min_landmarks_per_second: float = 6.0
    max_landmarks_per_second: float = 44.0
    density_window_seconds: float = 2.0
    density_background_seconds: float = 8.0
    density_adaptation_exponent: float = 0.75

    def with_density(
        self,
        landmarks_per_second: float,
        min_landmarks_per_second: float,
        max_landmarks_per_second: float,
    ) -> SaliencyConfig:
        return replace(
            self,
            landmarks_per_second=landmarks_per_second,
            min_landmarks_per_second=min_landmarks_per_second,
            max_landmarks_per_second=max_landmarks_per_second,
        )


@dataclass(frozen=True)
class DelaunayConfig:
    frequency_scale: float = 10.0
    time_scale: float = 14.0
    min_time_span: int = 1
    max_time_span: int = 63
    min_normalized_area: float = 0.025
    max_normalized_edge: float = 6.0
    max_circumradius: float = 4.0
    anchor_frequency_quantization: float = 2.0
    delta_frequency_quantization: float = 3.0
    time_quantization: int = 2
    neighborhood_hops: int = 1
    max_neighbors_per_anchor: int = 0
    max_triangles_per_anchor: int = 0
    multiprobe_radius: int = 0

    def query_supergraph(self, *, max_triangles_per_anchor: int) -> DelaunayConfig:
        return replace(
            self,
            neighborhood_hops=2,
            max_neighbors_per_anchor=12,
            max_triangles_per_anchor=max_triangles_per_anchor,
            multiprobe_radius=1,
        )


@dataclass(frozen=True)
class MatcherConfig:
    idf_smoothing: float = 1.0
    idf_power: float = 1.0
    offset_smoothing_radius: int = 1
    offset_neighbor_weight: float = 0.5
    song_hash_normalization_exponent: float = 0.0
    topn: int = 2


@dataclass(frozen=True)
class AdaptiveQueryConfig:
    """Development-frozen stopping rule for track identification."""

    canonical_margin_fraction: float = 0.4
    canonical_min_matching_count: int = 8
    two_hop_margin_fraction: float = 0.15
    two_hop_min_matching_count: int = 12


@dataclass(frozen=True)
class PolarisConfig:
    audio: AudioConfig = AudioConfig()
    reference_landmarks: SaliencyConfig = SaliencyConfig()
    dense_query_landmarks: SaliencyConfig = SaliencyConfig().with_density(
        64.0,
        23.0,
        98.0,
    )
    reference_pairing: DelaunayConfig = DelaunayConfig()
    same_density_query_pairing: DelaunayConfig = DelaunayConfig().query_supergraph(
        max_triangles_per_anchor=24
    )
    dense_query_pairing: DelaunayConfig = DelaunayConfig().query_supergraph(
        max_triangles_per_anchor=16
    )
    matcher: MatcherConfig = MatcherConfig()
    adaptive_query: AdaptiveQueryConfig = AdaptiveQueryConfig()

    def to_dict(self) -> dict[str, object]:
        """Return JSON-stable metadata for the frozen implementation."""

        return {
            "name": "polaris",
            "audio": {
                **asdict(self.audio),
                "hop_length": self.audio.hop_length,
                "shift_spacing_samples": (self.audio.hop_length / self.audio.query_stft_shifts),
            },
            "reference_landmarks": asdict(self.reference_landmarks),
            "dense_query_landmarks": asdict(self.dense_query_landmarks),
            "reference_pairing": asdict(self.reference_pairing),
            "same_density_query_pairing": asdict(self.same_density_query_pairing),
            "dense_query_pairing": asdict(self.dense_query_pairing),
            "matcher": asdict(self.matcher),
            "adaptive_query": asdict(self.adaptive_query),
        }


DEFAULT_CONFIG = PolarisConfig()
