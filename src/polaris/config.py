"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 40_000
    window_size: int = 2_560
    hop_size: int = 1_280
    query_stft_shifts: int = 4

    @property
    def query_shift_spacing(self) -> int:
        return self.hop_size // self.query_stft_shifts


@dataclass(frozen=True)
class LandmarkConfig:
    magnitude_threshold: float = 10.0
    saliency_threshold: float = 5.0
    min_frequency_bin: int = 50
    max_frequency_bin: int = 350
    maximum_frequency_radius: int = 2
    maximum_time_radius: int = 3
    fine_frequency_sigma: float = 1.0
    fine_time_sigma: float = 1.5
    background_frequency_sigma: float = 8.0
    background_time_sigma: float = 16.0
    normalization_frequency_sigma: float = 6.0
    normalization_time_sigma: float = 12.0
    contrast_floor: float = 0.75
    contrast_weight: float = 3.0
    magnitude_weight: float = 0.015
    landmarks_per_second: float = 22.0
    min_landmarks_per_second: float = 6.0
    max_landmarks_per_second: float = 44.0
    density_window_seconds: float = 2.0
    density_background_seconds: float = 8.0
    density_adaptation_exponent: float = 0.75


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


@dataclass(frozen=True)
class QueryConfig:
    neighborhood_hops: int = 2
    max_neighbors_per_anchor: int = 12
    max_triangles_per_anchor: int = 24
    multiprobe_radius: int = 1


@dataclass(frozen=True)
class MatcherConfig:
    idf_smoothing: float = 1.0
    idf_power: float = 1.0
    offset_smoothing_radius: int = 1
    offset_neighbor_weight: float = 0.5
    topn: int = 2


@dataclass(frozen=True)
class AdaptiveConfig:
    min_matching_hashes: int = 8
    min_score_margin: float = 0.40


@dataclass(frozen=True)
class PolarisConfig:
    audio: AudioConfig = AudioConfig()
    landmarks: LandmarkConfig = LandmarkConfig()
    delaunay: DelaunayConfig = DelaunayConfig()
    query: QueryConfig = QueryConfig()
    matcher: MatcherConfig = MatcherConfig()
    adaptive: AdaptiveConfig = AdaptiveConfig()

    def to_dict(self) -> dict[str, object]:
        return {
            "name": "polaris",
            "audio": {
                **asdict(self.audio),
                "query_shift_spacing": self.audio.query_shift_spacing,
            },
            "landmarks": asdict(self.landmarks),
            "delaunay": asdict(self.delaunay),
            "query": asdict(self.query),
            "matcher": asdict(self.matcher),
            "adaptive": asdict(self.adaptive),
        }


DEFAULT_CONFIG = PolarisConfig()
