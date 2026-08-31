"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

#!/usr/bin/env python3

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from polaris_eval.datasets import PexAnnotation


@dataclass(frozen=True)
class OracleSegment:
    annotation: PexAnnotation
    begin: float
    end: float

    @property
    def trial_id(self) -> str:
        return self.annotation.annotation_id


def build_oracle_segments(
    annotations: list[PexAnnotation],
) -> tuple[list[PexAnnotation], list[OracleSegment]]:
    selected = [annotation for annotation in annotations if annotation.exact_scale]
    segments = [
        OracleSegment(
            annotation,
            float(annotation.query_begin),
            float(annotation.query_end),
        )
        for annotation in selected
    ]
    return selected, segments


def group_oracle_segments(segments: list[OracleSegment]) -> dict[str, list[OracleSegment]]:
    grouped: dict[str, list[OracleSegment]] = defaultdict(list)
    for segment in segments:
        grouped[segment.annotation.query_id].append(segment)
    return dict(sorted(grouped.items()))
