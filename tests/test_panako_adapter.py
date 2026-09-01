"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path


def _adapter_module():
    path = Path(__file__).parents[1] / "benchmarks" / "panako" / "common.py"
    spec = importlib.util.spec_from_file_location("polaris_panako_common", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_panako_candidate_key_is_total_and_preserves_native_evidence_order() -> None:
    adapter = _adapter_module()

    def match(*, seconds: float, span: float):
        return adapter.PanakoWindowMatch(
            query_path=Path("query.wav"),
            query_start=0.0,
            reference_path=Path("reference.wav"),
            reference_start=3.0,
            reference_stop=3.0 + span,
            score=2.0,
            seconds_with_match=seconds,
        )

    candidates = [
        ("150663", match(seconds=1.0, span=0.064)),
        ("143314", match(seconds=1.0, span=0.456)),
        ("999999", match(seconds=math.nan, span=10.0)),
    ]
    ranked = sorted(
        candidates,
        key=lambda item: adapter._match_rank_key(item[0], item[1]),
        reverse=True,
    )

    assert [reference_id for reference_id, _ in ranked] == ["143314", "150663", "999999"]
