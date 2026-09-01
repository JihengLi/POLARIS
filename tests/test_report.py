"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

from polaris_eval.metrics import summarize_pex, summarize_sdrr
from polaris_eval.report import (
    _correct,
    _mean_query_payload,
    _protocol_reference_seconds,
    _trial_key,
)


def test_pex_rows_have_a_common_pairing_key() -> None:
    first = {
        "trial_id": "129795:query0000:0:10:0:10",
        "reference_id": "129795",
        "predicted_reference_id": "129795",
    }
    second = dict(first)
    assert _trial_key("pex", first) == _trial_key("pex", second)
    assert _correct(first) == 1


def test_hash_query_records_are_counted() -> None:
    payload = _mean_query_payload([{"query_records": "1190"}], "hash")
    assert payload == 1190 * 12 / (1024**2)


def test_protocol_duration_comes_from_polaris_summary(tmp_path: Path) -> None:
    summary = tmp_path / "sdrr" / "polaris_o" / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(
        json.dumps({"index": {"reference_audio_seconds": 123.5}}),
        encoding="utf-8",
    )
    assert _protocol_reference_seconds(tmp_path, "sdrr") == 123.5


def test_metric_summaries_expose_only_frozen_paper_metrics() -> None:
    pex_rows = [
        {"reference_id": "a", "predicted_reference_id": "a"},
        {"reference_id": "b", "predicted_reference_id": "x"},
    ]
    assert summarize_pex(pex_rows) == {
        "trials": 2,
        "errors": 0,
        "track_top1": 0.5,
    }
    sdrr_rows = [
        {
            "reference_id": "a",
            "predicted_reference_id": "a",
            "absolute_offset_error_seconds": 0.08,
        },
        {
            "reference_id": "b",
            "predicted_reference_id": "x",
            "absolute_offset_error_seconds": "",
        },
    ]
    assert summarize_sdrr(sdrr_rows) == {
        "trials": 2,
        "errors": 0,
        "track_top1": 0.5,
        "track_and_offset_top1_at_0.1s": 0.5,
    }
