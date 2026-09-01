"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

#!/usr/bin/env python3

from __future__ import annotations

from collections.abc import Mapping, Sequence

PEX_RESULT_FIELDS = (
    "trial_id",
    "query_id",
    "reference_id",
    "query_begin",
    "status",
    "predicted_reference_id",
    "total_time",
    "error",
)

SDRR_RESULT_FIELDS = (
    "query_id",
    "reference_id",
    "status",
    "predicted_reference_id",
    "predicted_offset_seconds",
    "absolute_offset_error_seconds",
    "query_records",
    "total_time",
    "error",
)


def summarize_pex(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not rows:
        raise ValueError("PEX metrics require at least one trial")
    correct = sum(
        row.get("predicted_reference_id") == row.get("reference_id")
        for row in rows
    )
    return {
        "trials": len(rows),
        "errors": sum(str(row.get("status") or "") == "error" for row in rows),
        "track_top1": correct / len(rows),
    }


def summarize_sdrr(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("SD-RR metrics require at least one query")
    correct = [row.get("predicted_reference_id") == row.get("reference_id") for row in rows]
    track_and_offset = sum(
        track
        and row.get("absolute_offset_error_seconds") not in (None, "")
        and float(row["absolute_offset_error_seconds"]) <= 0.1
        for row, track in zip(rows, correct, strict=True)
    )
    return {
        "trials": len(rows),
        "errors": sum(str(row.get("status") or "") == "error" for row in rows),
        "track_top1": sum(correct) / len(rows),
        "track_and_offset_top1_at_0.1s": track_and_offset / len(rows),
    }
