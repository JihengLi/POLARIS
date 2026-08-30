"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

#!/usr/bin/env python3

from __future__ import annotations

from collections.abc import Mapping, Sequence


def summarize_single_reference_trials(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not rows:
        raise ValueError("PEX metrics require at least one trial")
    correct = sum(
        row.get("predicted_reference_id")
        == (row.get("reference_id") or row.get("expected_reference_id"))
        for row in rows
    )
    return {
        "trials": len(rows),
        "errors": sum(str(row.get("status") or "") == "error" for row in rows),
        "track_top1": correct / len(rows),
    }
