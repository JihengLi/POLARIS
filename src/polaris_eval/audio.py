"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydub import AudioSegment


def read(
    path: str | Path,
    fs: int | None = None,
) -> tuple[list[np.ndarray], int]:
    audio = AudioSegment.from_file(str(path))
    audio = audio.set_sample_width(2).set_frame_rate(fs or audio.frame_rate).set_channels(1)
    samples = np.frombuffer(audio.raw_data, dtype=np.int16)
    return [samples], audio.frame_rate
