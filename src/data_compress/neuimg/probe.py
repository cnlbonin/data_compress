"""Summarise a widefield / cellular imaging TIFF stack in one read-only report."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data_compress.tiff_source import scan_tiff_dir


@dataclass(frozen=True)
class ProbeReport:
    tif_dir: Path
    files: list[Path]
    frame_count: int
    shape: tuple[int, int]
    dtype: np.dtype
    size_bytes: int


def build_probe_report(tif_dir: Path) -> ProbeReport:
    info = scan_tiff_dir(tif_dir)
    return ProbeReport(
        tif_dir=tif_dir,
        files=info.files,
        frame_count=info.frame_count,
        shape=info.shape,
        dtype=info.dtype,
        size_bytes=sum(f.stat().st_size for f in info.files),
    )
