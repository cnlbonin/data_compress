"""Summarise an imaging TIFF stack or Scanbox `.sbx` recording in one read-only report."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data_compress.neuimg.encode import scan_input
from data_compress.sbx_source import SbxInfo


@dataclass(frozen=True)
class ProbeReport:
    input_path: Path
    frame_count: int
    shape: tuple[int, ...]
    dtype: np.dtype
    size_bytes: int
    files: list[Path] | None = None  # TIFF only
    n_channels: int | None = None  # .sbx only
    n_planes: int | None = None  # .sbx only
    is_volumetric: bool = False  # .sbx only


def build_probe_report(input_path: Path) -> ProbeReport:
    info = scan_input(input_path)
    if isinstance(info, SbxInfo):
        return ProbeReport(
            input_path=input_path,
            frame_count=info.frame_count,
            shape=info.frame_shape,
            dtype=info.dtype,
            size_bytes=info.sbx_path.stat().st_size + info.mat_path.stat().st_size,
            n_channels=info.n_channels,
            n_planes=info.n_planes,
            is_volumetric=info.is_volumetric,
        )
    return ProbeReport(
        input_path=input_path,
        frame_count=info.frame_count,
        shape=info.shape,
        dtype=info.dtype,
        size_bytes=sum(f.stat().st_size for f in info.files),
        files=info.files,
    )
