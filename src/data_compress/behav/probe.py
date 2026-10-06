"""Combine TIFF-header and camlog inspection into one read-only report."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data_compress.behav.camlog import find_camlog, parse_camlog
from data_compress.tiff_source import scan_tiff_dir


@dataclass(frozen=True)
class ProbeReport:
    tif_dir: Path
    files: list[Path]
    frame_count: int
    shape: tuple[int, int]
    dtype: np.dtype
    camlog_path: Path | None
    camlog_frame_count: int | None
    camlog_fps: float | None
    frame_count_mismatch: bool


def build_probe_report(tif_dir: Path) -> ProbeReport:
    info = scan_tiff_dir(tif_dir)

    camlog_path = find_camlog(tif_dir)
    camlog_frame_count: int | None = None
    camlog_fps: float | None = None
    mismatch = False
    if camlog_path is not None:
        camlog_info = parse_camlog(camlog_path)
        camlog_frame_count = camlog_info.frame_count
        camlog_fps = camlog_info.fps
        mismatch = camlog_frame_count != info.frame_count

    return ProbeReport(
        tif_dir=tif_dir,
        files=info.files,
        frame_count=info.frame_count,
        shape=info.shape,
        dtype=info.dtype,
        camlog_path=camlog_path,
        camlog_frame_count=camlog_frame_count,
        camlog_fps=camlog_fps,
        frame_count_mismatch=mismatch,
    )
