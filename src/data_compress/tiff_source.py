"""Reading TIFF frame sequences: natural-sorted files, each possibly multi-page."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile


TIFF_SUFFIXES = (".tif", ".tiff")


def _natural_sort_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part for part in parts]


def _find_tiff_files(tif_dir: Path) -> list[Path]:
    files = [p for p in tif_dir.iterdir() if p.suffix.lower() in TIFF_SUFFIXES]
    if not files:
        raise ValueError(f"no .tif/.tiff files found in {tif_dir}")
    return sorted(files, key=_natural_sort_key)


@dataclass(frozen=True)
class TiffStackInfo:
    files: list[Path]
    frame_count: int
    shape: tuple[int, int]
    dtype: np.dtype
    frames_per_file: list[int]


def scan_tiff_dir(tif_dir: Path) -> TiffStackInfo:
    files = _find_tiff_files(tif_dir)

    frame_count = 0
    frames_per_file: list[int] = []
    shape: tuple[int, int] | None = None
    dtype: np.dtype | None = None
    for file in files:
        file_frames = 0
        with tifffile.TiffFile(file) as tif:
            for page in tif.pages:
                frame_count += 1
                file_frames += 1
                if shape is None:
                    shape = page.shape
                    dtype = page.dtype
                elif page.shape != shape or page.dtype != dtype:
                    raise ValueError(
                        f"inconsistent frame shape/dtype in {file.name} (frame {frame_count}): "
                        f"expected shape={shape} dtype={dtype}, got shape={page.shape} dtype={page.dtype}"
                    )
        frames_per_file.append(file_frames)

    assert shape is not None and dtype is not None
    return TiffStackInfo(
        files=files, frame_count=frame_count, shape=shape, dtype=dtype, frames_per_file=frames_per_file
    )


def iter_frames(tif_dir: Path) -> Iterator[np.ndarray]:
    for file in _find_tiff_files(tif_dir):
        with tifffile.TiffFile(file) as tif:
            for page in tif.pages:
                yield page.asarray()


def iter_frame_range(
        files: list[Path], frames_per_file: list[int], start: int, stop: int
) -> Iterator[np.ndarray]:
    """Yield global frames [start, stop) across the ordered files, opening each file once."""
    offset = 0
    for file, count in zip(files, frames_per_file):
        lo = max(start - offset, 0)
        hi = min(stop - offset, count)
        if lo < hi:
            with tifffile.TiffFile(file) as tif:
                for page in tif.pages[lo:hi]:
                    yield page.asarray()
        offset += count
        if offset >= stop:
            break
