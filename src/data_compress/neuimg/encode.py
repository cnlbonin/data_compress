"""Losslessly compress imaging TIFF stacks to Zarr v3, verify them, and restore the TIFFs.

The output is one array `data` with dimensions (t, y, x), in the source dtype, with
one frame per chunk and Blosc zstd + bitshuffle. OME-NGFF 0.5 axis metadata lets
napari and Fiji open it directly. The source file names and per-file frame counts are
stored, so `decompress` rebuilds the original TIFF layout.

Compress and verify split frames into contiguous ranges, one range per worker process.
Decompress runs one source file per worker, so each output TIFF is written by one process.
Every chunk is written by exactly one worker, and the output does not depend on `--jobs`.
"""

from __future__ import annotations

import functools
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import zarr
from zarr.codecs import BloscCodec, BloscShuffle

from data_compress.parallel import dir_size, map_chunk_ranges, map_items, prepare_output, resolve_jobs
from data_compress.tiff_source import TiffStackInfo, iter_frame_range, scan_tiff_dir

DEFAULT_CLEVEL = 5
ARRAY_NAME = "data"
AXES = ("t", "y", "x")
SUPPORTED_KINDS = "uif"  # unsigned/signed int or float; Blosc shuffles any itemsize


def _check_dtype(dtype: np.dtype) -> None:
    if dtype.kind not in SUPPORTED_KINDS:
        raise ValueError(f"unsupported dtype {dtype}; expected an integer or float TIFF")


def _codec(clevel: int) -> BloscCodec:
    return BloscCodec(cname="zstd", clevel=clevel, shuffle=BloscShuffle.bitshuffle)


def _ome_attributes(source_name: str, info: TiffStackInfo, fps: float | None) -> dict[str, Any]:
    # Time is in seconds when fps is known; otherwise the scale stays at 1 frame per unit.
    time_scale = 1.0 / fps if fps is not None else 1.0
    return {
        "ome": {
            "version": "0.5",
            "multiscales": [
                {
                    "name": source_name,
                    "axes": [
                        {"name": "t", "type": "time"},
                        {"name": "y", "type": "space"},
                        {"name": "x", "type": "space"},
                    ],
                    "datasets": [
                        {
                            "path": ARRAY_NAME,
                            "coordinateTransformations": [
                                {"type": "scale", "scale": [time_scale, 1.0, 1.0]}
                            ],
                        }
                    ],
                    "version": "0.5",
                }
            ],
        },
        "source_dir": source_name,
        "source_files": [f.name for f in info.files],
        "frames_per_file": info.frames_per_file,
        "fps": fps,
    }


@dataclass(frozen=True)
class CompressResult:
    output: Path
    n_frames: int
    original_bytes: int
    compressed_bytes: int

    @property
    def ratio(self) -> float:
        return self.original_bytes / self.compressed_bytes


def _write_frames(start: int, stop: int, *, files: list[Path], frames_per_file: list[int], output: Path) -> None:
    data = zarr.open_group(output, mode="r+")[ARRAY_NAME]
    for index, frame in enumerate(iter_frame_range(files, frames_per_file, start, stop), start):
        data[index] = frame


def run_compress(
        tif_dir: Path,
        output: Path,
        *,
        fps: float | None = None,
        clevel: int = DEFAULT_CLEVEL,
        overwrite: bool = False,
        show_progress: bool = False,
        jobs: int | None = None,
) -> CompressResult:
    jobs = resolve_jobs(jobs)
    if output.suffix.lower() != ".zarr":
        raise ValueError(f"output {output} must have a .zarr extension")
    if not 0 <= clevel <= 9:
        raise ValueError(f"clevel must be in [0, 9], got {clevel}")
    if fps is not None and fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    info = scan_tiff_dir(tif_dir)
    _check_dtype(info.dtype)
    original_bytes = sum(f.stat().st_size for f in info.files)

    prepare_output(output, overwrite)
    try:
        root = zarr.open_group(output, mode="w", zarr_format=3)
        height, width = info.shape
        root.create_array(
            ARRAY_NAME,
            shape=(info.frame_count, height, width),
            chunks=(1, height, width),
            dtype=info.dtype,
            compressors=_codec(clevel),
            dimension_names=list(AXES),
        )
        root.attrs.update(
            {
                **_ome_attributes(tif_dir.name, info, fps),
                "compression": {"codec": "blosc-zstd", "clevel": clevel, "shuffle": "bitshuffle"},
            }
        )
        del root  # workers reopen the group; the array is already created

        map_chunk_ranges(
            _write_frames,
            info.frame_count,
            jobs=jobs,
            desc="compressing",
            show_progress=show_progress,
            files=info.files,
            frames_per_file=info.frames_per_file,
            output=output,
        )
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise

    return CompressResult(
        output=output,
        n_frames=info.frame_count,
        original_bytes=original_bytes,
        compressed_bytes=dir_size(output),
    )


@dataclass(frozen=True)
class VerifyResult:
    frame_count: int
    shape_matches: bool
    mismatched_frames: int

    @property
    def passed(self) -> bool:
        return self.shape_matches and self.mismatched_frames == 0


def _open_compressed(output: Path) -> zarr.Group:
    try:
        root = zarr.open_group(output, mode="r")
    except Exception as exc:  # zarr raises several unrelated types for "not a group"
        raise ValueError(f"{output} is not a Zarr group: {exc}") from exc
    if ARRAY_NAME not in root or "source_files" not in root.attrs:
        raise ValueError(f"{output} is not a recording written by `dc neuimg compress`")
    return root


def _verify_frames(start: int, stop: int, *, files: list[Path], frames_per_file: list[int], output: Path) -> int:
    data = zarr.open_group(output, mode="r")[ARRAY_NAME]
    mismatched = 0
    for index, frame in enumerate(iter_frame_range(files, frames_per_file, start, stop), start):
        if not np.array_equal(data[index], frame):
            mismatched += 1
    return mismatched


def verify(
        tif_dir: Path, output: Path, *, show_progress: bool = False, jobs: int | None = None
) -> VerifyResult:
    """Decode `output` frame by frame and compare each frame with the source TIFF stack."""
    jobs = resolve_jobs(jobs)
    info = scan_tiff_dir(tif_dir)
    root = _open_compressed(output)
    data = root[ARRAY_NAME]
    expected_shape = (info.frame_count, *info.shape)
    same_layout = (
        data.shape == expected_shape
        and data.dtype == info.dtype
        and root.attrs["frames_per_file"] == info.frames_per_file
    )
    if not same_layout:
        return VerifyResult(frame_count=info.frame_count, shape_matches=False, mismatched_frames=-1)

    mismatches = map_chunk_ranges(
        _verify_frames,
        info.frame_count,
        jobs=jobs,
        desc="verifying",
        show_progress=show_progress,
        files=info.files,
        frames_per_file=info.frames_per_file,
        output=output,
    )
    return VerifyResult(
        frame_count=info.frame_count, shape_matches=True, mismatched_frames=sum(mismatches)
    )


def _restore_file(task: tuple[str, int, int], *, output: Path, output_dir: Path) -> None:
    name, start, stop = task
    data = zarr.open_group(output, mode="r")[ARRAY_NAME]
    with tifffile.TiffWriter(output_dir / name, bigtiff=True) as writer:
        for index in range(start, stop):
            writer.write(np.asarray(data[index]), photometric="minisblack")


def run_decompress(
        output: Path,
        output_dir: Path,
        *,
        overwrite: bool = False,
        show_progress: bool = False,
        jobs: int | None = None,
) -> list[Path]:
    """Write the frames back as TIFF files with the original names and frame counts."""
    jobs = resolve_jobs(jobs)
    root = _open_compressed(output)
    names: list[str] = root.attrs["source_files"]
    counts: list[int] = root.attrs["frames_per_file"]

    targets = [output_dir / name for name in names]
    for path in targets:
        prepare_output(path, overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    start = 0
    for name, count in zip(names, counts):
        tasks.append((name, start, start + count))
        start += count

    try:
        map_items(
            functools.partial(_restore_file, output=output, output_dir=output_dir),
            tasks,
            jobs=jobs,
            desc="restoring",
            show_progress=show_progress,
        )
    except BaseException:
        for path in targets:
            path.unlink(missing_ok=True)
        raise
    return targets
