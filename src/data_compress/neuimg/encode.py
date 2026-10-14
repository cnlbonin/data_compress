"""Losslessly compress imaging TIFF stacks or Scanbox `.sbx` recordings to Zarr v3, verify, and restore them.

Input is either a directory of TIFF files (stack of 2-D frames) or a single `.sbx`
file with its `.mat` sidecar (stack of (plane, y, x, channel) blocks). Either way the
output is one array `data`, in the source dtype, with one timepoint per chunk and Blosc
zstd + bitshuffle. OME-NGFF 0.5 axis metadata lets napari and Fiji open it directly.
For TIFF, the source file names and per-file frame counts are stored so `decompress`
rebuilds the original files; for `.sbx`, the raw `.mat` sidecar bytes are stored
verbatim alongside the pixel data so `decompress` restores both files byte-for-byte.

Compress and verify split frames into contiguous ranges, one range per worker process.
TIFF decompress runs one source file per worker; `.sbx` decompress runs one range of
timepoints per worker. Every chunk is written by exactly one worker, and the output
does not depend on `--jobs`.
"""

from __future__ import annotations

import functools
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import tifffile
import zarr
from zarr.codecs import BloscCodec

from data_compress.parallel import dir_size, map_chunk_ranges, map_items, prepare_output, resolve_jobs
from data_compress.sbx_source import SBX_SUFFIX, SbxInfo, iter_frame_range as _sbx_iter_frame_range, scan_sbx_file
from data_compress.tiff_source import TiffStackInfo, iter_frame_range as _tiff_iter_frame_range, scan_tiff_dir

DEFAULT_CLEVEL = 5
ARRAY_NAME = "data"
MAT_SIDECAR_NAME = "mat_sidecar"
SUPPORTED_KINDS = "uif"  # unsigned/signed int or float; Blosc shuffles any itemsize

SourceInfo = TiffStackInfo | SbxInfo


def _check_dtype(dtype: np.dtype) -> None:
    if dtype.kind not in SUPPORTED_KINDS:
        raise ValueError(f"unsupported dtype {dtype}; expected an integer or float source")


def _codec(clevel: int) -> BloscCodec:
    return BloscCodec(cname="zstd", clevel=clevel, shuffle="bitshuffle")


def scan_input(input_path: Path) -> SourceInfo:
    if input_path.is_dir():
        return scan_tiff_dir(input_path)
    if input_path.suffix.lower() == SBX_SUFFIX:
        return scan_sbx_file(input_path)
    raise ValueError(f"{input_path} is neither a TIFF directory nor a {SBX_SUFFIX} file")


def _frame_shape(info: SourceInfo) -> tuple[int, ...]:
    if isinstance(info, SbxInfo):
        return info.frame_shape
    return info.shape


def _axes_for(info: SourceInfo) -> tuple[tuple[str, str], ...]:
    """(name, type) pairs per OME-NGFF axis, matching the array's actual dimension order."""
    if isinstance(info, SbxInfo):
        return (("t", "time"), ("z", "space"), ("y", "space"), ("x", "space"), ("c", "channel"))
    return (("t", "time"), ("y", "space"), ("x", "space"))


def _iter_frames(info: SourceInfo, start: int, stop: int) -> Iterator[np.ndarray]:
    if isinstance(info, SbxInfo):
        return _sbx_iter_frame_range(info, start, stop)
    return _tiff_iter_frame_range(info.files, info.frames_per_file, start, stop)


def _ome_attributes(source_name: str, info: SourceInfo, fps: float | None) -> dict[str, Any]:
    # Time is in seconds when fps is known; otherwise the scale stays at 1 frame per unit.
    axes = _axes_for(info)
    time_scale = 1.0 / fps if fps is not None else 1.0
    scale = [time_scale] + [1.0] * (len(axes) - 1)
    ome = {
        "ome": {
            "version": "0.5",
            "multiscales": [
                {
                    "name": source_name,
                    "axes": [{"name": name, "type": kind} for name, kind in axes],
                    "datasets": [
                        {
                            "path": ARRAY_NAME,
                            "coordinateTransformations": [{"type": "scale", "scale": scale}],
                        }
                    ],
                    "version": "0.5",
                }
            ],
        },
        "fps": fps,
    }
    if isinstance(info, SbxInfo):
        return {
            **ome,
            "source_kind": "sbx",
            "sbx_stem": info.sbx_path.stem,
            "n_channels": info.n_channels,
            "n_planes": info.n_planes,
            "is_volumetric": info.is_volumetric,
        }
    return {
        **ome,
        "source_kind": "tiff",
        "source_dir": source_name,
        "source_files": [f.name for f in info.files],
        "frames_per_file": info.frames_per_file,
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


def _write_frames(start: int, stop: int, *, info: SourceInfo, output: Path) -> None:
    data = zarr.open_group(output, mode="r+")[ARRAY_NAME]
    for index, frame in enumerate(_iter_frames(info, start, stop), start):
        data[index] = frame


def run_compress(
        input_path: Path,
        output: Path,
        *,
        fps: float | None = None,
        clevel: int = DEFAULT_CLEVEL,
        overwrite: bool = False,
        show_progress: bool = False,
        on_progress: Callable[[int, int], None] | None = None,
        jobs: int | None = None,
) -> CompressResult:
    jobs = resolve_jobs(jobs)
    if output.suffix.lower() != ".zarr":
        raise ValueError(f"output {output} must have a .zarr extension")
    if not 0 <= clevel <= 9:
        raise ValueError(f"clevel must be in [0, 9], got {clevel}")
    if fps is not None and fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    info = scan_input(input_path)
    _check_dtype(info.dtype)
    if isinstance(info, SbxInfo):
        original_bytes = info.sbx_path.stat().st_size + info.mat_path.stat().st_size
    else:
        original_bytes = sum(f.stat().st_size for f in info.files)

    prepare_output(output, overwrite)
    try:
        root = zarr.open_group(output, mode="w", zarr_format=3)
        frame_shape = _frame_shape(info)
        root.create_array(
            ARRAY_NAME,
            shape=(info.frame_count, *frame_shape),
            chunks=(1, *frame_shape),
            dtype=info.dtype,
            compressors=_codec(clevel),
            dimension_names=[name for name, _ in _axes_for(info)],
        )
        if isinstance(info, SbxInfo):
            root.create_array(MAT_SIDECAR_NAME, data=np.frombuffer(info.mat_bytes, dtype=np.uint8))
        root.attrs.update(
            {
                **_ome_attributes(input_path.name, info, fps),
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
            on_progress=on_progress,
            info=info,
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
    has_marker = "source_kind" in root.attrs or "source_files" in root.attrs
    if ARRAY_NAME not in root or not has_marker:
        raise ValueError(f"{output} is not a recording written by `dc neuimg compress`")
    return root


def _same_layout(info: SourceInfo, root: zarr.Group) -> bool:
    data = root[ARRAY_NAME]
    if data.shape != (info.frame_count, *_frame_shape(info)) or data.dtype != info.dtype:
        return False
    if isinstance(info, SbxInfo):
        return bytes(np.asarray(root[MAT_SIDECAR_NAME][:])) == info.mat_bytes
    return root.attrs["frames_per_file"] == info.frames_per_file


def _verify_frames(start: int, stop: int, *, info: SourceInfo, output: Path) -> int:
    data = zarr.open_group(output, mode="r")[ARRAY_NAME]
    mismatched = 0
    for index, frame in enumerate(_iter_frames(info, start, stop), start):
        if not np.array_equal(data[index], frame):
            mismatched += 1
    return mismatched


def verify(
        input_path: Path, output: Path, *, show_progress: bool = False, jobs: int | None = None,
        on_progress: Callable[[int, int], None] | None = None
) -> VerifyResult:
    """Decode `output` frame by frame and compare each frame with the source."""
    jobs = resolve_jobs(jobs)
    info = scan_input(input_path)
    root = _open_compressed(output)
    if not _same_layout(info, root):
        return VerifyResult(frame_count=info.frame_count, shape_matches=False, mismatched_frames=-1)

    mismatches = map_chunk_ranges(
        _verify_frames,
        info.frame_count,
        jobs=jobs,
        desc="verifying",
        show_progress=show_progress,
        on_progress=on_progress,
        info=info,
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


def _decompress_tiff(
        root: zarr.Group, output: Path, output_dir: Path, *, overwrite: bool, show_progress: bool, jobs: int
) -> list[Path]:
    """Write the frames back as TIFF files with the original names and frame counts."""
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


def _restore_sbx_chunk(start: int, stop: int, *, output: Path, sbx_path: Path) -> None:
    data = zarr.open_group(output, mode="r")[ARRAY_NAME]
    frame_bytes = int(np.prod(data.shape[1:])) * data.dtype.itemsize
    with sbx_path.open("r+b") as f:
        for index in range(start, stop):
            f.seek(index * frame_bytes)
            f.write(np.ascontiguousarray(data[index]).tobytes())


def _decompress_sbx(
        root: zarr.Group, output: Path, output_dir: Path, *, overwrite: bool, show_progress: bool, jobs: int
) -> list[Path]:
    """Write the pixel data back to `<stem>.sbx` and the stored sidecar bytes to `<stem>.mat`."""
    stem = root.attrs["sbx_stem"]
    sbx_path = output_dir / f"{stem}.sbx"
    mat_path = output_dir / f"{stem}.mat"
    for path in (sbx_path, mat_path):
        prepare_output(path, overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = root[ARRAY_NAME]
    n_timepoints = data.shape[0]
    frame_bytes = int(np.prod(data.shape[1:])) * data.dtype.itemsize
    mat_bytes = bytes(np.asarray(root[MAT_SIDECAR_NAME][:]))

    try:
        mat_path.write_bytes(mat_bytes)
        with sbx_path.open("wb") as f:
            f.truncate(n_timepoints * frame_bytes)
        map_chunk_ranges(
            _restore_sbx_chunk,
            n_timepoints,
            jobs=jobs,
            desc="restoring",
            show_progress=show_progress,
            output=output,
            sbx_path=sbx_path,
        )
    except BaseException:
        sbx_path.unlink(missing_ok=True)
        mat_path.unlink(missing_ok=True)
        raise
    return [sbx_path, mat_path]


def run_decompress(
        output: Path,
        output_dir: Path,
        *,
        overwrite: bool = False,
        show_progress: bool = False,
        jobs: int | None = None,
) -> list[Path]:
    """Restore a compressed recording: original TIFFs, or `.sbx` + `.mat`."""
    jobs = resolve_jobs(jobs)
    root = _open_compressed(output)
    kind = root.attrs.get("source_kind", "tiff")
    if kind == "sbx":
        return _decompress_sbx(root, output, output_dir, overwrite=overwrite, show_progress=show_progress, jobs=jobs)
    return _decompress_tiff(root, output, output_dir, overwrite=overwrite, show_progress=show_progress, jobs=jobs)
