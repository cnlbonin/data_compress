"""Compress SpikeGLX recordings to Zarr with WavPack, verify, and restore them.

Follows Buccino et al. (eLife 2025, doi:10.7554/eLife.110170): WavPack via
wavpack-numcodecs, 1-s chunks over all channels, lossless by default and lossy
only at an explicit bits-per-sample (bps) target. The output layout matches
SpikeInterface's Zarr recordings (`traces_seg0`, `channel_ids`,
`sampling_frequency`) so it can be loaded later with `spikeinterface.read_zarr`.

Compress, verify and restore work on contiguous ranges of 1-s chunks. Each range
runs in its own worker process (WavPack encoding holds the GIL, so threads do not
help). Every range covers whole chunks, so workers never write to the same Zarr
chunk or the same byte range of the restored `.bin`.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import zarr
from wavpack_numcodecs import WavPack

from data_compress.ephys.spikeglx import SAMPLE_DTYPE, find_meta, open_bin, parse_meta
from data_compress.parallel import (
    chunk_bounds,
    dir_size,
    map_chunk_ranges,
    n_chunks,
    prepare_output,
    resolve_jobs,
)

DEFAULT_LEVEL = 3
MIN_BPS = 2.25  # lowest bps WavPack's lossy mode supports
MAX_BPS = 16.0  # int16 data: at 16 bps "lossy" is no smaller than the raw samples
CHUNK_SECONDS = 1.0


def validate_bps(bps: float | None) -> None:
    if bps is not None and not MIN_BPS <= bps < MAX_BPS:
        raise ValueError(f"bps must be in [{MIN_BPS}, {MAX_BPS}) for lossy WavPack, got {bps}")


@dataclass(frozen=True)
class CompressResult:
    output: Path
    n_samples: int
    original_bytes: int
    compressed_bytes: int

    @property
    def ratio(self) -> float:
        return self.original_bytes / self.compressed_bytes


def _compress_chunks(start_chunk: int, stop_chunk: int, *, bin_path: Path, output: Path, chunk: int) -> None:
    meta = parse_meta(find_meta(bin_path))
    data = open_bin(bin_path, meta)
    n_neural = meta.n_neural_chans
    root = zarr.open_group(output, mode="r+")
    traces = root["traces_seg0"]
    sync = root["sync_seg0"] if "sync_seg0" in root else None
    for index in range(start_chunk, stop_chunk):
        start, stop = chunk_bounds(index, chunk, data.shape[0])
        block = np.asarray(data[start:stop])
        traces[start:stop] = block[:, :n_neural]
        if sync is not None:
            sync[start:stop] = block[:, n_neural:]


def run_compress(
        bin_path: Path,
        output: Path,
        *,
        bps: float | None = None,
        level: int = DEFAULT_LEVEL,
        overwrite: bool = False,
        show_progress: bool = False,
        on_progress: Callable[[int, int], None] | None = None,
        jobs: int | None = None,
) -> CompressResult:
    validate_bps(bps)
    jobs = resolve_jobs(jobs)
    if output.suffix.lower() != ".zarr":
        raise ValueError(f"output {output} must have a .zarr extension")

    meta = parse_meta(find_meta(bin_path))
    data = open_bin(bin_path, meta)
    n_samples = data.shape[0]
    n_neural = meta.n_neural_chans
    chunk = max(1, int(round(meta.sample_rate * CHUNK_SECONDS)))

    prepare_output(output, overwrite)
    try:
        # zarr_format=2: wavpack-numcodecs is a numcodecs (v2-style) codec, and
        # SpikeInterface writes its Zarr recordings in this format too.
        root = zarr.open_group(output, mode="w", zarr_format=2)
        root.attrs.update(
            {
                "sampling_frequency": meta.sample_rate,
                "num_segments": 1,
                "source_file": bin_path.name,
                "spikeglx_meta": meta.raw,
                "spikeglx_meta_text": meta.text,
                "compression": {
                    "codec": "wavpack",
                    "level": level,
                    "bps": bps,
                    "wavpack_numcodecs": version("wavpack-numcodecs"),
                    "sync": "lossless",
                },
            }
        )
        root.create_array("channel_ids", data=np.arange(n_neural))
        root.create_array(
            "traces_seg0",
            shape=(n_samples, n_neural),
            chunks=(chunk, n_neural),
            dtype=SAMPLE_DTYPE,
            compressors=WavPack(level=level, bps=bps),
        )
        if meta.n_sync_chans:
            root.create_array(
                "sync_seg0",
                shape=(n_samples, meta.n_sync_chans),
                chunks=(chunk, meta.n_sync_chans),
                dtype=SAMPLE_DTYPE,
                compressors=WavPack(level=level),  # TTL bits: always lossless
            )
        del root  # workers reopen the group; the arrays are already created

        map_chunk_ranges(
            _compress_chunks,
            n_chunks(n_samples, chunk),
            jobs=jobs,
            desc="compressing",
            show_progress=show_progress,
            on_progress=on_progress,
            bin_path=bin_path,
            output=output,
            chunk=chunk,
        )
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise

    return CompressResult(
        output=output,
        n_samples=n_samples,
        original_bytes=bin_path.stat().st_size,
        compressed_bytes=dir_size(output),
    )


@dataclass(frozen=True)
class VerifyResult:
    bps: float | None  # lossy target the output was written with; None = lossless
    shape_matches: bool
    exact: bool
    sync_exact: bool
    max_abs_error: int
    # RMS of (decoded - original) over RMS of the mean-removed original, neural channels only
    relative_rms_error: float

    @property
    def lossless(self) -> bool:
        return self.bps is None

    @property
    def passed(self) -> bool:
        if not self.shape_matches:
            return False
        return self.exact if self.lossless else self.sync_exact


@dataclass(frozen=True)
class _VerifyPartial:
    max_abs_error: int
    err_sq: float
    sums: np.ndarray
    sum_sq: np.ndarray
    sync_exact: bool


def _open_compressed(output: Path) -> zarr.Group:
    try:
        root = zarr.open_group(output, mode="r")
    except Exception as exc:  # zarr raises several unrelated types for "not a group"
        raise ValueError(f"{output} is not a Zarr group: {exc}") from exc
    if "traces_seg0" not in root or "compression" not in root.attrs:
        raise ValueError(f"{output} is not a recording written by `dc ephys compress`")
    return root


def _verify_chunks(start_chunk: int, stop_chunk: int, *, bin_path: Path, output: Path, chunk: int) -> _VerifyPartial:
    meta = parse_meta(find_meta(bin_path))
    data = open_bin(bin_path, meta)
    root = _open_compressed(output)
    traces = root["traces_seg0"]
    sync = root["sync_seg0"] if "sync_seg0" in root else None
    n_neural = traces.shape[1]

    sync_exact = True
    max_err = 0
    err_sq = 0.0
    sums = np.zeros(n_neural)
    sum_sq = np.zeros(n_neural)
    for index in range(start_chunk, stop_chunk):
        start, stop = chunk_bounds(index, chunk, data.shape[0])
        original = np.asarray(data[start:stop])
        neural = original[:, :n_neural].astype(np.float64)
        diff = traces[start:stop].astype(np.float64) - neural
        if diff.size:
            max_err = max(max_err, int(np.abs(diff).max()))
        err_sq += float(np.square(diff).sum())
        sums += neural.sum(axis=0)
        sum_sq += np.square(neural).sum(axis=0)
        if sync is not None and not np.array_equal(sync[start:stop], original[:, n_neural:]):
            sync_exact = False
    return _VerifyPartial(max_err, err_sq, sums, sum_sq, sync_exact)


def verify(
        bin_path: Path, output: Path, *, show_progress: bool = False, jobs: int | None = None,
        on_progress: Callable[[int, int], None] | None = None
) -> VerifyResult:
    """Decode `output` chunk by chunk and compare it with the original SpikeGLX `.bin`."""
    jobs = resolve_jobs(jobs)
    meta = parse_meta(find_meta(bin_path))
    data = open_bin(bin_path, meta)
    root = _open_compressed(output)
    traces = root["traces_seg0"]
    sync = root["sync_seg0"] if "sync_seg0" in root else None
    n_neural = traces.shape[1]
    bps = root.attrs["compression"]["bps"]
    chunk = traces.chunks[0]

    n_sync_out = sync.shape[1] if sync is not None else 0
    if traces.shape[0] != data.shape[0] or n_neural + n_sync_out != data.shape[1]:
        return VerifyResult(bps, False, False, False, -1, float("nan"))

    partials = map_chunk_ranges(
        _verify_chunks,
        n_chunks(data.shape[0], chunk),
        jobs=jobs,
        desc="verifying",
        show_progress=show_progress,
            on_progress=on_progress,
        bin_path=bin_path,
        output=output,
        chunk=chunk,
    )

    sync_exact = True
    max_err = 0
    err_sq = 0.0
    sums = np.zeros(n_neural)
    sum_sq = np.zeros(n_neural)
    for part in partials:
        max_err = max(max_err, part.max_abs_error)
        err_sq += part.err_sq
        sums += part.sums
        sum_sq += part.sum_sq
        sync_exact = sync_exact and part.sync_exact

    signal_sq = float((sum_sq - np.square(sums) / max(data.shape[0], 1)).sum())
    relative = float(np.sqrt(err_sq / signal_sq)) if signal_sq > 0 else 0.0
    return VerifyResult(
        bps=bps,
        shape_matches=True,
        exact=max_err == 0 and sync_exact,
        sync_exact=sync_exact,
        max_abs_error=max_err,
        relative_rms_error=relative,
    )


def _restore_chunks(start_chunk: int, stop_chunk: int, *, output: Path, bin_path: Path) -> None:
    root = _open_compressed(output)
    traces = root["traces_seg0"]
    sync = root["sync_seg0"] if "sync_seg0" in root else None
    chunk = traces.chunks[0]
    n_samples = traces.shape[0]
    row_bytes = (traces.shape[1] + (sync.shape[1] if sync is not None else 0)) * SAMPLE_DTYPE.itemsize
    with bin_path.open("r+b") as f:
        for index in range(start_chunk, stop_chunk):
            start, stop = chunk_bounds(index, chunk, n_samples)
            block = traces[start:stop]
            if sync is not None:
                block = np.hstack([block, sync[start:stop]])
            f.seek(start * row_bytes)
            f.write(np.ascontiguousarray(block, dtype=SAMPLE_DTYPE).tobytes())


def run_decompress(
        output: Path,
        bin_path: Path,
        *,
        overwrite: bool = False,
        show_progress: bool = False,
        jobs: int | None = None,
) -> Path:
    """Write the Zarr recording back to an interleaved int16 SpikeGLX `.bin` + `.meta`."""
    jobs = resolve_jobs(jobs)
    if bin_path.suffix.lower() != ".bin":
        raise ValueError(f"restored file {bin_path} must have a .bin extension")
    meta_path = bin_path.with_suffix(".meta")
    for path in (bin_path, meta_path):
        prepare_output(path, overwrite)

    root = _open_compressed(output)
    traces = root["traces_seg0"]
    sync = root["sync_seg0"] if "sync_seg0" in root else None
    n_samples = traces.shape[0]
    row_bytes = (traces.shape[1] + (sync.shape[1] if sync is not None else 0)) * SAMPLE_DTYPE.itemsize
    meta_text = root.attrs["spikeglx_meta_text"]

    bin_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Preallocate the full size so each worker can write its byte range in place.
        with bin_path.open("wb") as f:
            f.truncate(n_samples * row_bytes)
        map_chunk_ranges(
            _restore_chunks,
            n_chunks(n_samples, traces.chunks[0]),
            jobs=jobs,
            desc="restoring",
            show_progress=show_progress,
            output=output,
            bin_path=bin_path,
        )
        with meta_path.open("w", newline="") as f:
            f.write(meta_text)
    except BaseException:
        bin_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        raise
    return bin_path
