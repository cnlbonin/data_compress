"""Compress SpikeGLX recordings to Zarr with WavPack, verify, and restore them.

Follows Buccino et al. (eLife 2025, doi:10.7554/eLife.110170): WavPack via
wavpack-numcodecs, 1-s chunks over all channels, lossless by default and lossy
only at an explicit bits-per-sample (bps) target. The output layout matches
SpikeInterface's Zarr recordings (`traces_seg0`, `channel_ids`,
`sampling_frequency`) so it can be loaded later with `spikeinterface.read_zarr`.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import zarr
from tqdm import tqdm
from wavpack_numcodecs import WavPack

from data_compress.ephys.spikeglx import SAMPLE_DTYPE, find_meta, open_bin, parse_meta

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


def _dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _iter_chunks(n_samples: int, chunk: int, *, desc: str, show_progress: bool):
    starts = range(0, n_samples, chunk)
    for start in tqdm(starts, unit="chunk", desc=desc, disable=not show_progress):
        yield start, min(start + chunk, n_samples)


def _prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists; pass overwrite to replace it")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def run_compress(
        bin_path: Path,
        output: Path,
        *,
        bps: float | None = None,
        level: int = DEFAULT_LEVEL,
        overwrite: bool = False,
        show_progress: bool = False,
) -> CompressResult:
    validate_bps(bps)
    if output.suffix.lower() != ".zarr":
        raise ValueError(f"output {output} must have a .zarr extension")

    meta = parse_meta(find_meta(bin_path))
    data = open_bin(bin_path, meta)
    n_samples = data.shape[0]
    n_neural = meta.n_neural_chans
    chunk = max(1, int(round(meta.sample_rate * CHUNK_SECONDS)))

    _prepare_output(output, overwrite)
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
        traces = root.create_array(
            "traces_seg0",
            shape=(n_samples, n_neural),
            chunks=(chunk, n_neural),
            dtype=SAMPLE_DTYPE,
            compressors=WavPack(level=level, bps=bps),
        )
        sync = None
        if meta.n_sync_chans:
            sync = root.create_array(
                "sync_seg0",
                shape=(n_samples, meta.n_sync_chans),
                chunks=(chunk, meta.n_sync_chans),
                dtype=SAMPLE_DTYPE,
                compressors=WavPack(level=level),  # TTL bits: always lossless
            )

        for start, stop in _iter_chunks(n_samples, chunk, desc="compressing", show_progress=show_progress):
            block = np.asarray(data[start:stop])
            traces[start:stop] = block[:, :n_neural]
            if sync is not None:
                sync[start:stop] = block[:, n_neural:]
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise

    return CompressResult(
        output=output,
        n_samples=n_samples,
        original_bytes=bin_path.stat().st_size,
        compressed_bytes=_dir_size(output),
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


def _open_compressed(output: Path) -> zarr.Group:
    try:
        root = zarr.open_group(output, mode="r")
    except Exception as exc:  # zarr raises several unrelated types for "not a group"
        raise ValueError(f"{output} is not a Zarr group: {exc}") from exc
    if "traces_seg0" not in root or "compression" not in root.attrs:
        raise ValueError(f"{output} is not a recording written by `dc ephys compress`")
    return root


def verify(bin_path: Path, output: Path, *, show_progress: bool = False) -> VerifyResult:
    """Decode `output` chunk by chunk and compare it with the original SpikeGLX `.bin`."""
    meta = parse_meta(find_meta(bin_path))
    data = open_bin(bin_path, meta)
    root = _open_compressed(output)
    traces = root["traces_seg0"]
    sync = root["sync_seg0"] if "sync_seg0" in root else None
    n_neural = traces.shape[1]
    bps = root.attrs["compression"]["bps"]

    n_sync_out = sync.shape[1] if sync is not None else 0
    if traces.shape[0] != data.shape[0] or n_neural + n_sync_out != data.shape[1]:
        return VerifyResult(bps, False, False, False, -1, float("nan"))

    sync_exact = True
    max_err = 0
    err_sq = 0.0
    sums = np.zeros(n_neural)
    sum_sq = np.zeros(n_neural)
    for start, stop in _iter_chunks(data.shape[0], traces.chunks[0], desc="verifying", show_progress=show_progress):
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


def run_decompress(
        output: Path, bin_path: Path, *, overwrite: bool = False, show_progress: bool = False
) -> Path:
    """Write the Zarr recording back to an interleaved int16 SpikeGLX `.bin` + `.meta`."""
    if bin_path.suffix.lower() != ".bin":
        raise ValueError(f"restored file {bin_path} must have a .bin extension")
    meta_path = bin_path.with_suffix(".meta")
    for path in (bin_path, meta_path):
        _prepare_output(path, overwrite)

    root = _open_compressed(output)
    traces = root["traces_seg0"]
    sync = root["sync_seg0"] if "sync_seg0" in root else None

    bin_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with bin_path.open("wb") as f:
            for start, stop in _iter_chunks(traces.shape[0], traces.chunks[0], desc="restoring", show_progress=show_progress):
                block = traces[start:stop]
                if sync is not None:
                    block = np.hstack([block, sync[start:stop]])
                f.write(np.ascontiguousarray(block, dtype=SAMPLE_DTYPE).tobytes())
        with meta_path.open("w", newline="") as f:
            f.write(root.attrs["spikeglx_meta_text"])
    except BaseException:
        bin_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        raise
    return bin_path
