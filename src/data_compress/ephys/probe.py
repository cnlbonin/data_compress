"""Read-only summary of a SpikeGLX recording."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from data_compress.ephys.spikeglx import SAMPLE_DTYPE, find_meta, parse_meta, stream_type


@dataclass(frozen=True)
class ProbeReport:
    bin_path: Path
    stream: str
    n_neural_chans: int
    n_sync_chans: int
    sample_rate: float
    n_samples: int
    duration_s: float
    size_bytes: int
    size_mismatch: bool  # file size disagrees with .meta fileSizeBytes or isn't whole samples


def build_probe_report(bin_path: Path) -> ProbeReport:
    meta = parse_meta(find_meta(bin_path))
    size = bin_path.stat().st_size
    bytes_per_sample = SAMPLE_DTYPE.itemsize * meta.n_saved_chans
    n_samples = size // bytes_per_sample
    mismatch = size % bytes_per_sample != 0 or (
        meta.file_size_bytes is not None and size != meta.file_size_bytes
    )
    return ProbeReport(
        bin_path=bin_path,
        stream=stream_type(bin_path),
        n_neural_chans=meta.n_neural_chans,
        n_sync_chans=meta.n_sync_chans,
        sample_rate=meta.sample_rate,
        n_samples=n_samples,
        duration_s=n_samples / meta.sample_rate,
        size_bytes=size,
        size_mismatch=mismatch,
    )
