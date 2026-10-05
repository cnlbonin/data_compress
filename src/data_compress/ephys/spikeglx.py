"""Read SpikeGLX `.bin` recordings and their `.meta` sidecars."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_DTYPE = np.dtype("<i2")  # SpikeGLX always stores interleaved little-endian int16


@dataclass(frozen=True)
class SpikeGLXMeta:
    path: Path
    n_saved_chans: int
    sample_rate: float
    # Trailing digital channels (imec SY sync word, nidq digital words). They carry
    # TTL bits rather than voltages, so they must never go through a lossy codec.
    n_sync_chans: int
    file_size_bytes: int | None
    raw: dict[str, str]
    text: str

    @property
    def n_neural_chans(self) -> int:
        return self.n_saved_chans - self.n_sync_chans


def find_meta(bin_path: Path) -> Path:
    meta_path = bin_path.with_suffix(".meta")
    if not meta_path.is_file():
        raise FileNotFoundError(f"SpikeGLX sidecar {meta_path.name} not found next to {bin_path}")
    return meta_path


def stream_type(bin_path: Path) -> str:
    name = bin_path.name.lower()
    for stream in ("ap", "lf", "nidq"):
        if name.endswith(f".{stream}.bin"):
            return stream
    return "unknown"


def parse_meta(meta_path: Path) -> SpikeGLXMeta:
    # newline="" keeps CRLF endings (SpikeGLX on Windows) so the text restores byte-for-byte
    with meta_path.open(newline="") as f:
        text = f.read()
    raw: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            raw[key.strip()] = value.strip()

    try:
        n_saved_chans = int(raw["nSavedChans"])
    except KeyError as exc:
        raise ValueError(f"{meta_path} has no nSavedChans entry; not a SpikeGLX .meta file?") from exc

    rate = raw.get("imSampRate") or raw.get("niSampRate")
    if rate is None:
        raise ValueError(f"{meta_path} has neither imSampRate nor niSampRate")

    if "snsApLfSy" in raw:  # imec: AP, LF, SY counts
        n_sync = int(raw["snsApLfSy"].split(",")[2])
    elif "snsMnMaXaDw" in raw:  # nidq: MN, MA, XA, DW counts
        n_sync = int(raw["snsMnMaXaDw"].split(",")[3])
    else:
        n_sync = 0

    file_size = raw.get("fileSizeBytes")
    return SpikeGLXMeta(
        path=meta_path,
        n_saved_chans=n_saved_chans,
        sample_rate=float(rate),
        n_sync_chans=n_sync,
        file_size_bytes=int(file_size) if file_size is not None else None,
        raw=raw,
        text=text,
    )


def open_bin(bin_path: Path, meta: SpikeGLXMeta) -> np.memmap:
    size = bin_path.stat().st_size
    if meta.file_size_bytes is not None and size != meta.file_size_bytes:
        raise ValueError(
            f"{bin_path.name} is {size} bytes but its .meta says fileSizeBytes={meta.file_size_bytes}; "
            "the recording may be truncated or still being written"
        )
    bytes_per_sample = SAMPLE_DTYPE.itemsize * meta.n_saved_chans
    if size % bytes_per_sample:
        raise ValueError(
            f"{bin_path.name} ({size} bytes) is not a whole number of samples "
            f"of {meta.n_saved_chans} int16 channels"
        )
    return np.memmap(
        bin_path, dtype=SAMPLE_DTYPE, mode="r", shape=(size // bytes_per_sample, meta.n_saved_chans)
    )
