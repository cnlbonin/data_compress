"""Reading Scanbox `.sbx` recordings and their `.mat` sidecars.

A `.sbx` file is a flat binary of uint16 samples (channel-interleaved), with a
companion `<name>.mat` file holding a MATLAB struct named `info` with the
acquisition geometry. Frame count is not stored in the sidecar; it is derived
from the `.sbx` file size. Scanbox's own tools invert pixel values for display
(``65535 - raw``); this reader never applies that, since it feeds a
byte-exact archiver, not a viewer.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

SBX_SUFFIX = ".sbx"
DTYPE = np.dtype("<u2")  # Scanbox always stores interleaved little-endian uint16

# Legacy `info.channels` encoding (older Scanbox versions; newer ones use info.chan.nchan).
_LEGACY_N_CHANNELS = {1: 2, 2: 1, 3: 1}


def find_mat(sbx_path: Path) -> Path:
    mat_path = sbx_path.with_suffix(".mat")
    if not mat_path.is_file():
        raise FileNotFoundError(f"Scanbox sidecar {mat_path.name} not found next to {sbx_path.name}")
    return mat_path


def _load_info(mat_path: Path):
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if "info" not in mat:
        raise ValueError(f"{mat_path} has no 'info' struct; not a Scanbox .mat file?")
    return mat["info"]


def _n_channels(info, mat_path: Path) -> int:
    chan = getattr(info, "chan", None)
    nchan = getattr(chan, "nchan", None) if chan is not None else None
    if nchan is not None:
        return int(nchan)
    channels = getattr(info, "channels", None)
    if channels is not None:
        try:
            return _LEGACY_N_CHANNELS[int(channels)]
        except KeyError as exc:
            raise ValueError(f"{mat_path}: unrecognized info.channels value {channels!r}") from exc
    raise ValueError(f"{mat_path} has neither info.chan.nchan nor info.channels; not a Scanbox .mat file?")


def _n_planes(info) -> int:
    otwave = getattr(info, "otwave", None)
    if otwave is None:
        return 1
    return max(1, int(np.atleast_1d(np.asarray(otwave)).size))


@dataclass(frozen=True)
class SbxInfo:
    sbx_path: Path
    mat_path: Path
    n_timepoints: int
    n_planes: int
    n_channels: int
    height: int
    width: int
    dtype: np.dtype
    is_volumetric: bool
    mat_bytes: bytes

    @property
    def frame_count(self) -> int:
        return self.n_timepoints

    @property
    def frame_shape(self) -> tuple[int, int, int, int]:
        return (self.n_planes, self.height, self.width, self.n_channels)


def scan_sbx_file(sbx_path: Path) -> SbxInfo:
    mat_path = find_mat(sbx_path)
    info = _load_info(mat_path)

    try:
        height, width = (int(x) for x in np.atleast_1d(info.sz))
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{mat_path} has no valid info.sz field; not a Scanbox .mat file?") from exc

    n_channels = _n_channels(info, mat_path)
    n_planes = _n_planes(info)

    frame_bytes = n_channels * n_planes * height * width * DTYPE.itemsize
    size = sbx_path.stat().st_size
    if frame_bytes == 0 or size % frame_bytes:
        raise ValueError(
            f"{sbx_path.name} ({size} bytes) is not a whole number of timepoints of "
            f"{n_channels} channel(s) x {n_planes} plane(s) x {height}x{width} uint16"
        )

    return SbxInfo(
        sbx_path=sbx_path,
        mat_path=mat_path,
        n_timepoints=size // frame_bytes,
        n_planes=n_planes,
        n_channels=n_channels,
        height=height,
        width=width,
        dtype=DTYPE,
        is_volumetric=n_planes > 1,
        mat_bytes=mat_path.read_bytes(),
    )


def _memmap(info: SbxInfo) -> np.memmap:
    return np.memmap(
        info.sbx_path,
        dtype=info.dtype,
        mode="r",
        shape=(info.n_timepoints, info.n_planes, info.height, info.width, info.n_channels),
    )


def iter_frame_range(info: SbxInfo, start: int, stop: int) -> Iterator[np.ndarray]:
    """Yield global timepoints [start, stop) as (n_planes, height, width, n_channels) arrays."""
    data = _memmap(info)
    for index in range(start, stop):
        yield np.asarray(data[index])
