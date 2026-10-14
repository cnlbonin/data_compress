"""Write small synthetic Scanbox recordings (.sbx + .mat) for tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import savemat

# Inverse of sbx_source._LEGACY_N_CHANNELS for the common 1- and 2-channel cases.
_CHANNELS_FOR_N = {2: 1, 1: 2}


def write_sbx(
        directory: Path,
        frames: np.ndarray,  # (n_timepoints, n_planes, height, width, n_channels)
        *,
        name: str = "run00",
        legacy_channels_field: bool = True,
) -> Path:
    """Write `frames` as `<name>.sbx` (raw uint16) + `<name>.mat` (Scanbox `info` struct)."""
    n_timepoints, n_planes, height, width, n_channels = frames.shape
    directory.mkdir(parents=True, exist_ok=True)
    sbx_path = directory / f"{name}.sbx"
    np.ascontiguousarray(frames, dtype="<u2").tofile(sbx_path)

    info: dict[str, object] = {"sz": np.array([height, width], dtype=np.float64)}
    if legacy_channels_field:
        info["channels"] = float(_CHANNELS_FOR_N[n_channels])
    else:
        info["chan"] = {"nchan": float(n_channels)}
    if n_planes > 1:
        info["otwave"] = np.arange(n_planes, dtype=np.float64)

    savemat(directory / f"{name}.mat", {"info": info})
    return sbx_path


def make_frames(
        n_timepoints: int, *, n_planes: int = 1, height: int = 4, width: int = 5, n_channels: int = 2, seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 65536, size=(n_timepoints, n_planes, height, width, n_channels), dtype=np.uint16)
