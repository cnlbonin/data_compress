"""Write small synthetic SpikeGLX recordings (.bin + .meta) for tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def make_traces(
        n_samples: int, n_neural: int, n_sync: int = 1, *, seed: int = 0
) -> np.ndarray:
    """Noise + slow drift + a few spikes on the neural channels, TTL bits on the sync channel(s)."""
    rng = np.random.default_rng(seed)
    neural = rng.normal(0, 20, (n_samples, n_neural))
    neural += np.cumsum(rng.normal(0, 1, (n_samples, n_neural)), axis=0) * 0.05
    if n_samples > 30:
        for t in rng.integers(0, n_samples - 30, size=max(1, n_samples // 2000)):
            neural[t:t + 30] -= 150 * np.hanning(30)[:, None]
    sync = np.zeros((n_samples, n_sync))
    for k in range(n_sync):
        period = 997 + 100 * k
        sync[:, k] = np.where((np.arange(n_samples) // period) % 2 == 1, 1 << (6 + k), 0)
    return np.hstack([neural, sync]).astype(np.int16)


def write_spikeglx(
        directory: Path,
        traces: np.ndarray,
        *,
        stream: str = "ap",
        n_sync: int = 1,
        sample_rate: float = 30000.0,
        file_size_bytes: int | None = None,
) -> Path:
    """Write `traces` (n_samples, n_chans) int16 as `<run>_g0_t0.imec0.<stream>.bin` + `.meta`."""
    n_chans = traces.shape[1]
    n_neural = n_chans - n_sync
    if stream == "nidq":
        bin_path = directory / "run_g0_t0.nidq.bin"
        stream_lines = [
            "typeThis=nidq",
            f"niSampRate={sample_rate}",
            f"snsMnMaXaDw=0,0,{n_neural},{n_sync}",
        ]
    else:
        bin_path = directory / f"run_g0_t0.imec0.{stream}.bin"
        counts = f"{n_neural},0,{n_sync}" if stream == "ap" else f"0,{n_neural},{n_sync}"
        stream_lines = [
            "typeThis=imec",
            f"imSampRate={sample_rate}",
            f"snsApLfSy={counts}",
        ]

    traces.astype("<i2").tofile(bin_path)
    size = bin_path.stat().st_size if file_size_bytes is None else file_size_bytes
    lines = [
        "appVersion=20230425",
        f"fileSizeBytes={size}",
        f"nSavedChans={n_chans}",
        *stream_lines,
        f"~snsChanMap=({n_neural},0,{n_sync})",
    ]
    bin_path.with_suffix(".meta").write_text("\n".join(lines) + "\n")
    return bin_path
