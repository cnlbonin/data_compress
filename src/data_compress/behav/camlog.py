"""Parser for labcams .camlog sidecar files (frame_id,timestamp rows)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CamlogInfo:
    frame_count: int
    fps: float


def parse_camlog(path: Path) -> CamlogInfo:
    timestamps: list[float] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        _frame_id, timestamp = line.split(",")
        timestamps.append(float(timestamp))

    frame_count = len(timestamps)
    if frame_count < 2:
        raise ValueError(
            f"camlog {path} has {frame_count} frame(s); need at least 2 frames to derive fps"
        )

    duration = timestamps[-1] - timestamps[0]
    if duration <= 0:
        raise ValueError(
            f"camlog {path} timestamps are non-increasing "
            f"(first={timestamps[0]}, last={timestamps[-1]}); cannot derive fps"
        )
    fps = (frame_count - 1) / duration
    return CamlogInfo(frame_count=frame_count, fps=fps)


def find_camlog(tif_dir: Path) -> Path | None:
    matches = sorted(tif_dir.glob("*.camlog"))
    if not matches:
        return None
    if len(matches) > 1:
        names = ", ".join(m.name for m in matches)
        raise ValueError(f"multiple .camlog files found in {tif_dir}: {names}")
    return matches[0]
