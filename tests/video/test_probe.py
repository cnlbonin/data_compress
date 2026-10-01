from pathlib import Path

import numpy as np
import tifffile

from data_compress.video.probe import build_probe_report


def _write_stack(path: Path, n_pages: int, *, shape=(4, 5), dtype=np.uint8) -> None:
    frames = np.stack([np.full(shape, i, dtype=dtype) for i in range(n_pages)])
    tifffile.imwrite(path, frames, photometric="minisblack")


def _write_camlog(path: Path, n_frames: int, fps: float) -> None:
    lines = ["# Log header:frame_id,timestamp"]
    lines += [f"{i},{i / fps}" for i in range(1, n_frames + 1)]
    path.write_text("\n".join(lines) + "\n")


def test_build_probe_report_without_camlog(tmp_path: Path) -> None:
    _write_stack(tmp_path / "run_0000.tif", n_pages=4)

    report = build_probe_report(tmp_path)

    assert report.frame_count == 4
    assert report.camlog_path is None
    assert report.camlog_frame_count is None
    assert report.camlog_fps is None
    assert report.frame_count_mismatch is False


def test_build_probe_report_with_matching_camlog(tmp_path: Path) -> None:
    _write_stack(tmp_path / "run_0000.tif", n_pages=4)
    _write_camlog(tmp_path / "run.camlog", n_frames=4, fps=30.0)

    report = build_probe_report(tmp_path)

    assert report.camlog_frame_count == 4
    assert report.camlog_fps == 30.0
    assert report.frame_count_mismatch is False


def test_build_probe_report_with_mismatched_camlog(tmp_path: Path) -> None:
    _write_stack(tmp_path / "run_0000.tif", n_pages=4)
    _write_camlog(tmp_path / "run.camlog", n_frames=6, fps=30.0)

    report = build_probe_report(tmp_path)

    assert report.frame_count_mismatch is True
