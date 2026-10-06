from pathlib import Path

import pytest

from data_compress.behav.camlog import find_camlog, parse_camlog


def _write_camlog(path: Path, rows: list[tuple[int, float]]) -> None:
    lines = [
        "# Camera: facecam log file",
        "# Date: 13-09-2025",
        "# labcams version: 0.2",
        "# Log header:frame_id,timestamp",
    ]
    lines += [f"{frame_id},{timestamp}" for frame_id, timestamp in rows]
    path.write_text("\n".join(lines) + "\n")


def test_parse_camlog_computes_fps_and_frame_count(tmp_path: Path) -> None:
    camlog_path = tmp_path / "run.camlog"
    # 31 frames spanning exactly 1.0 second -> 30 intervals -> 30 fps
    rows = [(i, i / 30.0) for i in range(1, 32)]
    _write_camlog(camlog_path, rows)

    info = parse_camlog(camlog_path)

    assert info.frame_count == 31
    assert info.fps == pytest.approx(30.0, rel=1e-3)


def test_parse_camlog_raises_on_non_increasing_timestamps(tmp_path: Path) -> None:
    camlog_path = tmp_path / "run.camlog"
    # All frames share the same timestamp -> zero duration, fps undefined.
    _write_camlog(camlog_path, [(1, 0.0), (2, 0.0), (3, 0.0)])

    with pytest.raises(ValueError, match="non-increasing"):
        parse_camlog(camlog_path)


def test_parse_camlog_raises_on_single_frame(tmp_path: Path) -> None:
    camlog_path = tmp_path / "run.camlog"
    _write_camlog(camlog_path, [(1, 0.0)])

    with pytest.raises(ValueError, match="at least 2 frames"):
        parse_camlog(camlog_path)


def test_find_camlog_returns_path_when_present(tmp_path: Path) -> None:
    camlog_path = tmp_path / "run.camlog"
    _write_camlog(camlog_path, [(1, 0.0), (2, 1 / 30.0)])
    (tmp_path / "frame_000.tif").touch()

    assert find_camlog(tmp_path) == camlog_path


def test_find_camlog_returns_none_when_absent(tmp_path: Path) -> None:
    (tmp_path / "frame_000.tif").touch()

    assert find_camlog(tmp_path) is None


def test_find_camlog_raises_when_multiple_present(tmp_path: Path) -> None:
    _write_camlog(tmp_path / "a.camlog", [(1, 0.0), (2, 1 / 30.0)])
    _write_camlog(tmp_path / "b.camlog", [(1, 0.0), (2, 1 / 30.0)])

    with pytest.raises(ValueError, match="multiple"):
        find_camlog(tmp_path)
