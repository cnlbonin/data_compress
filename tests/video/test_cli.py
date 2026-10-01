from pathlib import Path

import numpy as np
import tifffile
from typer.testing import CliRunner

from data_compress.video.cli import app

runner = CliRunner()


def _write_stack(path: Path, n_pages: int, *, shape=(64, 64), dtype=np.uint8) -> None:
    frames = np.stack([np.full(shape, i, dtype=dtype) for i in range(n_pages)])
    tifffile.imwrite(path, frames, photometric="minisblack")


def test_probe_command_reports_frame_count_and_resolution(tmp_path: Path) -> None:
    _write_stack(tmp_path / "run_0000.tif", n_pages=4)

    result = runner.invoke(app, ["probe", str(tmp_path)])

    assert result.exit_code == 0
    assert "4" in result.stdout
    assert "64x64" in result.stdout


def test_video_app_supports_h_shorthand_for_help() -> None:
    result = runner.invoke(app, ["-h"])

    assert result.exit_code == 0
    assert "probe" in result.stdout
    assert "compress" in result.stdout


def test_compress_subcommand_supports_h_shorthand_for_help() -> None:
    result = runner.invoke(app, ["compress", "-h"])

    assert result.exit_code == 0
    assert "--codec" in result.stdout


def test_probe_command_fails_on_missing_directory(tmp_path: Path) -> None:
    result = runner.invoke(app, ["probe", str(tmp_path / "does-not-exist")])

    assert result.exit_code != 0


def test_compress_command_succeeds_and_reports_pass(tmp_path: Path) -> None:
    _write_stack(tmp_path / "run_0000.tif", n_pages=5)
    output = tmp_path / "out.mp4"

    result = runner.invoke(app, ["compress", str(tmp_path), str(output), "--fps", "10"])

    assert result.exit_code == 0
    assert "PASS" in result.stdout
    assert output.exists()


def test_compress_command_rejects_16bit_lossy_without_force(tmp_path: Path) -> None:
    _write_stack(tmp_path / "run_0000.tif", n_pages=5, dtype=np.uint16)
    output = tmp_path / "out.mp4"

    result = runner.invoke(app, ["compress", str(tmp_path), str(output), "--fps", "10"])

    assert result.exit_code != 0
    assert "16-bit" in result.stdout
    assert not output.exists()
