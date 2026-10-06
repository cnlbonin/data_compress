from pathlib import Path

import numpy as np
import tifffile
from typer.testing import CliRunner

from data_compress.neuimg.cli import app

runner = CliRunner()


def _write_stack(tif_dir: Path, n_pages: int, *, dtype=np.uint16) -> None:
    tif_dir.mkdir(parents=True, exist_ok=True)
    frames = np.stack([np.full((16, 16), i, dtype=dtype) for i in range(n_pages)])
    tifffile.imwrite(tif_dir / "run_0000.tif", frames, photometric="minisblack")


def test_probe_reports_frames_and_resolution(tmp_path: Path) -> None:
    _write_stack(tmp_path / "src", n_pages=4)

    result = runner.invoke(app, ["probe", str(tmp_path / "src")])

    assert result.exit_code == 0
    assert "16x16" in result.stdout
    assert "4" in result.stdout


def test_compress_then_verify_reports_pass(tmp_path: Path) -> None:
    _write_stack(tmp_path / "src", n_pages=5)
    output = tmp_path / "out.zarr"

    result = runner.invoke(app, ["compress", str(tmp_path / "src"), str(output), "--jobs", "1"])

    assert result.exit_code == 0
    assert "PASS" in result.stdout
    assert output.exists()


def test_verify_command_fails_on_mismatch(tmp_path: Path) -> None:
    _write_stack(tmp_path / "src", n_pages=3)
    output = tmp_path / "out.zarr"
    assert runner.invoke(app, ["compress", str(tmp_path / "src"), str(output), "--jobs", "1"]).exit_code == 0

    _write_stack(tmp_path / "src", n_pages=3, dtype=np.uint16)  # rewrite source with same values
    tifffile.imwrite(
        tmp_path / "src" / "run_0000.tif", np.full((3, 16, 16), 9, dtype=np.uint16), photometric="minisblack"
    )
    result = runner.invoke(app, ["verify", str(tmp_path / "src"), str(output), "--jobs", "1"])

    assert result.exit_code == 1
    assert "FAIL" in result.stdout


def test_decompress_command_writes_tiffs(tmp_path: Path) -> None:
    _write_stack(tmp_path / "src", n_pages=3)
    output = tmp_path / "out.zarr"
    runner.invoke(app, ["compress", str(tmp_path / "src"), str(output), "--no-verify", "--jobs", "1"])

    result = runner.invoke(app, ["decompress", str(output), str(tmp_path / "restored"), "--jobs", "1"])

    assert result.exit_code == 0
    assert (tmp_path / "restored" / "run_0000.tif").exists()


def test_compress_command_refuses_overwrite_without_flag(tmp_path: Path) -> None:
    _write_stack(tmp_path / "src", n_pages=2)
    output = tmp_path / "out.zarr"
    runner.invoke(app, ["compress", str(tmp_path / "src"), str(output), "--no-verify", "--jobs", "1"])

    result = runner.invoke(app, ["compress", str(tmp_path / "src"), str(output), "--jobs", "1"])

    assert result.exit_code == 1
    assert "overwrite" in result.stdout
