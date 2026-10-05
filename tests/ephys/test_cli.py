from pathlib import Path

from typer.testing import CliRunner

from data_compress.ephys.cli import app
from tests.ephys.spikeglx_fixtures import make_traces, write_spikeglx

runner = CliRunner()


def _bin(tmp_path: Path, **kwargs) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    return write_spikeglx(src, make_traces(3000, n_neural=8), sample_rate=1000.0, **kwargs)


def test_ephys_app_supports_h_shorthand_for_help() -> None:
    result = runner.invoke(app, ["-h"])

    assert result.exit_code == 0
    for command in ("probe", "compress", "verify", "decompress"):
        assert command in result.stdout


def test_compress_help_lists_bps() -> None:
    result = runner.invoke(app, ["compress", "-h"])

    assert result.exit_code == 0
    assert "--bps" in result.stdout


def test_probe_command_reports_channels_and_rate(tmp_path: Path) -> None:
    result = runner.invoke(app, ["probe", str(_bin(tmp_path))])

    assert result.exit_code == 0
    assert "ap" in result.stdout
    assert "1000" in result.stdout
    assert "8" in result.stdout


def test_compress_lossless_passes_verification(tmp_path: Path) -> None:
    out = tmp_path / "rec.zarr"

    result = runner.invoke(app, ["compress", str(_bin(tmp_path)), str(out)])

    assert result.exit_code == 0, result.stdout
    assert "PASS" in result.stdout
    assert out.is_dir()


def test_compress_lossy_passes_and_reports_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["compress", str(_bin(tmp_path)), str(tmp_path / "rec.zarr"), "--bps", "3"])

    assert result.exit_code == 0, result.stdout
    assert "PASS" in result.stdout
    assert "RMS" in result.stdout


def test_compress_lossy_on_lf_stream_warns(tmp_path: Path) -> None:
    bin_path = _bin(tmp_path, stream="lf")

    result = runner.invoke(app, ["compress", str(bin_path), str(tmp_path / "rec.zarr"), "--bps", "3"])

    assert result.exit_code == 0, result.stdout
    assert "WARNING" in result.stdout


def test_compress_rejects_bad_bps(tmp_path: Path) -> None:
    result = runner.invoke(app, ["compress", str(_bin(tmp_path)), str(tmp_path / "rec.zarr"), "--bps", "1"])

    assert result.exit_code == 1
    assert "bps" in result.stdout


def test_compress_existing_output_errors_without_overwrite(tmp_path: Path) -> None:
    out = tmp_path / "rec.zarr"
    out.mkdir()

    result = runner.invoke(app, ["compress", str(_bin(tmp_path)), str(out)])

    assert result.exit_code == 1
    assert "already exists" in result.stdout


def test_decompress_restores_bin(tmp_path: Path) -> None:
    bin_path = _bin(tmp_path)
    out = tmp_path / "rec.zarr"
    runner.invoke(app, ["compress", str(bin_path), str(out), "--no-verify"])
    restored = tmp_path / "restored" / bin_path.name

    result = runner.invoke(app, ["decompress", str(out), str(restored)])

    assert result.exit_code == 0, result.stdout
    assert restored.read_bytes() == bin_path.read_bytes()


def test_verify_command_passes_on_lossless_output(tmp_path: Path) -> None:
    bin_path = _bin(tmp_path)
    out = tmp_path / "rec.zarr"
    runner.invoke(app, ["compress", str(bin_path), str(out), "--no-verify"])

    result = runner.invoke(app, ["verify", str(bin_path), str(out)])

    assert result.exit_code == 0, result.stdout
    assert "lossless" in result.stdout
    assert "PASS" in result.stdout


def test_verify_command_reports_lossy_error(tmp_path: Path) -> None:
    bin_path = _bin(tmp_path)
    out = tmp_path / "rec.zarr"
    runner.invoke(app, ["compress", str(bin_path), str(out), "--bps", "3", "--no-verify"])

    result = runner.invoke(app, ["verify", str(bin_path), str(out)])

    assert result.exit_code == 0, result.stdout
    assert "bps=3" in result.stdout
    assert "RMS" in result.stdout
    assert "PASS" in result.stdout


def test_verify_command_fails_on_corrupted_output(tmp_path: Path) -> None:
    import zarr

    bin_path = _bin(tmp_path)
    out = tmp_path / "rec.zarr"
    runner.invoke(app, ["compress", str(bin_path), str(out), "--no-verify"])
    traces = zarr.open_group(out, mode="r+")["traces_seg0"]
    traces[5, 2] = traces[5, 2] + 1

    result = runner.invoke(app, ["verify", str(bin_path), str(out)])

    assert result.exit_code == 1
    assert "FAIL" in result.stdout


def test_verify_command_fails_on_different_recording(tmp_path: Path) -> None:
    bin_path = _bin(tmp_path)
    out = tmp_path / "rec.zarr"
    runner.invoke(app, ["compress", str(bin_path), str(out), "--no-verify"])
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other = write_spikeglx(other_dir, make_traces(2000, n_neural=8, seed=1), sample_rate=1000.0)

    result = runner.invoke(app, ["verify", str(other), str(out)])

    assert result.exit_code == 1
    assert "FAIL" in result.stdout


def test_verify_command_errors_on_non_compressed_dir(tmp_path: Path) -> None:
    bin_path = _bin(tmp_path)
    not_zarr = tmp_path / "empty.zarr"
    not_zarr.mkdir()

    result = runner.invoke(app, ["verify", str(bin_path), str(not_zarr)])

    assert result.exit_code == 1
    assert "Error" in result.stdout
