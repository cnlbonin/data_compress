from typer.testing import CliRunner

from data_compress.cli import app

runner = CliRunner()


def test_root_app_registers_video_subcommand() -> None:
    result = runner.invoke(app, ["video", "--help"])

    assert result.exit_code == 0
    assert "probe" in result.stdout
    assert "compress" in result.stdout


def test_root_app_supports_h_shorthand_for_help() -> None:
    result = runner.invoke(app, ["-h"])

    assert result.exit_code == 0
    assert "video" in result.stdout
