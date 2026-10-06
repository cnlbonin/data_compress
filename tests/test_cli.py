from typer.testing import CliRunner

from data_compress.cli import app

runner = CliRunner()


def test_root_app_registers_behav_subcommand() -> None:
    result = runner.invoke(app, ["behav", "--help"])

    assert result.exit_code == 0
    assert "probe" in result.stdout
    assert "compress" in result.stdout


def test_root_app_supports_h_shorthand_for_help() -> None:
    result = runner.invoke(app, ["-h"])

    assert result.exit_code == 0
    assert "behav" in result.stdout


def test_root_app_registers_ephys_subcommand() -> None:
    result = runner.invoke(app, ["ephys", "--help"])

    assert result.exit_code == 0
    assert "compress" in result.stdout
    assert "decompress" in result.stdout
