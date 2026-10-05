"""`dc ephys ...` commands: probe, compress and restore SpikeGLX recordings."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from data_compress.ephys.encode import DEFAULT_LEVEL, run_compress, run_decompress, verify
from data_compress.ephys.probe import ProbeReport, build_probe_report

app = typer.Typer(
    help="Compress SpikeGLX ephys recordings with WavPack (lossless or lossy).",
    context_settings={"help_option_names": ["-h", "--help"]},
)
console = Console()

BinArg = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, help="SpikeGLX .bin file")


def _mb(n_bytes: int) -> str:
    return f"{n_bytes / 1e6:.1f} MB"


def _print_probe_report(report: ProbeReport) -> None:
    table = Table(title=f"SpikeGLX probe: {report.bin_path.name}")
    table.add_column("field")
    table.add_column("value")
    table.add_row("stream", report.stream)
    table.add_row("neural channels", str(report.n_neural_chans))
    table.add_row("sync channels", str(report.n_sync_chans))
    table.add_row("sample rate (Hz)", f"{report.sample_rate:g}")
    table.add_row("samples", str(report.n_samples))
    table.add_row("duration", f"{report.duration_s:.1f} s")
    table.add_row("size", _mb(report.size_bytes))
    if report.size_mismatch:
        table.add_row(
            "[bold red]WARNING[/]",
            "file size does not match .meta fileSizeBytes (truncated or still recording?)",
        )
    console.print(table)


@app.command()
def probe(bin_path: Path = BinArg) -> None:
    """Inspect a SpikeGLX .bin and its .meta sidecar. Read-only."""
    try:
        report = build_probe_report(bin_path)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc
    _print_probe_report(report)


@app.command()
def compress(
    bin_path: Path = BinArg,
    output: Path = typer.Argument(..., help="Output .zarr directory"),
    bps: float | None = typer.Option(
        None,
        help="Lossy target bits per sample (2.25-16); omit for lossless. "
        "The paper found no spike-sorting degradation at 3, 2.5 or 2.25.",
    ),
    level: int = typer.Option(DEFAULT_LEVEL, help="WavPack effort level 1-4 (higher: smaller, slower)"),
    overwrite: bool = typer.Option(False, help="Replace an existing output"),
    no_verify: bool = typer.Option(False, "--no-verify", help="Skip re-reading and checking the output"),
) -> None:
    """Compress a SpikeGLX .bin into a WavPack-compressed Zarr directory."""
    try:
        report = build_probe_report(bin_path)
        _print_probe_report(report)
        if bps is not None and report.stream != "ap":
            console.print(
                f"[bold yellow]WARNING[/] lossy compression on a {report.stream!r} stream; "
                "the paper only validated lossy WavPack on AP / wide-band data"
            )
        result = run_compress(bin_path, output, bps=bps, level=level, overwrite=overwrite, show_progress=True)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc

    mode = "lossless" if bps is None else f"lossy, bps={bps:g}"
    console.print(
        f"wrote {output} ({mode}): {_mb(result.original_bytes)} -> {_mb(result.compressed_bytes)} "
        f"({result.ratio:.2f}x)"
    )
    if not no_verify:
        _verify_and_report(bin_path, output)


def _verify_and_report(bin_path: Path, output: Path) -> None:
    check = verify(bin_path, output, show_progress=True)

    table = Table(title=f"verify: {output.name} vs {bin_path.name}")
    table.add_column("check")
    table.add_column("value")
    table.add_row("mode", "lossless" if check.lossless else f"lossy, bps={check.bps:g}")
    table.add_row("shape matches source", "yes" if check.shape_matches else "[bold red]no[/]")
    if check.shape_matches:
        table.add_row("neural + sync identical", "yes" if check.exact else "no")
        table.add_row("sync identical", "yes" if check.sync_exact else "[bold red]no[/]")
        table.add_row("max |diff|", str(check.max_abs_error))
        table.add_row("relative RMS error", f"{check.relative_rms_error:.3f}")
    console.print(table)

    if check.passed:
        detail = "decoded data identical to source" if check.lossless else "sync channel(s) identical to source"
        console.print(f"[bold green]PASS[/] {detail}")
    else:
        console.print(f"[bold red]FAIL[/] decoded {output} does not match {bin_path}")
        raise typer.Exit(code=1)


@app.command(name="verify")
def verify_command(
    bin_path: Path = BinArg,
    zarr_path: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, help="Compressed .zarr"),
) -> None:
    """Re-check a compressed .zarr against its original .bin (e.g. after copying it)."""
    try:
        _verify_and_report(bin_path, zarr_path)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def decompress(
    zarr_path: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, help="Compressed .zarr"),
    output: Path = typer.Argument(..., help="Restored .bin path (.meta is written next to it)"),
    overwrite: bool = typer.Option(False, help="Replace existing .bin/.meta"),
) -> None:
    """Restore a compressed recording to a SpikeGLX .bin + .meta (e.g. for Kilosort)."""
    try:
        run_decompress(zarr_path, output, overwrite=overwrite, show_progress=True)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[bold green]Done[/] wrote {output} and {output.with_suffix('.meta').name}")
