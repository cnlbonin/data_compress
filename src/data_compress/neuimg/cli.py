"""`dc neuimg ...` commands: probe, compress, verify and restore imaging TIFF stacks."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from data_compress.neuimg.encode import DEFAULT_CLEVEL, run_compress, run_decompress, verify
from data_compress.neuimg.probe import ProbeReport, build_probe_report
from data_compress.parallel import default_jobs

console = Console()

app = typer.Typer(
    help="Losslessly compress widefield / cellular imaging TIFF stacks to Zarr.",
    context_settings={"help_option_names": ["-h", "--help"]},
)

TifDirArg = typer.Argument(..., exists=True, file_okay=False, dir_okay=True)
JobsOpt = typer.Option(
    None,
    "--jobs",
    "-j",
    min=1,
    help=f"Worker processes (default: CPU count, up to {default_jobs()} on this machine)",
)


def _mb(n_bytes: int) -> str:
    return f"{n_bytes / 1e6:.1f} MB"


def _print_probe_report(report: ProbeReport) -> None:
    height, width = report.shape
    table = Table(title=f"TIFF probe: {report.tif_dir}")
    table.add_column("field")
    table.add_column("value")
    table.add_row("files", str(len(report.files)))
    table.add_row("frames", str(report.frame_count))
    table.add_row("resolution", f"{width}x{height}")
    table.add_row("dtype", str(report.dtype))
    table.add_row("size", _mb(report.size_bytes))
    console.print(table)


@app.command()
def probe(tif_dir: Path = TifDirArg) -> None:
    """Inspect a TIFF directory. Read-only."""
    try:
        report = build_probe_report(tif_dir)
    except ValueError as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc
    _print_probe_report(report)


@app.command()
def compress(
        tif_dir: Path = TifDirArg,
        output: Path = typer.Argument(..., help="Output .zarr directory"),
        fps: float | None = typer.Option(
            None, help="Frame rate, stored as the time-axis scale (omit if unknown)"
        ),
        clevel: int = typer.Option(DEFAULT_CLEVEL, help="Blosc zstd level 0-9 (higher: smaller, slower)"),
        overwrite: bool = typer.Option(False, help="Replace an existing output"),
        no_verify: bool = typer.Option(False, "--no-verify", help="Skip re-reading and checking the output"),
        jobs: int | None = JobsOpt,
) -> None:
    """Losslessly compress a TIFF directory into a Zarr directory."""
    try:
        _print_probe_report(build_probe_report(tif_dir))
        result = run_compress(
            tif_dir, output, fps=fps, clevel=clevel, overwrite=overwrite, show_progress=True, jobs=jobs
        )
    except (ValueError, FileExistsError) as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        f"wrote {output} (lossless): {_mb(result.original_bytes)} -> {_mb(result.compressed_bytes)} "
        f"({result.ratio:.2f}x)"
    )
    if not no_verify:
        _verify_and_report(tif_dir, output, jobs)


def _verify_and_report(tif_dir: Path, output: Path, jobs: int | None) -> None:
    check = verify(tif_dir, output, show_progress=True, jobs=jobs)

    table = Table(title=f"verify: {output.name} vs {tif_dir.name}")
    table.add_column("check")
    table.add_column("value")
    table.add_row("frames", str(check.frame_count))
    table.add_row("shape and dtype match", "yes" if check.shape_matches else "[bold red]no[/]")
    if check.shape_matches:
        table.add_row("mismatched frames", str(check.mismatched_frames))
    console.print(table)

    if check.passed:
        console.print("[bold green]PASS[/] decoded frames identical to source")
    else:
        console.print(f"[bold red]FAIL[/] decoded {output} does not match {tif_dir}")
        raise typer.Exit(code=1)


@app.command(name="verify")
def verify_command(
        tif_dir: Path = TifDirArg,
        zarr_path: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, help="Compressed .zarr"),
        jobs: int | None = JobsOpt,
) -> None:
    """Re-check a compressed .zarr against its source TIFF directory (e.g. after copying it)."""
    try:
        _verify_and_report(tif_dir, zarr_path, jobs)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def decompress(
        zarr_path: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, help="Compressed .zarr"),
        output_dir: Path = typer.Argument(..., help="Directory to write the restored TIFF files into"),
        overwrite: bool = typer.Option(False, help="Replace existing TIFF files"),
        jobs: int | None = JobsOpt,
) -> None:
    """Restore a compressed recording to TIFF files with the original names."""
    try:
        written = run_decompress(zarr_path, output_dir, overwrite=overwrite, show_progress=True, jobs=jobs)
    except (ValueError, FileExistsError, KeyError) as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[bold green]Done[/] wrote {len(written)} TIFF file(s) to {output_dir}")
