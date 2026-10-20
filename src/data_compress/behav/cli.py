"""`dcp behav ...` commands: probe and compress TIFF sequences."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from data_compress.behav.encode import run_encode, verify_frame_count
from data_compress.behav.probe import ProbeReport, build_probe_report

app = typer.Typer(
    help="Inspect and compress behavioral-camera TIFF sequences.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
console = Console()

TifDirArg = typer.Argument(..., exists=True, file_okay=False, dir_okay=True)


def _print_probe_report(report: ProbeReport) -> None:
    table = Table(title=f"TIFF probe: {report.tif_dir}")
    table.add_column("field")
    table.add_column("value")
    table.add_row("files", str(len(report.files)))
    table.add_row("frame count (tiff)", str(report.frame_count))
    height, width = report.shape
    table.add_row("resolution", f"{width}x{height}")
    table.add_row("dtype", str(report.dtype))
    if report.camlog_path is not None:
        table.add_row("camlog", report.camlog_path.name)
        table.add_row("frame count (camlog)", str(report.camlog_frame_count))
        table.add_row("fps (camlog)", f"{report.camlog_fps:.3f}")
        if report.frame_count_mismatch:
            table.add_row(
                "[bold red]WARNING[/]",
                "frame count mismatch between TIFF pages and camlog rows",
            )
    else:
        table.add_row("camlog", "not found")
    console.print(table)


@app.command()
def probe(tif_dir: Path = TifDirArg) -> None:
    """Inspect a TIFF directory (and its .camlog sidecar, if present). Read-only."""
    report = build_probe_report(tif_dir)
    _print_probe_report(report)


@app.command()
def compress(
    tif_dir: Path = TifDirArg,
    output: Path = typer.Argument(...),
    fps: float | None = typer.Option(None, help="Explicit fps; else parsed from .camlog"),
    codec: str = typer.Option("lossy", help="'lossy' (libx265/mp4) or 'lossless' (FFV1/mkv)"),
    crf: int | None = typer.Option(
        None, help="CRF (cpu encoder) or CQ (nvenc encoder) for the lossy codec (default 18)"
    ),
    force: bool = typer.Option(
        False, help="Allow a 16-bit source through the lossy (bit-depth-truncating) path"
    ),
    encoder: str = typer.Option(
        "cpu",
        help="'cpu' (libx265) or 'nvenc' (GPU hevc_nvenc, much faster; lossy codec only)",
    ),
    preset: str | None = typer.Option(
        None,
        help="Encoder preset, e.g. ultrafast..placebo for cpu, p1..p7 for nvenc (default: ffmpeg's own default)",
    ),
    verify: bool = typer.Option(
        False, help="Verify the written frame count with ffprobe after encoding"
    ),
) -> None:
    """Compress a TIFF directory into a single video file."""
    report = build_probe_report(tif_dir)
    _print_probe_report(report)

    try:
        result = run_encode(
            tif_dir,
            output,
            fps=fps,
            codec=codec,
            crf=crf,
            force=force,
            encoder=encoder,
            preset=preset,
            show_progress=True,
        )
    except (ValueError, RuntimeError) as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc

    if not verify:
        console.print(f"wrote {result.frame_count_written} frames to {output}")
        return

    with console.status("verifying frame count with ffprobe..."):
        actual = verify_frame_count(output)
    if actual == result.frame_count_written:
        console.print(f"[bold green]PASS[/] wrote {actual} frames to {output}")
    else:
        console.print(
            f"[bold red]FAIL[/] expected {result.frame_count_written} frames, "
            f"ffprobe reports {actual} in {output}"
        )
        raise typer.Exit(code=1)
