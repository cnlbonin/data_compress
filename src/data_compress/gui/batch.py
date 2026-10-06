"""Batch planning and execution for the GUI. Has no windowing dependency, so it is testable on its own.

A session is one unit of work: a TIFF directory (behav, neuimg) or a SpikeGLX `.bin`
(ephys). Outputs go next to the source, using the same names as the CLI batch examples.
"""

from __future__ import annotations

import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from data_compress.behav.encode import run_encode, verify_frame_count
from data_compress.ephys.encode import run_compress as run_ephys_compress
from data_compress.ephys.encode import verify as verify_ephys
from data_compress.neuimg.encode import run_compress as run_neuimg_compress
from data_compress.neuimg.encode import verify as verify_neuimg
from data_compress.tiff_source import TIFF_SUFFIXES

DataType = Literal["behav", "ephys", "neuimg"]
DATA_TYPE_LABELS: dict[str, str] = {
    "behav": "Behavioral video (TIFF → mp4 / mkv)",
    "ephys": "Ephys (SpikeGLX .bin → Zarr)",
    "neuimg": "Imaging (TIFF → lossless Zarr)",
}
DATA_TYPES: tuple[DataType, ...] = ("behav", "ephys", "neuimg")


@dataclass(frozen=True)
class Options:
    behav_codec: str = "lossy"  # "lossy" (mp4) or "lossless" (mkv)
    behav_fps: float | None = None  # None = read from each session's .camlog
    ephys_bps: float | None = None  # None = lossless
    verify: bool = True


@dataclass(frozen=True)
class Session:
    data_type: DataType
    source: Path
    output: Path


@dataclass(frozen=True)
class Result:
    session: Session
    status: Literal["ok", "skipped", "failed", "cancelled"]
    message: str


ProgressFn = Callable[[str, int, int], None]  # (phase, done, total)


def output_for(data_type: DataType, source: Path, options: Options) -> Path:
    if data_type == "behav":
        ext = ".mp4" if options.behav_codec == "lossy" else ".mkv"
        return source.with_name(source.name + ext)  # sibling of the run folder, as in the CLI batch loop
    if data_type == "neuimg":
        return source.with_name(source.name + ".zarr")
    return source.with_suffix(".zarr")  # ephys: run_g0.imec0.ap.bin -> run_g0.imec0.ap.zarr


def find_sessions(data_type: DataType, root: Path) -> list[Path]:
    """Find sessions under `root`: TIFF directories (any depth) or `.bin` files with a `.meta` sibling."""
    if data_type == "ephys":
        bins = sorted(p for p in root.rglob("*.bin") if p.is_file())
        return [p for p in bins if p.with_suffix(".meta").is_file()]
    dirs = {p.parent for p in root.rglob("*") if p.is_file() and p.suffix.lower() in TIFF_SUFFIXES}
    return sorted(dirs)


def make_session(data_type: DataType, source: Path, options: Options) -> Session:
    return Session(data_type=data_type, source=source, output=output_for(data_type, source, options))


def run_session(session: Session, options: Options, progress: ProgressFn) -> Result:
    if session.output.exists():
        return Result(session, "skipped", f"{session.output.name} already exists")
    try:
        message = _compress(session, options, progress)
        if options.verify:
            progress("verifying", 0, 1)
            message += "; " + _verify(session, options, progress)
        return Result(session, "ok", message)
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, OSError) as exc:
        _remove_partial(session.output)  # we checked it did not exist, so anything there is ours
        return Result(session, "failed", str(exc))


def _remove_partial(output: Path) -> None:
    if output.is_dir():
        shutil.rmtree(output, ignore_errors=True)
    else:
        output.unlink(missing_ok=True)


def _compress(session: Session, options: Options, progress: ProgressFn) -> str:
    def on_progress(done: int, total: int) -> None:
        progress("compressing", done, total)

    if session.data_type == "behav":
        result = run_encode(
            session.source,
            session.output,
            codec=options.behav_codec,
            fps=options.behav_fps,
            on_progress=on_progress,
        )
        return f"wrote {result.frame_count_written} frames"
    if session.data_type == "ephys":
        result = run_ephys_compress(session.source, session.output, bps=options.ephys_bps, on_progress=on_progress)
    else:
        result = run_neuimg_compress(session.source, session.output, on_progress=on_progress)
    return f"{result.ratio:.2f}x"


def _verify(session: Session, options: Options, progress: ProgressFn) -> str:
    def on_progress(done: int, total: int) -> None:
        progress("verifying", done, total)

    if session.data_type == "behav":
        frames = verify_frame_count(session.output)
        return f"verified {frames} frames"
    if session.data_type == "ephys":
        check = verify_ephys(session.source, session.output, on_progress=on_progress)
    else:
        check = verify_neuimg(session.source, session.output, on_progress=on_progress)
    if not check.passed:
        raise RuntimeError(f"verification FAILED for {session.output.name}")
    return "verified"


def run_batch(
        sessions: list[Session],
        options: Options,
        *,
        on_start: Callable[[int, Session], None],
        on_progress: Callable[[int, str, int, int], None],
        on_result: Callable[[int, Result], None],
        cancel: threading.Event,
) -> list[Result]:
    """Run sessions one after another. A cancel request takes effect between sessions."""
    results: list[Result] = []
    for index, session in enumerate(sessions):
        if cancel.is_set():
            result = Result(session, "cancelled", "not started")
        else:
            on_start(index, session)
            result = run_session(
                session, options, lambda phase, done, total, i=index: on_progress(i, phase, done, total)
            )
        results.append(result)
        on_result(index, result)
    return results
