"""Build ffmpeg encode configuration and run the TIFF-to-video pipeline."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from data_compress.behav.camlog import find_camlog, parse_camlog
from data_compress.tiff_source import iter_frames, scan_tiff_dir

DEFAULT_LOSSY_CRF = 18
REQUIRED_BINARIES = ("ffmpeg", "ffprobe")


@dataclass(frozen=True)
class CodecProfile:
    container_ext: str
    ffmpeg_video_codec: str
    pix_fmt: str  # raw format of the bytes we actually write to ffmpeg's stdin
    encode_pix_fmt: str  # format we ask the encoder to produce (may differ from pix_fmt)
    extra_args: list[str]
    downconvert_16_to_8: bool = False


def select_codec_profile(*, dtype: np.dtype, codec: str, crf: int | None, force: bool) -> CodecProfile:
    if dtype.kind != "u" or dtype.itemsize not in (1, 2):
        raise ValueError(
            f"unsupported dtype {dtype}; only uint8/uint16 grayscale TIFF sources are supported"
        )
    is_16bit = dtype.itemsize == 2

    if codec == "lossy":
        if is_16bit and not force:
            raise ValueError(
                "16-bit source would be truncated by the lossy codec; "
                "use codec='lossless' or pass force=True to proceed anyway"
            )
        return CodecProfile(
            container_ext=".mp4",
            ffmpeg_video_codec="libx265",
            pix_fmt="gray",
            # Encoding monochrome HEVC directly (4:0:0 chroma) forces the "Rext"
            # profile, which QuickTime and most hardware decoders (incl. Apple
            # VideoToolbox) refuse to open. yuv420p keeps the Main profile, at
            # near-zero extra cost since the added chroma plane is constant.
            encode_pix_fmt="yuv420p",
            # Apple players (QuickTime/QuickLook) only accept HEVC-in-mp4 tagged hvc1;
            # ffmpeg defaults to hev1.
            extra_args=[
                "-crf",
                str(crf if crf is not None else DEFAULT_LOSSY_CRF),
                "-tag:v",
                "hvc1",
            ],
            # forcing a 16-bit source through this 8-bit pix_fmt requires actually
            # truncating each frame's bit depth, or ffmpeg's declared frame byte-size
            # (1 byte/px) desyncs from the raw stream we send (2 bytes/px) and it
            # silently decodes garbage as twice as many frames.
            downconvert_16_to_8=is_16bit,
        )

    if codec == "lossless":
        pix_fmt = "gray16le" if is_16bit else "gray"
        return CodecProfile(
            container_ext=".mkv",
            ffmpeg_video_codec="ffv1",
            pix_fmt=pix_fmt,
            encode_pix_fmt=pix_fmt,
            extra_args=[],
        )

    raise ValueError(f"unknown codec {codec!r}; expected 'lossy' or 'lossless'")


def validate_output_extension(output: Path, profile: CodecProfile) -> None:
    if output.suffix.lower() != profile.container_ext:
        raise ValueError(
            f"output extension {output.suffix!r} does not match codec's expected "
            f"container extension {profile.container_ext!r} for this output"
        )


def resolve_fps(tif_dir: Path, explicit_fps: float | None) -> float:
    if explicit_fps is not None:
        return explicit_fps

    camlog_path = find_camlog(tif_dir)
    if camlog_path is not None:
        return parse_camlog(camlog_path).fps

    raise ValueError(f"no fps given and no .camlog sidecar found in {tif_dir}; pass --fps explicitly")


def check_binaries_available() -> None:
    missing = [name for name in REQUIRED_BINARIES if shutil.which(name) is None]
    if missing:
        raise RuntimeError(
            f"required binaries not found on PATH: {', '.join(missing)}. "
            "Install ffmpeg (e.g. `brew install ffmpeg`)."
        )


@dataclass(frozen=True)
class EncodeResult:
    output: Path
    frame_count_written: int


def _build_ffmpeg_cmd(*, width: int, height: int, fps: float, profile: CodecProfile, output: Path) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        profile.pix_fmt,
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        profile.ffmpeg_video_codec,
        *profile.extra_args,
        "-pix_fmt",
        profile.encode_pix_fmt,
        str(output),
    ]


def run_encode(
        tif_dir: Path,
        output: Path,
        *,
        fps: float | None = None,
        codec: str = "lossy",
        crf: int | None = None,
        force: bool = False,
        show_progress: bool = False,
        on_progress: Callable[[int, int], None] | None = None,
) -> EncodeResult:
    check_binaries_available()

    info = scan_tiff_dir(tif_dir)
    profile = select_codec_profile(dtype=info.dtype, codec=codec, crf=crf, force=force)
    validate_output_extension(output, profile)
    resolved_fps = resolve_fps(tif_dir, fps)

    height, width = info.shape
    cmd = _build_ffmpeg_cmd(width=width, height=height, fps=resolved_fps, profile=profile, output=output)

    # stderr goes to a real file, not a pipe: ffmpeg can write unboundedly to it
    # without blocking, while we're busy writing (potentially many GB of) frames
    # to stdin with nothing else draining stdout/stderr concurrently.
    with tempfile.TemporaryFile(mode="w+b") as stderr_file:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=stderr_file
        )
        assert proc.stdin is not None

        frame_count_written = 0
        frames = tqdm(
            iter_frames(tif_dir),
            total=info.frame_count,
            unit="frame",
            desc="encoding",
            disable=not show_progress,
        )
        try:
            for frame in frames:
                if profile.downconvert_16_to_8:
                    frame = (frame >> 8).astype(np.uint8)
                proc.stdin.write(frame.tobytes())
                frame_count_written += 1
                if on_progress is not None:
                    on_progress(frame_count_written, info.frame_count)
            frames.close()
            proc.stdin.close()
        except Exception as exc:
            frames.close()
            # ffmpeg may have already died (e.g. rejected the encoder params) while
            # we were still writing frames, surfacing as a raw BrokenPipeError here.
            # Make sure the process is actually gone before reporting a clean error.
            proc.kill()
            proc.wait()
            stderr_file.seek(0)
            raise RuntimeError(
                f"writing frames to ffmpeg failed: {exc}\n"
                f"ffmpeg stderr:\n{stderr_file.read().decode(errors='replace')}"
            ) from exc

        proc.wait()

        if proc.returncode != 0:
            stderr_file.seek(0)
            raise RuntimeError(
                f"ffmpeg failed (exit {proc.returncode}):\n"
                f"{stderr_file.read().decode(errors='replace')}"
            )

    return EncodeResult(output=output, frame_count_written=frame_count_written)


def verify_frame_count(output: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout.strip())
