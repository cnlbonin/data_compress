import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import tifffile

from data_compress.video.encode import (
    check_binaries_available,
    resolve_fps,
    run_encode,
    select_codec_profile,
    validate_output_extension,
    verify_frame_count,
)

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not found on PATH",
)


def test_select_codec_profile_lossy_8bit_defaults_to_libx265_mp4() -> None:
    profile = select_codec_profile(dtype=np.dtype(np.uint8), codec="lossy", crf=None, force=False)

    assert profile.container_ext == ".mp4"
    assert profile.ffmpeg_video_codec == "libx265"
    assert profile.pix_fmt == "gray"
    assert "-crf" in profile.extra_args
    assert "18" in profile.extra_args


def test_select_codec_profile_lossy_encodes_as_yuv420p_for_player_compatibility() -> None:
    # Encoding monochrome HEVC directly (pix_fmt=gray, 4:0:0 chroma) forces the
    # "Rext" profile, which QuickTime and most hardware decoders (incl. Apple
    # VideoToolbox) refuse to open. yuv420p keeps the Main profile (broadly
    # playable) at near-zero extra cost, since the added chroma plane is constant.
    profile = select_codec_profile(dtype=np.dtype(np.uint8), codec="lossy", crf=None, force=False)

    assert profile.pix_fmt == "gray"  # raw bytes we actually write to stdin
    assert profile.encode_pix_fmt == "yuv420p"  # what we ask the encoder to produce


def test_select_codec_profile_lossy_16bit_without_force_raises() -> None:
    with pytest.raises(ValueError, match="16-bit"):
        select_codec_profile(dtype=np.dtype(np.uint16), codec="lossy", crf=None, force=False)


def test_select_codec_profile_lossy_16bit_with_force_succeeds() -> None:
    profile = select_codec_profile(dtype=np.dtype(np.uint16), codec="lossy", crf=None, force=True)

    assert profile.ffmpeg_video_codec == "libx265"
    assert profile.pix_fmt == "gray"
    assert profile.downconvert_16_to_8 is True


def test_select_codec_profile_lossy_8bit_does_not_downconvert() -> None:
    profile = select_codec_profile(dtype=np.dtype(np.uint8), codec="lossy", crf=None, force=False)

    assert profile.downconvert_16_to_8 is False


def test_select_codec_profile_rejects_unsupported_dtype() -> None:
    with pytest.raises(ValueError, match="unsupported dtype"):
        select_codec_profile(dtype=np.dtype(np.float32), codec="lossy", crf=None, force=False)


def test_select_codec_profile_lossless_16bit_uses_ffv1_mkv() -> None:
    profile = select_codec_profile(dtype=np.dtype(np.uint16), codec="lossless", crf=None, force=False)

    assert profile.container_ext == ".mkv"
    assert profile.ffmpeg_video_codec == "ffv1"
    assert profile.pix_fmt == "gray16le"


def test_select_codec_profile_lossless_8bit_uses_gray_pix_fmt() -> None:
    profile = select_codec_profile(dtype=np.dtype(np.uint8), codec="lossless", crf=None, force=False)

    assert profile.pix_fmt == "gray"


def test_validate_output_extension_mismatch_raises(tmp_path: Path) -> None:
    profile = select_codec_profile(dtype=np.dtype(np.uint8), codec="lossy", crf=None, force=False)

    with pytest.raises(ValueError, match="extension"):
        validate_output_extension(tmp_path / "out.mkv", profile)


def test_validate_output_extension_match_is_ok(tmp_path: Path) -> None:
    profile = select_codec_profile(dtype=np.dtype(np.uint8), codec="lossy", crf=None, force=False)

    validate_output_extension(tmp_path / "out.mp4", profile)


def test_resolve_fps_prefers_explicit_value(tmp_path: Path) -> None:
    (tmp_path / "run.camlog").write_text(
        "# Log header:frame_id,timestamp\n1,0.0\n2,0.1\n"
    )

    assert resolve_fps(tmp_path, explicit_fps=60.0) == 60.0


def test_resolve_fps_falls_back_to_camlog(tmp_path: Path) -> None:
    rows = "\n".join(f"{i},{i / 30.0}" for i in range(1, 32))
    (tmp_path / "run.camlog").write_text("# Log header:frame_id,timestamp\n" + rows + "\n")

    fps = resolve_fps(tmp_path, explicit_fps=None)

    assert fps == pytest.approx(30.0, rel=1e-3)


def test_resolve_fps_raises_when_no_fps_and_no_camlog(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--fps"):
        resolve_fps(tmp_path, explicit_fps=None)


def test_check_binaries_available_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda name: None)

    with pytest.raises(RuntimeError, match="ffmpeg"):
        check_binaries_available()


def test_check_binaries_available_passes_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")

    check_binaries_available()


def _ffprobe_video_profile(path: Path) -> str:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=profile",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@requires_ffmpeg
def test_run_encode_lossy_produces_main_profile_not_rext(tmp_path: Path) -> None:
    frames = np.stack([np.full((64, 64), i, dtype=np.uint8) for i in range(5)])
    tifffile.imwrite(tmp_path / "run_0000.tif", frames, photometric="minisblack")

    output = tmp_path / "out.mp4"
    run_encode(tmp_path, output, fps=10.0, codec="lossy")

    assert _ffprobe_video_profile(output) == "Main"


@requires_ffmpeg
def test_run_encode_writes_video_with_matching_frame_count(tmp_path: Path) -> None:
    n_frames = 7
    frames = np.stack(
        [np.full((64, 64), i * 10, dtype=np.uint8) for i in range(n_frames)]
    )
    tifffile.imwrite(tmp_path / "run_0000.tif", frames, photometric="minisblack")

    output = tmp_path / "out.mp4"
    result = run_encode(tmp_path, output, fps=10.0, codec="lossy")

    assert result.frame_count_written == n_frames
    assert output.exists()
    assert verify_frame_count(output) == n_frames


@requires_ffmpeg
def test_run_encode_lossy_16bit_with_force_writes_correct_frame_count(tmp_path: Path) -> None:
    n_frames = 5
    frames = np.stack(
        [np.full((64, 64), i * 1000, dtype=np.uint16) for i in range(n_frames)]
    )
    tifffile.imwrite(tmp_path / "run_0000.tif", frames, photometric="minisblack")

    output = tmp_path / "out.mp4"
    result = run_encode(tmp_path, output, fps=10.0, codec="lossy", force=True)

    assert result.frame_count_written == n_frames
    assert verify_frame_count(output) == n_frames


@requires_ffmpeg
def test_run_encode_reports_clean_error_when_ffmpeg_dies_mid_stream(tmp_path: Path) -> None:
    # An image this small makes libx265 refuse to open the encoder and exit almost
    # immediately. Writing enough frames to exceed the OS pipe buffer guarantees we
    # keep writing to stdin after ffmpeg has already exited and closed its end —
    # this must surface as a clean RuntimeError, not a raw BrokenPipeError, and must
    # not leave the ffmpeg child process running.
    n_frames = 20_000
    frames = np.stack(
        [np.full((4, 6), i % 256, dtype=np.uint8) for i in range(n_frames)]
    )
    tifffile.imwrite(tmp_path / "run_0000.tif", frames, photometric="minisblack")

    output = tmp_path / "out.mp4"
    with pytest.raises(RuntimeError, match="ffmpeg"):
        run_encode(tmp_path, output, fps=10.0, codec="lossy")


@requires_ffmpeg
def test_run_encode_lossless_16bit_writes_ffv1_mkv(tmp_path: Path) -> None:
    n_frames = 5
    frames = np.stack(
        [np.full((64, 64), i, dtype=np.uint16) for i in range(n_frames)]
    )
    tifffile.imwrite(tmp_path / "run_0000.tif", frames, photometric="minisblack")

    output = tmp_path / "out.mkv"
    result = run_encode(tmp_path, output, fps=10.0, codec="lossless")

    assert result.frame_count_written == n_frames
    assert verify_frame_count(output) == n_frames
