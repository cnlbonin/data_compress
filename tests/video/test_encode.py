import shutil
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


def test_select_codec_profile_lossy_16bit_without_force_raises() -> None:
    with pytest.raises(ValueError, match="16-bit"):
        select_codec_profile(dtype=np.dtype(np.uint16), codec="lossy", crf=None, force=False)


def test_select_codec_profile_lossy_16bit_with_force_succeeds() -> None:
    profile = select_codec_profile(dtype=np.dtype(np.uint16), codec="lossy", crf=None, force=True)

    assert profile.ffmpeg_video_codec == "libx265"


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
