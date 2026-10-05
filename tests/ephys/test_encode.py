from pathlib import Path

import numpy as np
import pytest
import zarr

from data_compress.ephys.encode import (
    _chunk_ranges,
    run_compress,
    run_decompress,
    validate_bps,
    verify,
)
from tests.ephys.spikeglx_fixtures import make_traces, write_spikeglx

RATE = 1000.0  # small fake rate so 1-s chunks stay tiny in tests


def _recording(tmp_path: Path, n_samples: int = 3500, n_neural: int = 16, n_sync: int = 1):
    traces = make_traces(n_samples, n_neural=n_neural, n_sync=n_sync)
    src = tmp_path / "src"
    src.mkdir()
    return traces, write_spikeglx(src, traces, n_sync=n_sync, sample_rate=RATE)


def test_lossless_compress_round_trips_exactly(tmp_path: Path) -> None:
    traces, bin_path = _recording(tmp_path)
    out = tmp_path / "rec.zarr"

    result = run_compress(bin_path, out)

    root = zarr.open_group(out, mode="r")
    np.testing.assert_array_equal(root["traces_seg0"][:], traces[:, :16])
    np.testing.assert_array_equal(root["sync_seg0"][:], traces[:, 16:])
    assert result.n_samples == 3500
    assert result.original_bytes == traces.nbytes
    assert result.ratio > 1.0


def test_compress_writes_spikeinterface_style_layout(tmp_path: Path) -> None:
    _, bin_path = _recording(tmp_path)
    out = tmp_path / "rec.zarr"

    run_compress(bin_path, out)

    root = zarr.open_group(out, mode="r")
    assert root.attrs["sampling_frequency"] == RATE
    assert root.attrs["num_segments"] == 1
    assert root.attrs["spikeglx_meta"]["nSavedChans"] == "17"
    assert root.attrs["compression"]["bps"] is None
    np.testing.assert_array_equal(root["channel_ids"][:], np.arange(16))
    # 1-s chunks over all neural channels, as in the paper's pipeline
    assert root["traces_seg0"].chunks == (1000, 16)


def test_lossy_compress_keeps_sync_exact_and_is_smaller(tmp_path: Path) -> None:
    traces, bin_path = _recording(tmp_path)

    lossless = run_compress(bin_path, tmp_path / "lossless.zarr")
    lossy = run_compress(bin_path, tmp_path / "lossy.zarr", bps=2.25)

    root = zarr.open_group(tmp_path / "lossy.zarr", mode="r")
    np.testing.assert_array_equal(root["sync_seg0"][:], traces[:, 16:])
    assert not np.array_equal(root["traces_seg0"][:], traces[:, :16])
    assert lossy.compressed_bytes < lossless.compressed_bytes
    assert root.attrs["compression"]["bps"] == 2.25


def test_recording_without_sync_channel_has_no_sync_array(tmp_path: Path) -> None:
    traces, bin_path = _recording(tmp_path, n_sync=0)
    out = tmp_path / "rec.zarr"

    run_compress(bin_path, out)

    root = zarr.open_group(out, mode="r")
    assert "sync_seg0" not in root
    np.testing.assert_array_equal(root["traces_seg0"][:], traces)


def test_compress_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    _, bin_path = _recording(tmp_path)
    out = tmp_path / "rec.zarr"
    out.mkdir()

    with pytest.raises(FileExistsError):
        run_compress(bin_path, out)

    run_compress(bin_path, out, overwrite=True)
    assert (out / ".zattrs").exists() or (out / "zarr.json").exists()


def test_compress_requires_zarr_extension(tmp_path: Path) -> None:
    _, bin_path = _recording(tmp_path)

    with pytest.raises(ValueError, match=r"\.zarr"):
        run_compress(bin_path, tmp_path / "rec.out")


@pytest.mark.parametrize("bps", [1.0, 2.0, 16.0])
def test_validate_bps_rejects_out_of_range(bps: float) -> None:
    with pytest.raises(ValueError, match="bps"):
        validate_bps(bps)


@pytest.mark.parametrize("bps", [None, 2.25, 3.0])
def test_validate_bps_accepts_supported_values(bps: float | None) -> None:
    validate_bps(bps)


def test_verify_lossless_reports_exact(tmp_path: Path) -> None:
    _, bin_path = _recording(tmp_path)
    out = tmp_path / "rec.zarr"
    run_compress(bin_path, out)

    result = verify(bin_path, out)

    assert result.exact
    assert result.sync_exact
    assert result.max_abs_error == 0
    assert result.relative_rms_error == 0.0
    assert result.passed


def test_verify_lossy_reports_small_error_and_passes_on_exact_sync(tmp_path: Path) -> None:
    _, bin_path = _recording(tmp_path)
    out = tmp_path / "rec.zarr"
    run_compress(bin_path, out, bps=3)

    result = verify(bin_path, out)

    assert not result.exact
    assert result.sync_exact
    assert 0 < result.relative_rms_error < 0.5
    assert result.passed


def test_verify_lossless_fails_when_data_differs(tmp_path: Path) -> None:
    _, bin_path = _recording(tmp_path)
    out = tmp_path / "rec.zarr"
    run_compress(bin_path, out)
    root = zarr.open_group(out, mode="r+")
    root["traces_seg0"][10, 3] = root["traces_seg0"][10, 3] + 1

    result = verify(bin_path, out)

    assert not result.exact
    assert not result.passed


def test_decompress_lossless_restores_identical_bin_and_meta(tmp_path: Path) -> None:
    _, bin_path = _recording(tmp_path)
    out = tmp_path / "rec.zarr"
    run_compress(bin_path, out)
    restored = tmp_path / "restored" / "run_g0_t0.imec0.ap.bin"

    run_decompress(out, restored)

    assert restored.read_bytes() == bin_path.read_bytes()
    assert restored.with_suffix(".meta").read_text() == bin_path.with_suffix(".meta").read_text()


def test_decompress_refuses_to_overwrite(tmp_path: Path) -> None:
    _, bin_path = _recording(tmp_path)
    out = tmp_path / "rec.zarr"
    run_compress(bin_path, out)

    with pytest.raises(FileExistsError):
        run_decompress(out, bin_path)


def test_decompress_requires_bin_extension(tmp_path: Path) -> None:
    _, bin_path = _recording(tmp_path)
    out = tmp_path / "rec.zarr"
    run_compress(bin_path, out)

    with pytest.raises(ValueError, match=r"\.bin"):
        run_decompress(out, tmp_path / "restored.dat")


def test_decompress_preserves_crlf_meta_byte_for_byte(tmp_path: Path) -> None:
    # SpikeGLX on Windows writes .meta files with CRLF line endings
    _, bin_path = _recording(tmp_path)
    meta_path = bin_path.with_suffix(".meta")
    meta_path.write_bytes(meta_path.read_bytes().replace(b"\n", b"\r\n"))
    out = tmp_path / "rec.zarr"
    run_compress(bin_path, out)
    restored = tmp_path / "restored" / bin_path.name

    run_decompress(out, restored)

    assert restored.with_suffix(".meta").read_bytes() == meta_path.read_bytes()


@pytest.mark.parametrize(
    ("n_chunks", "jobs"), [(0, 4), (1, 4), (3, 8), (4, 1), (100, 3), (1000, 8)]
)
def test_chunk_ranges_cover_every_chunk_once_in_order(n_chunks: int, jobs: int) -> None:
    ranges = _chunk_ranges(n_chunks, jobs)

    covered = [i for start, stop in ranges for i in range(start, stop)]
    assert covered == list(range(n_chunks))
    assert all(start < stop for start, stop in ranges)


def test_parallel_compress_matches_serial(tmp_path: Path) -> None:
    traces, bin_path = _recording(tmp_path, n_samples=7300)

    run_compress(bin_path, tmp_path / "serial.zarr", jobs=1)
    run_compress(bin_path, tmp_path / "parallel.zarr", jobs=3)

    serial = zarr.open_group(tmp_path / "serial.zarr", mode="r")
    parallel = zarr.open_group(tmp_path / "parallel.zarr", mode="r")
    np.testing.assert_array_equal(parallel["traces_seg0"][:], traces[:, :16])
    np.testing.assert_array_equal(parallel["sync_seg0"][:], traces[:, 16:])
    np.testing.assert_array_equal(parallel["traces_seg0"][:], serial["traces_seg0"][:])


def test_parallel_verify_matches_serial(tmp_path: Path) -> None:
    _, bin_path = _recording(tmp_path, n_samples=7300)
    out = tmp_path / "rec.zarr"
    run_compress(bin_path, out, bps=3, jobs=2)

    serial = verify(bin_path, out, jobs=1)
    parallel = verify(bin_path, out, jobs=3)

    assert parallel.passed == serial.passed
    assert parallel.sync_exact == serial.sync_exact
    assert parallel.max_abs_error == serial.max_abs_error
    assert parallel.relative_rms_error == pytest.approx(serial.relative_rms_error)


def test_parallel_verify_still_detects_a_bad_chunk(tmp_path: Path) -> None:
    _, bin_path = _recording(tmp_path, n_samples=7300)
    out = tmp_path / "rec.zarr"
    run_compress(bin_path, out, jobs=2)
    root = zarr.open_group(out, mode="r+")
    root["traces_seg0"][6500, 3] = root["traces_seg0"][6500, 3] + 1

    assert not verify(bin_path, out, jobs=3).passed


def test_parallel_decompress_restores_identical_bin(tmp_path: Path) -> None:
    _, bin_path = _recording(tmp_path, n_samples=7300)
    out = tmp_path / "rec.zarr"
    run_compress(bin_path, out, jobs=2)
    restored = tmp_path / "restored" / bin_path.name

    run_decompress(out, restored, jobs=3)

    assert restored.read_bytes() == bin_path.read_bytes()


def test_jobs_must_be_positive(tmp_path: Path) -> None:
    _, bin_path = _recording(tmp_path)

    with pytest.raises(ValueError, match="jobs"):
        run_compress(bin_path, tmp_path / "rec.zarr", jobs=0)
