from pathlib import Path

import numpy as np
import pytest

from data_compress.ephys.spikeglx import find_meta, open_bin, parse_meta, stream_type
from tests.ephys.spikeglx_fixtures import make_traces, write_spikeglx


def test_parse_meta_ap_stream(tmp_path: Path) -> None:
    bin_path = write_spikeglx(tmp_path, make_traces(100, n_neural=8))

    meta = parse_meta(find_meta(bin_path))

    assert meta.n_saved_chans == 9
    assert meta.n_sync_chans == 1
    assert meta.n_neural_chans == 8
    assert meta.sample_rate == 30000.0
    assert meta.file_size_bytes == 100 * 9 * 2
    assert meta.raw["typeThis"] == "imec"
    assert meta.raw["~snsChanMap"] == "(8,0,1)"


def test_parse_meta_lf_stream(tmp_path: Path) -> None:
    bin_path = write_spikeglx(tmp_path, make_traces(100, n_neural=4), stream="lf", sample_rate=2500.0)

    meta = parse_meta(find_meta(bin_path))

    assert meta.sample_rate == 2500.0
    assert meta.n_sync_chans == 1
    assert meta.n_neural_chans == 4


def test_parse_meta_nidq_stream_treats_digital_words_as_sync(tmp_path: Path) -> None:
    traces = make_traces(100, n_neural=3, n_sync=2)
    bin_path = write_spikeglx(tmp_path, traces, stream="nidq", n_sync=2, sample_rate=25000.0)

    meta = parse_meta(find_meta(bin_path))

    assert meta.sample_rate == 25000.0
    assert meta.n_sync_chans == 2
    assert meta.n_neural_chans == 3


def test_parse_meta_keeps_original_text(tmp_path: Path) -> None:
    bin_path = write_spikeglx(tmp_path, make_traces(10, n_neural=2))
    meta_path = find_meta(bin_path)

    assert parse_meta(meta_path).text == meta_path.read_text()


def test_find_meta_missing_raises(tmp_path: Path) -> None:
    bin_path = tmp_path / "run_g0_t0.imec0.ap.bin"
    bin_path.write_bytes(b"")

    with pytest.raises(FileNotFoundError, match=r"\.meta"):
        find_meta(bin_path)


def test_open_bin_returns_samples_by_channels(tmp_path: Path) -> None:
    traces = make_traces(100, n_neural=8)
    bin_path = write_spikeglx(tmp_path, traces)

    data = open_bin(bin_path, parse_meta(find_meta(bin_path)))

    assert data.shape == (100, 9)
    assert data.dtype == np.int16
    np.testing.assert_array_equal(data, traces)


def test_open_bin_rejects_size_mismatch_with_meta(tmp_path: Path) -> None:
    bin_path = write_spikeglx(tmp_path, make_traces(100, n_neural=8), file_size_bytes=999 * 18)

    with pytest.raises(ValueError, match="fileSizeBytes"):
        open_bin(bin_path, parse_meta(find_meta(bin_path)))


def test_open_bin_rejects_partial_sample(tmp_path: Path) -> None:
    bin_path = write_spikeglx(tmp_path, make_traces(100, n_neural=8))
    with bin_path.open("ab") as f:
        f.write(b"\x00\x00")
    meta = parse_meta(find_meta(bin_path))
    object.__setattr__(meta, "file_size_bytes", None)

    with pytest.raises(ValueError, match="not a whole number of samples"):
        open_bin(bin_path, meta)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("run_g0_t0.imec0.ap.bin", "ap"),
        ("run_g0_t0.imec0.lf.bin", "lf"),
        ("run_g0_t0.nidq.bin", "nidq"),
        ("whatever.bin", "unknown"),
    ],
)
def test_stream_type_from_filename(name: str, expected: str) -> None:
    assert stream_type(Path(name)) == expected
