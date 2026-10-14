from pathlib import Path

import numpy as np
import pytest

from data_compress.sbx_source import find_mat, iter_frame_range, scan_sbx_file
from tests.sbx_fixtures import make_frames, write_sbx


def test_scan_sbx_file_reads_shape_dtype_and_frame_count(tmp_path: Path) -> None:
    frames = make_frames(5, height=4, width=5, n_channels=2)
    sbx_path = write_sbx(tmp_path, frames)

    info = scan_sbx_file(sbx_path)

    assert info.n_timepoints == 5
    assert info.frame_count == 5
    assert (info.n_planes, info.height, info.width, info.n_channels) == (1, 4, 5, 2)
    assert info.frame_shape == (1, 4, 5, 2)
    assert info.dtype == np.dtype("<u2")
    assert not info.is_volumetric
    assert info.mat_bytes == info.mat_path.read_bytes()


def test_scan_sbx_file_reads_nchan_from_newer_chan_struct(tmp_path: Path) -> None:
    frames = make_frames(3, n_channels=1)
    sbx_path = write_sbx(tmp_path, frames, legacy_channels_field=False)

    info = scan_sbx_file(sbx_path)

    assert info.n_channels == 1
    assert info.n_timepoints == 3


def test_scan_sbx_file_detects_volumetric_recordings(tmp_path: Path) -> None:
    frames = make_frames(4, n_planes=3, n_channels=1)
    sbx_path = write_sbx(tmp_path, frames)

    info = scan_sbx_file(sbx_path)

    assert info.n_planes == 3
    assert info.is_volumetric
    assert info.frame_shape == (3, 4, 5, 1)


def test_find_mat_missing_raises(tmp_path: Path) -> None:
    sbx_path = tmp_path / "run00.sbx"
    sbx_path.write_bytes(b"")

    with pytest.raises(FileNotFoundError, match=r"\.mat"):
        find_mat(sbx_path)


def test_scan_sbx_file_rejects_truncated_file(tmp_path: Path) -> None:
    sbx_path = write_sbx(tmp_path, make_frames(5))
    with sbx_path.open("ab") as f:
        f.write(b"\x00")  # one stray byte: not a whole number of timepoints

    with pytest.raises(ValueError, match="not a whole number of timepoints"):
        scan_sbx_file(sbx_path)


def test_iter_frame_range_yields_blocks_identical_to_source(tmp_path: Path) -> None:
    frames = make_frames(6, n_planes=2, height=3, width=3, n_channels=2)
    sbx_path = write_sbx(tmp_path, frames)
    info = scan_sbx_file(sbx_path)

    blocks = list(iter_frame_range(info, 2, 5))

    assert len(blocks) == 3
    for offset, block in enumerate(blocks):
        np.testing.assert_array_equal(block, frames[2 + offset])
