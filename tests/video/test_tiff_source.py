from pathlib import Path

import numpy as np
import pytest
import tifffile

from data_compress.video.tiff_source import iter_frames, scan_tiff_dir


def _write_stack(path: Path, n_pages: int, value_start: int, *, shape=(4, 5), dtype=np.uint8) -> None:
    frames = np.stack(
        [np.full(shape, value_start + i, dtype=dtype) for i in range(n_pages)]
    )
    tifffile.imwrite(path, frames, photometric="minisblack")


def test_scan_tiff_dir_sums_frames_across_files_and_reads_dtype_shape(tmp_path: Path) -> None:
    _write_stack(tmp_path / "run_00000000.tif", n_pages=3, value_start=0)
    _write_stack(tmp_path / "run_00000001.tif", n_pages=2, value_start=10)

    info = scan_tiff_dir(tmp_path)

    assert info.frame_count == 5
    assert info.shape == (4, 5)
    assert info.dtype == np.uint8
    assert info.files == [
        tmp_path / "run_00000000.tif",
        tmp_path / "run_00000001.tif",
    ]


def test_scan_tiff_dir_raises_when_no_tiff_files_found(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no .tif"):
        scan_tiff_dir(tmp_path)


def test_scan_tiff_dir_sorts_files_naturally_not_lexicographically(tmp_path: Path) -> None:
    # Lexicographic sort would put file_10 before file_2.
    _write_stack(tmp_path / "file_2.tif", n_pages=1, value_start=0)
    _write_stack(tmp_path / "file_10.tif", n_pages=1, value_start=0)
    _write_stack(tmp_path / "file_1.tif", n_pages=1, value_start=0)

    info = scan_tiff_dir(tmp_path)

    assert [f.name for f in info.files] == ["file_1.tif", "file_2.tif", "file_10.tif"]


def test_iter_frames_yields_frames_in_order_across_files(tmp_path: Path) -> None:
    _write_stack(tmp_path / "run_00000000.tif", n_pages=2, value_start=0)
    _write_stack(tmp_path / "run_00000001.tif", n_pages=2, value_start=10)

    frames = list(iter_frames(tmp_path))

    assert [int(frame[0, 0]) for frame in frames] == [0, 1, 10, 11]
