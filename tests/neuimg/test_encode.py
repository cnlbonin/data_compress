from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr

from data_compress.neuimg.encode import run_compress, run_decompress, verify
from tests.sbx_fixtures import make_frames, write_sbx

H, W = 20, 24


def _write_stack(tif_dir: Path, pages_per_file: list[int], *, dtype=np.uint16, seed: int = 0) -> np.ndarray:
    """Write one TIFF per entry in pages_per_file; return all frames in order."""
    tif_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    frames = rng.integers(0, np.iinfo(dtype).max, size=(sum(pages_per_file), H, W), dtype=dtype)
    start = 0
    for i, n in enumerate(pages_per_file):
        tifffile.imwrite(tif_dir / f"run_{i:04d}.tif", frames[start:start + n], photometric="minisblack")
        start += n
    return frames


def _read_pages(path: Path) -> np.ndarray:
    with tifffile.TiffFile(path) as tif:
        return np.stack([page.asarray() for page in tif.pages])


def _read_tree(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_compress_then_verify_is_lossless(tmp_path: Path, dtype) -> None:
    _write_stack(tmp_path / "src", [3, 4, 2], dtype=dtype)
    output = tmp_path / "out.zarr"

    run_compress(tmp_path / "src", output, jobs=2)
    check = verify(tmp_path / "src", output, jobs=2)

    assert check.passed
    assert check.mismatched_frames == 0


def test_decompress_restores_identical_pixels_and_file_layout(tmp_path: Path) -> None:
    frames = _write_stack(tmp_path / "src", [3, 4, 2])
    output = tmp_path / "out.zarr"
    run_compress(tmp_path / "src", output, jobs=3)

    written = run_decompress(output, tmp_path / "restored", jobs=3)

    assert [p.name for p in written] == ["run_0000.tif", "run_0001.tif", "run_0002.tif"]
    restored = np.concatenate([_read_pages(p) for p in written])
    np.testing.assert_array_equal(restored, frames)
    assert [len(_read_pages(p)) for p in written] == [3, 4, 2]


def test_output_is_independent_of_job_count(tmp_path: Path) -> None:
    _write_stack(tmp_path / "src", [5, 5, 5])
    run_compress(tmp_path / "src", tmp_path / "one.zarr", jobs=1)
    run_compress(tmp_path / "src", tmp_path / "many.zarr", jobs=4)

    assert _read_tree(tmp_path / "one.zarr") == _read_tree(tmp_path / "many.zarr")


def test_root_attrs_carry_ome_axes_and_source_layout(tmp_path: Path) -> None:
    _write_stack(tmp_path / "src", [3, 2])
    output = tmp_path / "out.zarr"

    run_compress(tmp_path / "src", output, fps=10.0, jobs=1)
    root = zarr.open_group(output, mode="r")

    multiscale = root.attrs["ome"]["multiscales"][0]
    assert [axis["name"] for axis in multiscale["axes"]] == ["t", "y", "x"]
    assert multiscale["datasets"][0]["coordinateTransformations"][0]["scale"] == pytest.approx([0.1, 1.0, 1.0])
    assert root.attrs["source_files"] == ["run_0000.tif", "run_0001.tif"]
    assert root.attrs["frames_per_file"] == [3, 2]
    assert root["data"].shape == (5, H, W)
    assert root["data"].chunks == (1, H, W)


def test_verify_fails_when_source_changes(tmp_path: Path) -> None:
    _write_stack(tmp_path / "src", [3, 3], seed=0)
    output = tmp_path / "out.zarr"
    run_compress(tmp_path / "src", output, jobs=1)

    _write_stack(tmp_path / "src", [3, 3], seed=1)  # same layout, different pixels
    check = verify(tmp_path / "src", output, jobs=1)

    assert not check.passed
    assert check.shape_matches
    assert check.mismatched_frames > 0


def test_verify_fails_when_frame_layout_differs(tmp_path: Path) -> None:
    _write_stack(tmp_path / "src", [3, 3])
    output = tmp_path / "out.zarr"
    run_compress(tmp_path / "src", output, jobs=1)

    _write_stack(tmp_path / "src", [2, 4])  # same total frames, different per-file split
    check = verify(tmp_path / "src", output, jobs=1)

    assert not check.passed
    assert not check.shape_matches


def test_compress_refuses_existing_output_without_overwrite(tmp_path: Path) -> None:
    _write_stack(tmp_path / "src", [2])
    output = tmp_path / "out.zarr"
    run_compress(tmp_path / "src", output, jobs=1)

    with pytest.raises(FileExistsError):
        run_compress(tmp_path / "src", output, jobs=1)
    run_compress(tmp_path / "src", output, overwrite=True, jobs=1)
    assert verify(tmp_path / "src", output, jobs=1).passed


def test_compress_rejects_non_zarr_output_and_unsupported_dtype(tmp_path: Path) -> None:
    _write_stack(tmp_path / "src", [2])
    with pytest.raises(ValueError, match=".zarr"):
        run_compress(tmp_path / "src", tmp_path / "out.mkv", jobs=1)

    complex_dir = tmp_path / "complex"
    complex_dir.mkdir()
    tifffile.imwrite(complex_dir / "run_0000.tif", np.zeros((2, H, W), dtype=np.complex64))
    with pytest.raises(ValueError, match="unsupported dtype"):
        run_compress(complex_dir, tmp_path / "out.zarr", jobs=1)
    assert not (tmp_path / "out.zarr").exists()


def test_verify_rejects_a_directory_that_is_not_a_compressed_recording(tmp_path: Path) -> None:
    _write_stack(tmp_path / "src", [2])
    (tmp_path / "plain.zarr").mkdir()
    zarr.open_group(tmp_path / "plain.zarr", mode="w", zarr_format=3)

    with pytest.raises(ValueError, match="not a recording written by"):
        verify(tmp_path / "src", tmp_path / "plain.zarr", jobs=1)


@pytest.mark.parametrize("n_planes", [1, 3])
def test_sbx_compress_then_verify_is_lossless(tmp_path: Path, n_planes: int) -> None:
    frames = make_frames(6, n_planes=n_planes, height=4, width=5, n_channels=2)
    sbx_path = write_sbx(tmp_path / "src", frames)
    output = tmp_path / "out.zarr"

    run_compress(sbx_path, output, jobs=2)
    check = verify(sbx_path, output, jobs=2)

    assert check.passed
    assert check.mismatched_frames == 0


def test_sbx_decompress_restores_identical_sbx_and_mat_bytes(tmp_path: Path) -> None:
    frames = make_frames(5, n_planes=2, height=4, width=5, n_channels=2)
    sbx_path = write_sbx(tmp_path / "src", frames, name="run00")
    mat_path = sbx_path.with_suffix(".mat")
    output = tmp_path / "out.zarr"
    run_compress(sbx_path, output, jobs=3)

    written = run_decompress(output, tmp_path / "restored", jobs=3)

    restored_sbx = tmp_path / "restored" / "run00.sbx"
    restored_mat = tmp_path / "restored" / "run00.mat"
    assert set(written) == {restored_sbx, restored_mat}
    assert restored_sbx.read_bytes() == sbx_path.read_bytes()
    assert restored_mat.read_bytes() == mat_path.read_bytes()


def test_sbx_root_attrs_carry_axes_and_channel_plane_metadata(tmp_path: Path) -> None:
    frames = make_frames(4, n_planes=3, height=4, width=5, n_channels=2)
    sbx_path = write_sbx(tmp_path / "src", frames)
    output = tmp_path / "out.zarr"

    run_compress(sbx_path, output, fps=10.0, jobs=1)
    root = zarr.open_group(output, mode="r")

    multiscale = root.attrs["ome"]["multiscales"][0]
    assert [axis["name"] for axis in multiscale["axes"]] == ["t", "z", "y", "x", "c"]
    assert root.attrs["source_kind"] == "sbx"
    assert root.attrs["n_channels"] == 2
    assert root.attrs["n_planes"] == 3
    assert root.attrs["is_volumetric"] is True
    assert root["data"].shape == (4, 3, 4, 5, 2)
    assert root["data"].chunks == (1, 3, 4, 5, 2)


def test_sbx_verify_fails_when_source_pixels_change(tmp_path: Path) -> None:
    write_sbx(tmp_path / "src", make_frames(5, seed=0))
    sbx_path = tmp_path / "src" / "run00.sbx"
    output = tmp_path / "out.zarr"
    run_compress(sbx_path, output, jobs=1)

    write_sbx(tmp_path / "src", make_frames(5, seed=1))  # same layout, different pixels
    check = verify(sbx_path, output, jobs=1)

    assert not check.passed
    assert check.shape_matches
    assert check.mismatched_frames > 0
