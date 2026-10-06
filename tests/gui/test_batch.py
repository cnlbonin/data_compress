import threading
from pathlib import Path

import numpy as np
import pytest
import tifffile

from data_compress.gui.batch import (
    Options,
    Session,
    _remove_partial,
    find_sessions,
    make_session,
    output_for,
    run_batch,
    run_session,
)
from tests.ephys.spikeglx_fixtures import make_traces, write_spikeglx


def _tiff_run(directory: Path, n_pages: int = 3, *, seed: int = 0, dtype=np.uint16) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    frames = rng.integers(0, np.iinfo(dtype).max, size=(n_pages, 16, 16), dtype=dtype)
    tifffile.imwrite(directory / "run_0000.tif", frames, photometric="minisblack")


def _noop_progress(*_args) -> None:
    pass


def test_output_names_follow_cli_batch_convention() -> None:
    run = Path("/data/sess/run00")
    assert output_for("behav", run, Options()) == Path("/data/sess/run00.mp4")
    assert output_for("behav", run, Options(behav_codec="lossless")) == Path("/data/sess/run00.mkv")
    assert output_for("neuimg", run, Options()) == Path("/data/sess/run00.zarr")
    assert output_for("ephys", Path("/data/g0.imec0.ap.bin"), Options()) == Path("/data/g0.imec0.ap.zarr")


def test_find_sessions_finds_tiff_dirs_at_any_depth(tmp_path: Path) -> None:
    _tiff_run(tmp_path / "day1" / "run00")
    _tiff_run(tmp_path / "day2" / "run01")
    (tmp_path / "day2" / "notes.txt").write_text("not a tiff")

    found = find_sessions("behav", tmp_path)

    assert found == [tmp_path / "day1" / "run00", tmp_path / "day2" / "run01"]


def test_find_sessions_for_ephys_needs_meta_sibling(tmp_path: Path) -> None:
    traces = make_traces(200, n_neural=4, n_sync=1)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    good = write_spikeglx(tmp_path / "a", traces)
    orphan = write_spikeglx(tmp_path / "b", traces)
    orphan.with_suffix(".meta").unlink()

    assert find_sessions("ephys", tmp_path) == [good]


def test_run_batch_compresses_and_verifies_each_session_with_progress(tmp_path: Path) -> None:
    _tiff_run(tmp_path / "s1", seed=1)
    _tiff_run(tmp_path / "s2", seed=2)
    options = Options(behav_codec="lossless", verify=True)
    sessions = [make_session("neuimg", tmp_path / "s1", options), make_session("neuimg", tmp_path / "s2", options)]
    progress: list[tuple[int, str, int, int]] = []
    started: list[int] = []
    results_seen: list[int] = []

    results = run_batch(
        sessions,
        options,
        on_start=lambda i, s: started.append(i),
        on_progress=lambda i, phase, done, total: progress.append((i, phase, done, total)),
        on_result=lambda i, r: results_seen.append(i),
        cancel=threading.Event(),
    )

    assert [r.status for r in results] == ["ok", "ok"]
    assert started == [0, 1]
    assert results_seen == [0, 1]
    assert (0, "compressing", 3, 3) in progress
    assert (1, "verifying", 3, 3) in progress
    assert (tmp_path / "s1.zarr").exists() and (tmp_path / "s2.zarr").exists()


def test_run_batch_runs_behav_session_into_mp4(tmp_path: Path) -> None:
    _tiff_run(tmp_path / "run00", n_pages=5, dtype=np.uint8)
    options = Options(behav_codec="lossy", behav_fps=10.0, verify=True)
    session = make_session("behav", tmp_path / "run00", options)

    result = run_session(session, options, _noop_progress)

    assert result.status == "ok", result.message
    assert session.output == tmp_path / "run00.mp4"
    assert session.output.exists()


def test_existing_output_is_skipped_not_overwritten(tmp_path: Path) -> None:
    _tiff_run(tmp_path / "s1")
    options = Options()
    session = make_session("neuimg", tmp_path / "s1", options)
    session.output.mkdir()
    marker = session.output / "keep.txt"
    marker.write_text("untouched")

    result = run_session(session, options, _noop_progress)

    assert result.status == "skipped"
    assert marker.read_text() == "untouched"


def test_failed_session_is_reported_and_batch_continues(tmp_path: Path) -> None:
    _tiff_run(tmp_path / "good")
    bad = tmp_path / "bad"
    bad.mkdir()
    tifffile.imwrite(bad / "run_0000.tif", np.zeros((2, 8, 8), dtype=np.complex64))
    options = Options()
    sessions = [make_session("neuimg", bad, options), make_session("neuimg", tmp_path / "good", options)]

    results = run_batch(
        sessions, options, on_start=lambda *a: None, on_progress=lambda *a: None,
        on_result=lambda *a: None, cancel=threading.Event(),
    )

    assert [r.status for r in results] == ["failed", "ok"]
    assert "unsupported dtype" in results[0].message
    assert not sessions[0].output.exists()


def test_cancel_before_start_runs_nothing(tmp_path: Path) -> None:
    _tiff_run(tmp_path / "s1")
    options = Options()
    sessions = [make_session("neuimg", tmp_path / "s1", options)]
    cancel = threading.Event()
    cancel.set()
    started: list[int] = []

    results = run_batch(
        sessions, options, on_start=lambda i, s: started.append(i), on_progress=lambda *a: None,
        on_result=lambda *a: None, cancel=cancel,
    )

    assert [r.status for r in results] == ["cancelled"]
    assert started == []
    assert not sessions[0].output.exists()


def test_remove_partial_deletes_directories_and_files(tmp_path: Path) -> None:
    (tmp_path / "x.zarr" / "data").mkdir(parents=True)
    (tmp_path / "x.mp4").write_bytes(b"partial")

    _remove_partial(tmp_path / "x.zarr")
    _remove_partial(tmp_path / "x.mp4")
    _remove_partial(tmp_path / "never-created")

    assert list(tmp_path.iterdir()) == []
