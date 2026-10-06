import time
import tkinter as tk
from pathlib import Path

import numpy as np
import pytest
import tifffile

import data_compress.gui.app as app_module
from data_compress.gui.app import BatchApp
from tests.ephys.spikeglx_fixtures import make_traces, write_spikeglx


class _StubEntry:
    """Stands in for ttk.Entry, only when tk_problem() reports this Python cannot create one.

    The window reads its values from the StringVars, which the tests set directly.
    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    def pack(self, **kwargs) -> None:
        pass

    def configure(self, **kwargs) -> None:
        pass


@pytest.fixture
def app(monkeypatch):
    if app_module.tk_problem() is not None:  # this Python cannot create Entry widgets safely
        monkeypatch.setattr(app_module.ttk, "Entry", _StubEntry)
    try:
        window = BatchApp()
    except tk.TclError as exc:  # no display available
        pytest.skip(f"no display for tkinter: {exc}")
    yield window
    window.destroy()


def _pump_until_idle(app: BatchApp, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while app.running and time.monotonic() < deadline:
        app.update()
        time.sleep(0.02)
    app.update()
    assert not app.running, "batch did not finish in time"


def _silence_dialogs(monkeypatch, errors: list[str], warnings: list[str]) -> None:
    monkeypatch.setattr("data_compress.gui.app.messagebox.showerror", lambda title, msg: errors.append(msg))
    monkeypatch.setattr("data_compress.gui.app.messagebox.showwarning", lambda title, msg: warnings.append(msg))
    monkeypatch.setattr("data_compress.gui.app.messagebox.showinfo", lambda title, msg: None)


def test_add_and_run_neuimg_batch_through_window(app: BatchApp, tmp_path: Path) -> None:
    for name in ("run00", "run01"):
        d = tmp_path / name
        d.mkdir()
        tifffile.imwrite(d / "run_0000.tif", np.zeros((2, 8, 8), dtype=np.uint16), photometric="minisblack")
    app.data_type.set("neuimg")
    app._on_type_change()

    app._add_paths([tmp_path / "run00", tmp_path / "run01"])
    assert len(app.sessions) == 2
    assert len(app.tree.get_children()) == 2

    app._start()
    _pump_until_idle(app)

    assert all(status.startswith("done") for status in app.session_status), app.session_status
    assert (tmp_path / "run00.zarr").exists() and (tmp_path / "run01.zarr").exists()
    assert app.overall_label.cget("text").startswith("Finished: 2 done")


def test_rejects_paths_that_are_not_sessions(app: BatchApp, tmp_path: Path, monkeypatch) -> None:
    errors: list[str] = []
    warnings: list[str] = []
    _silence_dialogs(monkeypatch, errors, warnings)
    empty = tmp_path / "empty"
    empty.mkdir()
    app.data_type.set("neuimg")
    app._on_type_change()

    app._add_paths([empty])

    assert app.sessions == []
    assert warnings and str(empty) in warnings[0]


def test_invalid_bits_per_sample_blocks_start(app: BatchApp, tmp_path: Path, monkeypatch) -> None:
    errors: list[str] = []
    warnings: list[str] = []
    _silence_dialogs(monkeypatch, errors, warnings)
    bin_path = write_spikeglx(tmp_path, make_traces(200, n_neural=4, n_sync=1))
    app.data_type.set("ephys")
    app._on_type_change()
    app._add_paths([bin_path])
    app.bps.set("20")  # outside the lossy WavPack range

    app._start()

    assert errors and "bps must be in" in errors[0]
    assert not app.running
    assert not bin_path.with_suffix(".zarr").exists()


def test_batch_app_methods_do_not_shadow_tkinter_internals() -> None:
    # tkinter calls private helpers such as master._options() (e.g. in filedialog);
    # a method with the same name would break dialogs at runtime.
    own_methods = [name for name in vars(BatchApp) if not name.startswith("__")]
    assert [name for name in own_methods if hasattr(tk.Tk, name)] == []
