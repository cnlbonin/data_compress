"""Tkinter window for batch compression: pick a data type, add sessions, run with progress bars.

The batch runs in a worker thread and reports through a queue, which the window polls,
so the UI stays responsive while encoders run in their own processes.
"""

from __future__ import annotations

import functools
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from data_compress.ephys.encode import validate_bps
from data_compress.gui.batch import (
    DATA_TYPE_LABELS,
    DATA_TYPES,
    DataType,
    Options,
    Result,
    Session,
    find_sessions,
    make_session,
    output_for,
    run_batch,
)
from data_compress.tiff_source import TIFF_SUFFIXES

POLL_MS = 100
# Some Python builds (uv / conda with their own bundled Tk and X libraries) abort the whole
# process when a text-entry widget is created. Probe that in a child process first.
_TK_PROBE = "import tkinter as tk\nroot = tk.Tk()\ntk.Entry(root).pack()\nroot.update()\nroot.destroy()\n"
STATUS_LABELS = {"ok": "done", "skipped": "skipped", "failed": "FAILED", "cancelled": "cancelled"}


def _is_valid_source(data_type: DataType, path: Path) -> bool:
    try:
        if data_type == "ephys":
            return path.is_file() and path.suffix.lower() == ".bin" and path.with_suffix(".meta").is_file()
        return path.is_dir() and any(p.suffix.lower() in TIFF_SUFFIXES for p in path.iterdir())
    except OSError:
        return False


class BatchApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("data-compress")
        self.geometry("1000x700")
        self.minsize(820, 560)

        self.sessions: list[Session] = []
        self.session_status: list[str] = []
        self.events: queue.Queue = queue.Queue()
        self.cancel = threading.Event()
        self.running = False
        self.close_requested = False
        self.last_type: DataType = "behav"

        self.data_type = tk.StringVar(value="behav")
        self.codec = tk.StringVar(value="lossy")
        self.bps = tk.StringVar(value="")
        self.behav_fps = tk.StringVar(value="")
        self.verify = tk.BooleanVar(value=True)
        self.controls: list[ttk.Widget] = []  # disabled while a batch runs

        self._build()
        self._show_options()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # layout

    def _build(self) -> None:
        pad = {"padx": 8, "pady": 4}

        type_box = ttk.LabelFrame(self, text="1. Data type")
        type_box.pack(fill="x", **pad)
        for value in DATA_TYPES:
            button = ttk.Radiobutton(
                type_box,
                text=DATA_TYPE_LABELS[value],
                value=value,
                variable=self.data_type,
                command=self._on_type_change,
            )
            button.pack(side="left", padx=8, pady=4)
            self.controls.append(button)

        self.options_box = ttk.LabelFrame(self, text="2. Options")
        self.options_box.pack(fill="x", **pad)
        self.behav_options = ttk.Frame(self.options_box)
        ttk.Label(self.behav_options, text="Codec:").pack(side="left", padx=(8, 4))
        for value, text in (("lossy", "lossy mp4 (~50x)"), ("lossless", "lossless mkv (~2x)")):
            button = ttk.Radiobutton(
                self.behav_options, text=text, value=value, variable=self.codec, command=self._refresh_outputs
            )
            button.pack(side="left", padx=4)
            self.controls.append(button)
        ttk.Label(self.behav_options, text="fps (blank = from .camlog):").pack(side="left", padx=(16, 4))
        fps_entry = ttk.Entry(self.behav_options, textvariable=self.behav_fps, width=6)
        fps_entry.pack(side="left")
        self.controls.append(fps_entry)
        self.ephys_options = ttk.Frame(self.options_box)
        ttk.Label(self.ephys_options, text="Bits per sample (blank = lossless, 2.25–16):").pack(side="left", padx=(8, 4))
        entry = ttk.Entry(self.ephys_options, textvariable=self.bps, width=8)
        entry.pack(side="left")
        self.bps.trace_add("write", lambda *_: self._refresh_outputs())
        self.controls.append(entry)
        self.neuimg_options = ttk.Frame(self.options_box)
        ttk.Label(self.neuimg_options, text="Lossless zstd + bitshuffle; no options.").pack(side="left", padx=8)
        check = ttk.Checkbutton(self.options_box, text="Verify each output after compressing", variable=self.verify)
        check.pack(anchor="w", padx=8, pady=(0, 4))
        self.controls.append(check)

        sessions_box = ttk.LabelFrame(self, text="3. Sessions")
        sessions_box.pack(fill="both", expand=True, **pad)
        buttons = ttk.Frame(sessions_box)
        buttons.pack(fill="x", padx=4, pady=4)
        for text, command in (
            ("Add…", self._add),
            ("Scan folder…", self._scan),
            ("Remove selected", self._remove_selected),
            ("Clear", self._clear),
        ):
            button = ttk.Button(buttons, text=text, command=command)
            button.pack(side="left", padx=4)
            self.controls.append(button)

        tree_frame = ttk.Frame(sessions_box)
        tree_frame.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(tree_frame, columns=("source", "output", "status"), show="headings", selectmode="extended")
        self.tree.heading("source", text="Source (full path)")
        self.tree.heading("output", text="Output (full path)")
        self.tree.heading("status", text="Status")
        self.tree.column("source", width=520, stretch=False)
        self.tree.column("output", width=520, stretch=False)
        self.tree.column("status", width=220, stretch=False)
        vscroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        hscroll = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vscroll.set, xscrollcommand=hscroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vscroll.grid(row=0, column=1, sticky="ns")
        hscroll.grid(row=1, column=0, sticky="ew")

        run_box = ttk.LabelFrame(self, text="4. Run")
        run_box.pack(fill="x", **pad)
        controls = ttk.Frame(run_box)
        controls.pack(fill="x", padx=4, pady=4)
        self.start_button = ttk.Button(controls, text="Start batch", command=self._start)
        self.start_button.pack(side="left", padx=4)
        self.cancel_button = ttk.Button(controls, text="Cancel", command=self._cancel, state="disabled")
        self.cancel_button.pack(side="left", padx=4)
        self.controls.append(self.start_button)

        self.overall_label = ttk.Label(run_box, text="Idle")
        self.overall_label.pack(anchor="w", padx=8)
        self.overall_bar = ttk.Progressbar(run_box, mode="determinate")
        self.overall_bar.pack(fill="x", padx=8, pady=(0, 4))
        self.session_label = ttk.Label(run_box, text="")
        self.session_label.pack(anchor="w", padx=8)
        self.session_bar = ttk.Progressbar(run_box, mode="determinate")
        self.session_bar.pack(fill="x", padx=8, pady=(0, 6))

        log_frame = ttk.Frame(self)
        log_frame.pack(fill="both", padx=8, pady=(0, 8))
        self.log = tk.Listbox(log_frame, height=7)
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=log_scroll.set)
        self.log.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

    def _show_options(self) -> None:
        for frame in (self.behav_options, self.ephys_options, self.neuimg_options):
            frame.pack_forget()
        {"behav": self.behav_options, "ephys": self.ephys_options, "neuimg": self.neuimg_options}[
            self.data_type.get()
        ].pack(fill="x", pady=2)

    # options and session list

    def _current_options(self) -> Options:
        bps = None
        if self.data_type.get() == "ephys" and self.bps.get().strip():
            bps = float(self.bps.get())
            validate_bps(bps)
        fps = None
        if self.data_type.get() == "behav" and self.behav_fps.get().strip():
            fps = float(self.behav_fps.get())
            if fps <= 0:
                raise ValueError(f"fps must be positive, got {fps}")
        return Options(behav_codec=self.codec.get(), behav_fps=fps, ephys_bps=bps, verify=self.verify.get())

    def _options_or_default(self) -> Options:
        try:
            return self._current_options()
        except ValueError:
            return Options(behav_codec=self.codec.get(), verify=self.verify.get())

    def _on_type_change(self) -> None:
        new = self.data_type.get()
        if new == self.last_type:
            return
        if self.sessions and not messagebox.askyesno("Change data type", "Changing the data type clears the session list. Continue?"):
            self.data_type.set(self.last_type)
            return
        self.last_type = new
        self.sessions.clear()
        self.session_status.clear()
        self._show_options()
        self._rebuild_tree()

    def _refresh_outputs(self) -> None:
        if self.running:
            return
        options = self._options_or_default()
        data_type = self.data_type.get()
        self.sessions = [make_session(data_type, s.source, options) for s in self.sessions]
        self._rebuild_tree()

    def _add(self) -> None:
        data_type = self.data_type.get()
        if data_type == "ephys":
            paths = [Path(p) for p in filedialog.askopenfilenames(filetypes=[("SpikeGLX binary", "*.bin"), ("All files", "*.*")])]
        else:
            chosen = filedialog.askdirectory(mustexist=True)
            paths = [Path(chosen)] if chosen else []
        self._add_paths(paths)

    def _scan(self) -> None:
        root = filedialog.askdirectory(mustexist=True)
        if not root:
            return
        found = find_sessions(self.data_type.get(), Path(root))
        if not found:
            messagebox.showinfo("Nothing found", f"No {self.data_type.get()} sessions under {root}.")
            return
        self._add_paths(found)

    def _add_paths(self, paths: list[Path]) -> None:
        data_type = self.data_type.get()
        options = self._options_or_default()
        existing = {s.source for s in self.sessions}
        rejected: list[Path] = []
        for path in paths:
            if path in existing:
                continue
            if not _is_valid_source(data_type, path):
                rejected.append(path)
                continue
            self.sessions.append(make_session(data_type, path, options))
            self.session_status.append("ready")
            existing.add(path)
        self._rebuild_tree()
        if rejected:
            listed = "\n".join(str(p) for p in rejected[:10])
            messagebox.showwarning("Not a session", f"Skipped {len(rejected)} path(s) that are not valid sessions:\n{listed}")

    def _remove_selected(self) -> None:
        for index in sorted((int(iid) for iid in self.tree.selection()), reverse=True):
            del self.sessions[index]
            del self.session_status[index]
        self._rebuild_tree()

    def _clear(self) -> None:
        self.sessions.clear()
        self.session_status.clear()
        self._rebuild_tree()

    def _rebuild_tree(self) -> None:
        self.tree.delete(*self.tree.get_children())
        for index, session in enumerate(self.sessions):
            self.tree.insert(
                "", "end", iid=str(index),
                values=(str(session.source), str(session.output), self.session_status[index]),
            )

    # running a batch

    def _set_running(self, running: bool) -> None:
        self.running = running
        for widget in self.controls:
            widget.configure(state="disabled" if running else "normal")
        self.start_button.configure(state="disabled" if running else "normal")
        self.cancel_button.configure(state="normal" if running else "disabled")

    def _start(self) -> None:
        if not self.sessions:
            messagebox.showinfo("No sessions", "Add or scan sessions first.")
            return
        try:
            options = self._current_options()
        except ValueError as exc:
            messagebox.showerror("Invalid option", str(exc))
            return
        self.cancel.clear()
        self.session_status = ["ready"] * len(self.sessions)
        self._rebuild_tree()
        self.overall_bar.configure(maximum=len(self.sessions), value=0)
        self.overall_label.configure(text=f"Starting {len(self.sessions)} session(s)…")
        self._set_running(True)
        self._log(f"--- batch: {len(self.sessions)} session(s), {self.data_type.get()} ---")
        worker = threading.Thread(target=self._worker, args=(list(self.sessions), options), daemon=True)
        worker.start()
        self.after(POLL_MS, self._poll)

    def _worker(self, sessions: list[Session], options: Options) -> None:
        results = run_batch(
            sessions,
            options,
            on_start=lambda i, s: self.events.put(("start", i, s)),
            on_progress=lambda i, phase, done, total: self.events.put(("progress", i, phase, done, total)),
            on_result=lambda i, r: self.events.put(("result", i, r)),
            cancel=self.cancel,
        )
        self.events.put(("finished", results))

    def _poll(self) -> None:
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            self._handle(event)
        if self.running:
            self.after(POLL_MS, self._poll)

    def _handle(self, event: tuple) -> None:
        kind = event[0]
        if kind == "start":
            _, index, session = event
            self._set_status(index, "running")
            self.overall_bar.configure(value=index)
            self.overall_label.configure(text=f"Session {index + 1} of {len(self.sessions)}: {session.source.name}")
            self.session_bar.configure(maximum=1, value=0)
        elif kind == "progress":
            _, index, phase, done, total = event
            self.session_bar.configure(maximum=max(total, 1), value=done)
            self.session_label.configure(text=f"{phase} {done}/{total}")
            self._set_status(index, f"{phase} {done}/{total}")
        elif kind == "result":
            _, index, result = event
            self._set_status(index, self._status_text(result))
            self.overall_bar.configure(value=index + 1)
            self._log(f"[{index + 1}] {result.session.source}: {self._status_text(result)}")
        elif kind == "finished":
            self._finish(event[1])

    def _set_status(self, index: int, text: str) -> None:
        self.session_status[index] = text
        self.tree.set(str(index), "status", text)

    @staticmethod
    def _status_text(result: Result) -> str:
        if result.status == "ok":
            return f"done: {result.message}"
        return f"{STATUS_LABELS[result.status]}: {result.message}" if result.message else STATUS_LABELS[result.status]

    def _finish(self, results: list[Result]) -> None:
        counts = {status: sum(r.status == status for r in results) for status in STATUS_LABELS}
        summary = ", ".join(f"{counts[s]} {STATUS_LABELS[s]}" for s in STATUS_LABELS if counts[s])
        self.overall_label.configure(text=f"Finished: {summary or 'nothing run'}")
        self.session_label.configure(text="")
        self.session_bar.configure(value=0)
        self._log(f"--- finished: {summary} ---")
        self._set_running(False)
        if self.close_requested:
            self.destroy()

    def _cancel(self) -> None:
        self.cancel.set()
        self.cancel_button.configure(state="disabled")
        self.overall_label.configure(text="Cancelling after the current session finishes…")

    def _on_close(self) -> None:
        if not self.running:
            self.destroy()
            return
        if messagebox.askyesno("Quit", "A batch is running. Stop after the current session and quit?"):
            self.close_requested = True
            self._cancel()

    def _log(self, text: str) -> None:
        self.log.insert("end", text)
        self.log.see("end")


@functools.cache
def tk_problem() -> str | None:
    """Return why this Python cannot show the window, or None if the probe succeeds."""
    try:
        proc = subprocess.run([sys.executable, "-c", _TK_PROBE], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"could not run the Tk check: {exc}"
    if proc.returncode == 0:
        return None
    if proc.returncode < 0:
        return f"Tk aborted (signal {-proc.returncode}) while creating a text-entry widget"
    last = (proc.stderr or proc.stdout).strip().splitlines()
    return f"Tk could not start: {last[-1] if last else 'unknown error'}"


def run() -> None:
    problem = tk_problem()
    if problem is None:
        BatchApp().mainloop()
        return
    if "display" in problem:
        hint = (
            "This shell has no graphical display. Run `dc gui` from a desktop terminal, "
            "or over SSH with X forwarding (`ssh -X` or `ssh -Y`), or set DISPLAY."
        )
    else:
        hint = (
            "Try a Python whose Tk matches your system, e.g. the distro python3 with tkinter, "
            "or `conda install -c conda-forge tk` in the environment."
        )
    print(f"dc gui cannot open a window with this Python ({sys.executable}): {problem}.\n{hint}", file=sys.stderr)
    raise SystemExit(1)
