# data-compress

Command-line tools that cut storage for systems-neuroscience data without
compromising later analysis.

## Install

Needs Python ≥ 3.11 and `ffmpeg` (behav); the ephys module builds a
WavPack extension at install time (on macOS run `brew install wavpack` first; see the
[ephys README](src/data_compress/ephys/README.md#install)). Commands are bash; on Windows use Git Bash.

Recommended, with [uv](https://docs.astral.sh/uv/) (install ffmpeg separately).
uv installs straight from GitHub, so you don't need to clone the repo:

```bash
uv tool install git+https://github.com/cnlbonin/data_compress
dc -h
```

| Task | Command |
|---|---|
| update to the latest `main` | `uv tool upgrade data-compress` |
| install a specific branch, tag or commit | `uv tool install --reinstall git+https://github.com/cnlbonin/data_compress@<ref>` |
| use SSH instead of HTTPS | `uv tool install git+ssh://git@github.com/cnlbonin/data_compress.git` |
| remove | `uv tool uninstall data-compress` |

Upgrading also replaces the old `data-compress` command from installs made
before the command was renamed to `dc`.

Or with conda (installs ffmpeg too):

```bash
conda create -n data-compress -c conda-forge python=3.11 ffmpeg
conda activate data-compress
pip install git+https://github.com/cnlbonin/data_compress
dc -h
```

Update with `pip install --upgrade --force-reinstall --no-deps git+https://github.com/cnlbonin/data_compress`.

See the [behav README](src/data_compress/behav/README.md#install) for details.

## Modules

| Command | Data | Docs |
|---|---|---|
| `dc behav` | Behavioral camera TIFF sequences (eye/face cams) → mp4/mkv | [behav README](src/data_compress/behav/README.md) |
| `dc neuimg` | Widefield / cellular imaging TIFF stacks → lossless Zarr (OME-NGFF) | [neuimg README](src/data_compress/neuimg/README.md) |
| `dc ephys` | SpikeGLX `.bin` recordings (Neuropixels) → WavPack-compressed Zarr | [ephys README](src/data_compress/ephys/README.md) |
| `dc gui` | Window for batch compression of all three data types (see [GUI](#gui)) | this README |

## GUI

`dc gui` opens a window for batch compression, as an alternative to the CLI:

1. choose the data type: behavioral video, ephys, or imaging;
2. add sessions with **Add…**, or find all sessions under a folder with **Scan folder…**;
3. set the options: codec (behav), bits per sample (ephys), fps (behav, optional), and whether to verify;
4. press **Start batch**. The bars show overall progress and the current session.

Session paths are shown in full in the table, and the table scrolls horizontally for long paths.

Each output is written next to its source, with the same name as the CLI batch
examples (for example `run00/` → `run00.mp4`, or `run_g0.imec0.ap.bin` → `run_g0.imec0.ap.zarr`).
Sessions whose output already exists are skipped, and a failed session's partial
output is removed. **Cancel** stops after the current session finishes; a session
that is already running is not interrupted.

Worker processes for ephys and imaging use the CLI default, up to 8 per session. The
GUI does not expose `--jobs`.

### Troubleshooting

`dc gui` checks Tk before opening the window and reports a problem if it can't:

- **"no graphical display"**: the shell has no `DISPLAY` set. Run `dc gui` from a
  desktop terminal, or over SSH with X forwarding (`ssh -X` or `ssh -Y`).
- **"Tk aborted (signal 6) while creating a text-entry widget"**: the Python in use
  has a Tk build that does not match the system X libraries. Some uv builds (the
  default `uv venv` Python) do this. Use the system Python instead:

  ```bash
  uv sync --python /usr/bin/python3
  ```

  Set `UV_PYTHON` to the same interpreter, so later `uv run` commands keep using it:

  ```fish
  set -Ux UV_PYTHON /usr/bin/python3
  ```

  The system Python needs tkinter (`sudo apt install python3-tk` on Debian/Ubuntu),
  and version 3.11 or newer.

## Development

From a clone (`git clone git@github.com:cnlbonin/data_compress.git`):

```bash
uv run pytest
# or, in a conda env:
pip install -e . pytest && pytest
```
