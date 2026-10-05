# data-compress

Command-line tools that cut storage for systems-neuroscience data without
compromising later analysis.

## Install

Needs Python ≥ 3.11 and `ffmpeg` (video); the ephys module builds a
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

See the [video README](src/data_compress/video/README.md#install) for details.

## Modules

| Command | Data | Docs |
|---|---|---|
| `dc video` | Behavioral camera TIFF sequences (eye/face cams) → mp4/mkv | [video README](src/data_compress/video/README.md) |
| `dc ephys` | SpikeGLX `.bin` recordings (Neuropixels) → WavPack-compressed Zarr | [ephys README](src/data_compress/ephys/README.md) |

## Development

From a clone (`git clone git@github.com:cnlbonin/data_compress.git`):

```bash
uv run pytest
# or, in a conda env:
pip install -e . pytest && pytest
```
