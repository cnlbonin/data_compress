# data-compress

Command-line tools that cut storage for systems-neuroscience data without
compromising later analysis.

## Install

Needs Python ≥ 3.11 and `ffmpeg`. Commands are bash; on Windows use Git Bash.

With conda (installs ffmpeg too):

```bash
conda create -n data-compress -c conda-forge python=3.11 ffmpeg
conda activate data-compress
pip install .
data-compress -h
```

Or with [uv](https://docs.astral.sh/uv/), after installing ffmpeg yourself:

```bash
uv tool install .
data-compress -h
```

See the [video README](src/data_compress/video/README.md#install) for details.

## Modules

| Command | Data | Docs |
|---|---|---|
| `data-compress video` | Behavioral camera TIFF sequences (eye/face cams) → mp4/mkv | [video README](src/data_compress/video/README.md) |

## Development

```bash
uv run pytest
# or, in a conda env:
pip install -e . pytest && pytest
```
