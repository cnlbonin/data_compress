# data-compress

Command-line tools that cut storage for systems-neuroscience data without
compromising later analysis.

## Install

Requires [uv](https://docs.astral.sh/uv/) and `ffmpeg`/`ffprobe` on `PATH`.

Windows (PowerShell):

```powershell
winget install --id astral-sh.uv -e
winget install --id Gyan.FFmpeg -e
# open a new terminal, then from the repo folder:
uv tool install .
data-compress -h
```

macOS:

```bash
brew install uv ffmpeg
uv tool install .
data-compress -h
```

See the [video README](src/data_compress/video/README.md) for details.

## Modules

| Command | Data | Docs |
|---|---|---|
| `data-compress video` | Behavioral camera TIFF sequences (eye/face cams) → mp4/mkv | [video README](src/data_compress/video/README.md) |

## Development

```bash
uv run pytest
```
