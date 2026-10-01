# data-compress

Command-line tools that cut storage for systems-neuroscience data without
compromising later analysis.

## Install

Requires [uv](https://docs.astral.sh/uv/) and `ffmpeg`/`ffprobe` on `PATH`.

```bash
uv sync
uv run data-compress -h
```

## Modules

| Command | Data | Docs |
|---|---|---|
| `data-compress video` | Behavioral camera TIFF sequences (eye/face cams) → mp4/mkv | [video README](src/data_compress/video/README.md) |

## Development

```bash
uv run pytest
```
