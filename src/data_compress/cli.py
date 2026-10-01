"""Root CLI: `data-compress <data-type> <command>`."""

from __future__ import annotations

import typer

from data_compress.video.cli import app as video_app

app = typer.Typer(help="Storage-saving compression tools for systems-neuroscience data.")
app.add_typer(video_app, name="video")
