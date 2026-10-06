"""Root CLI: `dc <data-type> <command>`."""

from __future__ import annotations

import typer

from data_compress.behav.cli import app as behav_app
from data_compress.ephys.cli import app as ephys_app
from data_compress.neuimg.cli import app as neuimg_app

app = typer.Typer(
    help="Storage-saving compression tools for systems-neuroscience data.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
app.add_typer(behav_app, name="behav")
app.add_typer(ephys_app, name="ephys")
app.add_typer(neuimg_app, name="neuimg")
