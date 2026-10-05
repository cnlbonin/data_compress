# `dc ephys`

Compress SpikeGLX recordings (Neuropixels `.ap.bin` / `.lf.bin`, `.nidq.bin`)
with **WavPack**, the codec that Buccino et al. found best for extracellular
ephys ([eLife 2025, doi:10.7554/eLife.110170](https://doi.org/10.7554/eLife.110170);
codec benchmark in Buccino et al. 2023). The output is a Zarr directory that
[SpikeInterface](https://spikeinterface.readthedocs.io) loads directly, and it
can be turned back into a SpikeGLX `.bin` + `.meta` for tools that need raw binary
(Kilosort, etc.).

| Mode | Neuropixels 1.0 / 2.0 ratio (paper) | Spike sorting impact (Kilosort4, paper) |
|---|---|---|
| lossless (default) | ~3.5× (~28% of original) | none (bit-exact) |
| `--bps 3` | ~5.3× | not significant |
| `--bps 2.5` | ~6.4× | not significant |
| `--bps 2.25` (lowest supported) | ~7.1× | not significant; still beats Kilosort2.5 on lossless data |

The paper validated lossy compression on the **AP** stream (NP1) and the wide-band
stream (NP2) only. Keep LF and NIDQ streams lossless. The CLI warns if you pass
`--bps` for them.

## Install

This module depends on
[`wavpack-numcodecs`](https://github.com/AllenNeuralDynamics/wavpack-numcodecs),
which PyPI ships only as source, so it is compiled during install. Do the
platform step first:

- **macOS**: install the WavPack library: `brew install wavpack`.
- **Windows**: the package bundles the WavPack DLL but still compiles a small C
  extension. If the install fails with a compiler error, install the
  [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
  ("Desktop development with C++") and rerun the install.
- **Linux**: uses the system `wavpack` if present, else bundled builds (glibc 2.35 / 2.39).

Then install the `dc` command with [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv tool install git+https://github.com/cnlbonin/data_compress
dc ephys -h
```

uv installs straight from GitHub (no clone needed), keeps the tool in its own
isolated environment and puts `dc` on your PATH.
If `dc` is then "command not found", run `uv tool update-shell` and reopen the
terminal.

| Task | Command |
|---|---|
| update to the latest `main` | `uv tool upgrade data-compress` |
| check what is installed | `uv tool list` |
| remove | `uv tool uninstall data-compress` |

If you installed before the command was renamed to `dc`, the upgrade above
replaces the old `data-compress` command with `dc`.

conda users: see the [top-level install](../../../README.md#install) and run
`brew install wavpack` (macOS) before `pip install git+https://github.com/cnlbonin/data_compress`.

## Commands

### `probe`: inspect without writing

```bash
dc ephys probe .../run_g0_t0.imec0.ap.bin
```

Shows the stream type, neural and sync channel counts, sample rate, duration and
size. It warns if the file size disagrees with `fileSizeBytes` in the `.meta`,
which usually means the recording is truncated or still being written.

### `compress`

```bash
# lossless (default)
dc ephys compress run_g0_t0.imec0.ap.bin run_g0_t0.imec0.ap.zarr

# lossy, ~5x
dc ephys compress run_g0_t0.imec0.ap.bin run_g0_t0.imec0.ap.zarr --bps 3
```

| Option | Default | Meaning |
|---|---|---|
| `--bps` | none (lossless) | lossy target bits per sample, 2.25–16 |
| `--level` | 3 | WavPack effort 1–4: higher is smaller but slower |
| `--overwrite` | off | replace an existing output |
| `--no-verify` | off | skip the read-back check |
| `--jobs` / `-j` | CPU count, max 8 | worker processes for compress, verify and decompress |

The work is split into 1-s chunk ranges that run in parallel worker processes, so
`compress` and `verify` scale with `--jobs` (about 8x faster at 8 jobs on a 60-s AP recording).
The output does not depend on `--jobs`.

The `.meta` sidecar must sit next to the `.bin`. After writing, the output is
decoded again and compared with the source:

- lossless: **PASS** only if every sample is identical.
- lossy: **PASS** if the sync channel(s) are identical. The command also prints the
  max absolute error and the RMS error relative to the signal's RMS.

The **sync channel is always stored losslessly**, even with `--bps`, because its
TTL bits are what you align behaviour and imaging to. Lossy coding would corrupt it.

### `verify`: re-check an existing output

```bash
dc ephys verify run_g0_t0.imec0.ap.bin run_g0_t0.imec0.ap.zarr
```

It accepts `--jobs` too.

Runs the same read-back check as `compress`, for an output written earlier, e.g.
after copying it to the server and before deleting the raw `.bin`. It prints the
mode, whether the data and sync channel(s) are identical, the max absolute error
and the relative RMS error, then PASS or FAIL. It exits with code 1 on FAIL, so you
can chain it in scripts: `dc ephys verify a.bin a.zarr && rm a.bin`.

### `decompress`: back to SpikeGLX binary

```bash
dc ephys decompress run_g0_t0.imec0.ap.zarr restored/run_g0_t0.imec0.ap.bin
```

Writes the interleaved int16 `.bin` plus the original `.meta`. For lossless
output, both files are byte-identical to the originals.

## Output layout

```
run_g0_t0.imec0.ap.zarr/        # Zarr v2 format, as SpikeInterface writes it
  traces_seg0   (samples, neural channels) int16, WavPack, 1-s chunks
  sync_seg0     (samples, sync channels)   int16, WavPack lossless
  channel_ids   0..n-1
  .zattrs       sampling_frequency, num_segments, compression settings,
                full .meta as a dict and as original text
```

Loading in Python:

```python
import spikeinterface as si
import wavpack_numcodecs  # registers the codec with numcodecs
recording = si.read_zarr("run_g0_t0.imec0.ap.zarr")  # neural channels only

import zarr
sync = zarr.open_group("run_g0_t0.imec0.ap.zarr")["sync_seg0"][:]
```

SpikeInterface sees the neural channels only. Probe geometry is not attached;
it is available from the stored `.meta` (`spikeglx_meta` attribute) if needed.
