# `dcp neuimg`

Losslessly compress widefield and cellular (2P / calcium) imaging recordings
stored as multi-page **TIFF stacks** or as a **Scanbox `.sbx`** recording
(raw binary + `.mat` sidecar). The output is a Zarr v3 directory with
OME-NGFF axis metadata, so napari, Fiji and other OME tools can open it
directly. Decompress restores the original input exactly: TIFF files with the
same names and frame counts, or the `.sbx` + `.mat` pair byte-for-byte.

Both `.tif` stacks and `.sbx` recordings are good source formats to archive
losslessly as Zarr — pick whichever matches your acquisition software's
native output; there's no need to convert one to the other just for
archiving.

Compression is always lossless. There is no `--bps` or other lossy option.

Suite2p reads `.sbx` itself (via the `sbxreader` package) or TIFF directly,
but it does not read Zarr. So if you plan to run suite2p on data you've
archived here, convert the archived Zarr back to TIFF (`dcp neuimg
decompress`) or to `.sbx` first — archiving to Zarr does not change how you
run suite2p afterwards.

## Install

See the [top-level install](../../../README.md#install). You need
`dcp` on your PATH; no extra system packages are required.

## Input layout

**TIFF**: one directory per recording, containing one or more TIFF files, each of
which may hold many frames (pages):

```
E:/data/widefield/250913_YW071/run00/
  run00_00000.tif   # pages are frames, in order
  run00_00001.tif
  ...
```

- Files are ordered by natural sort (`file_2` before `file_10`), and pages inside
  each file are read in order.
- All frames must be grayscale and share the same shape and dtype. Integer and
  float dtypes are accepted; complex dtypes are rejected.

**Scanbox**: a single `.sbx` file next to its `.mat` sidecar of the same name
(both written by Scanbox itself):

```
E:/data/2p/250913_YW071/run00.sbx
E:/data/2p/250913_YW071/run00.mat
```

- Frame count isn't stored in the `.mat`; it's derived from the `.sbx` file size.
- Multi-channel and volumetric (multi-plane / ETL) recordings are supported.
- Scanbox's own tools invert pixel values for display (`65535 - raw`); this tool
  never applies that correction, so the archived values and the restored `.sbx`
  are byte-identical to the original.

## Commands

### `probe`: inspect without writing

```bash
dcp neuimg probe E:/data/widefield/250913_YW071/run00
dcp neuimg probe E:/data/2p/250913_YW071/run00.sbx
```

Prints frame count, resolution, dtype and total size (plus file count for TIFF,
or channel/plane counts for `.sbx`).

### `compress`

```bash
dcp neuimg compress E:/data/widefield/run00 E:/data/widefield/run00.zarr --fps 30
dcp neuimg compress E:/data/2p/run00.sbx E:/data/2p/run00.zarr --fps 15
```

| Option | Default | Meaning |
|---|---|---|
| `--fps FLOAT` | none | Frame rate. Stored as the time-axis scale (seconds per frame = 1/fps). Omit if unknown; the time axis then counts frames. |
| `--clevel INT` | `5` | Blosc zstd level 0–9: higher is smaller but slower. |
| `--overwrite` | off | Replace an existing output. |
| `--no-verify` | off | Skip the read-back check. |
| `--jobs` / `-j` | CPU count, max 8 | Worker processes. The output does not depend on this. |

After writing, the output is decoded and compared with the source frame by frame.
**PASS** means every frame is identical to the source.

### `verify`: re-check an existing output

```bash
dcp neuimg verify E:/data/widefield/run00 E:/data/widefield/run00.zarr
dcp neuimg verify E:/data/2p/run00.sbx E:/data/2p/run00.zarr
```

Use it after copying the `.zarr` to the server and before deleting the source.
Exits with code 1 on FAIL.

### `decompress`: back to the original format

```bash
dcp neuimg decompress E:/data/widefield/run00.zarr restored/run00
dcp neuimg decompress E:/data/2p/run00.zarr restored/run00
```

For TIFF input, writes one TIFF per original file, with the original names and
page counts. For `.sbx` input, writes `<name>.sbx` + `<name>.mat`, byte-identical
to the originals.

## Output layout

```
run00.zarr/                 # Zarr v3
  data          uint16 (or the source dtype), 1 timepoint per chunk, Blosc zstd + bitshuffle
                 TIFF: shape (frames, y, x)
                 .sbx: shape (frames, planes, y, x, channels)
  mat_sidecar   .sbx only: the original .mat file's raw bytes, verbatim
  zarr.json     attributes: ome (OME-NGFF 0.5 multiscales, axes matching the array's
                dimension order), source_kind ("tiff" or "sbx"), fps, compression
                settings, plus source_files/frames_per_file (TIFF) or
                sbx_stem/n_channels/n_planes/is_volumetric (.sbx)
```

Loading in Python:

```python
import zarr
frames = zarr.open_group("run00.zarr", mode="r")["data"]
frames[100]            # one frame (or timepoint), decoded on demand
frames[100:200]        # a range, without decoding the whole recording
```

## Size

The compression ratio depends on the data: sensor noise limits how much a lossless
codec can save. Run `dcp neuimg compress` on a real recording and record the ratio
it prints here.
