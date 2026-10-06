# `dc neuimg`

Losslessly compress widefield and cellular (2P / calcium) imaging recordings
stored as multi-page TIFF stacks. The output is a Zarr v3 directory with
OME-NGFF axis metadata, so napari, Fiji and other OME tools can open it
directly. Decompress restores the original TIFF files with the same names and
frame counts, and the pixel values are identical.

Compression is always lossless. There is no `--bps` or other lossy option.

## Install

See the [top-level install](../../../README.md#install). You need
`dc` on your PATH; no extra system packages are required.

## Input layout

One directory per recording, containing one or more TIFF files, each of which may
hold many frames (pages):

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

## Commands

### `probe`: inspect without writing

```bash
dc neuimg probe E:/data/widefield/250913_YW071/run00
```

Prints file count, frame count, resolution, dtype and total size.

### `compress`

```bash
dc neuimg compress E:/data/widefield/run00 E:/data/widefield/run00.zarr --fps 30
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
dc neuimg verify E:/data/widefield/run00 E:/data/widefield/run00.zarr
```

Use it after copying the `.zarr` to the server and before deleting the TIFFs.
Exits with code 1 on FAIL.

### `decompress`: back to TIFF

```bash
dc neuimg decompress E:/data/widefield/run00.zarr restored/run00
```

Writes one TIFF per original file, with the original names and page counts.

## Output layout

```
run00.zarr/                 # Zarr v3
  data          (frames, y, x)  uint16 (or the source dtype), 1 frame per chunk,
                                 Blosc zstd + bitshuffle
  zarr.json     attributes: ome (OME-NGFF 0.5 multiscales, axes t, y, x),
                source_files, frames_per_file, fps, compression settings
```

Loading in Python:

```python
import zarr
frames = zarr.open_group("run00.zarr", mode="r")["data"]
frames[100]            # one frame, decoded on demand
frames[100:200, :, :]  # a range, without decoding the whole recording
```

## Size

The compression ratio depends on the data: sensor noise limits how much a lossless
codec can save. Run `dc neuimg compress` on a real recording and record the ratio
it prints here.
