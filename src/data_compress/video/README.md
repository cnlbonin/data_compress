# `data-compress video`

Compress behavioral-camera recordings (eye/face cams) stored as TIFF frame
sequences into a single video file, with checks that the compression doesn't
silently corrupt data you'll analyze later.

On real facecam data (644×484, 8-bit, ~30 fps) the default lossy mode gave
**~48x** smaller files (159.7 MB of TIFF → 3.3 MB of mp4), with decoded frames
differing from the source by ~1 gray level on average.

## Requirements

Examples below use Windows PowerShell, since most acquisition PCs run Windows.
macOS/Linux commands are given where they differ.

1. Install [uv](https://docs.astral.sh/uv/) and ffmpeg (which includes `ffprobe`):

   ```powershell
   winget install --id astral-sh.uv -e
   winget install --id Gyan.FFmpeg -e
   ```

   Open a **new** terminal afterwards so both are on `PATH`, then check with
   `ffmpeg -version`. (macOS: `brew install uv ffmpeg`.)

2. Install the CLI. Either run it from the repo folder:

   ```powershell
   cd C:\path\to\data_compress
   uv sync
   uv run data-compress -h
   ```

   or install it once as a command usable from any folder:

   ```powershell
   cd C:\path\to\data_compress
   uv tool install .
   data-compress -h
   ```

   If `data-compress` is then "not recognized", run `uv tool update-shell` and
   open a new terminal. After pulling new code, rerun `uv tool install . --reinstall`.

   The examples below assume the second option; with the first, prefix them with `uv run`.

## Input layout

One directory per recording, containing:

```
E:\data\facecam\250913_YW071__2P_YW\run00_152642_linear_combine\
  20250913_run000_00000000.tif   # each .tif may hold many frames (multi-page)
  20250913_run000_00000001.tif
  ...
  20250913_run000.camlog          # optional labcams log: frame_id,timestamp rows
```

- Files are ordered by natural sort (`file_2` before `file_10`), and pages
  inside each file are read in order.
- All frames must be grayscale and share the same shape and dtype (`uint8` or `uint16`).
- If a `.camlog` is present, it is used to derive fps and to cross-check the
  frame count. More than one `.camlog` in the directory is an error.

## Commands

### `probe` — inspect without encoding

```powershell
data-compress video probe E:\data\facecam\250913_YW071__2P_YW\run00_152642_linear_combine
```

Put paths containing spaces in quotes: `"E:\my data\run00"`.

Prints file count, frame count, resolution, dtype, and (if a `.camlog` exists)
its frame count and fps. Shows a WARNING if the TIFF and camlog frame counts differ.

### `compress` — encode to a single video

```powershell
# default: lossy H.265 -> .mp4
data-compress video compress E:\data\facecam\run00 E:\data\facecam\run00.mp4

# exact pixels: lossless FFV1 -> .mkv
data-compress video compress E:\data\facecam\run00 E:\data\facecam\run00.mkv --codec lossless

# no .camlog? give the frame rate explicitly
data-compress video compress E:\data\facecam\run00 E:\data\facecam\run00.mp4 --fps 30
```

| Option | Default | Meaning |
|---|---|---|
| `--fps FLOAT` | from `.camlog` | Frame rate of the output. Required if there's no `.camlog`. |
| `--codec lossy\|lossless` | `lossy` | See [Choosing a codec](#choosing-a-codec). |
| `--crf INT` | `18` | Lossy quality: lower = better quality and bigger files. |
| `--force` | off | Allow a 16-bit source through the lossy path (keeps only the top 8 bits). |
| `-h`, `--help` | | Show help. |

The output file's extension must match the codec (`.mp4` for lossy, `.mkv` for
lossless). After encoding, the frame count of the output is checked with
`ffprobe` and the command prints `PASS` or `FAIL`.

## Choosing a codec

| | `lossy` (default) | `lossless` |
|---|---|---|
| Encoder / container | H.265 / `.mp4` | FFV1 / `.mkv` |
| Pixels | Approximate (~±1 gray level at CRF 18) | Bit-exact, 8- or 16-bit |
| Size reduction | ~48x on facecam data | ~2–3x (typical, not measured here) |
| Good for | Pose/landmark tracking (DeepLabCut, SLEAP, Facemap), visual QC | Analyses reading raw intensities (e.g. pupil size by thresholding, frame differencing) |

Why lossy shrinks files so much: most of a face-camera image is static, so
H.265 mostly stores what changed since the previous frame, and it rounds off
fine detail that is largely sensor noise. That rounding is also why
intensity-based analyses should use `lossless`.

16-bit sources are refused on the lossy path unless you pass `--force`,
because lossy output is 8-bit and would discard the lower 8 bits.


## Batch processing

There is no built-in batch mode; loop over recordings in the shell. Each
`runXX` folder becomes `runXX.mp4` next to it.

PowerShell (Windows):

```powershell
Get-ChildItem -Directory E:\data\facecam\250913_YW071__2P_YW\run* | ForEach-Object {
    data-compress video compress $_.FullName "$($_.FullName).mp4"
}
```

bash (macOS/Linux):

```bash
for d in /path/to/session/run*/; do
  data-compress video compress "$d" "${d%/}.mp4"
done
```

Encoding uses most of the CPU, so don't run it on an acquisition PC while a
recording is in progress, where it could cause dropped frames. Compressing on
the acquisition PC between sessions, before copying to the server, also cuts
transfer time.

## Limitations

- Grayscale only (RGB TIFFs are not supported yet).
- The output uses a constant frame rate. Per-frame timestamps from the
  `.camlog` are not embedded, so keep the `.camlog` next to the video.
- A TIFF/camlog frame-count mismatch is a warning, not an error. Check it
  before deleting the original TIFFs.
- The source TIFFs are never deleted. Remove them yourself once you've
  checked the output.
