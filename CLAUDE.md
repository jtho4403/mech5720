# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Lab resources for MECH5720 (Sensors and Signals) at the University of Sydney. The code implements multiplexed illumination capture and relighting using a Raspberry Pi camera and a monitor as a light stage. There is no build system, test suite, or package — these are standalone Python scripts.

## Two-machine architecture

The workflow is split between a **Raspberry Pi** (capture/server) and a **laptop/PC** (client/analysis):

| Machine | Scripts |
|---------|---------|
| Raspberry Pi | `CodeBasics/stream.py`, `Lab2-Multiplexing/capture_relighting.py` |
| Laptop / PC | `CodeBasics/camview_capture.py`, `CodeBasics/view_raw.py`, `Lab2-Multiplexing/relighting.py`, `tools/convert_dng_to_tif.py` |

The Pi exposes an HTTP API on port 8000 (MJPEG stream, DNG/raw capture, camera controls). The laptop client connects to it by IP.

## Data pipeline

```
Pi capture → *.dng files  →  convert_dng_to_tif.py  →  *.tif files  →  relighting.py  →  out/*.png
```

1. **Capture** (`Lab2-Multiplexing/capture_relighting.py` on Pi): produces five named DNG stacks — `Multiplexed`, `Ambient`, `Impulse`, `AllOn`, `FirstOn` — written to `data/captured/`.
2. **Convert** (`tools/convert_dng_to_tif.py` on laptop): batch-converts the DNGs to 16-bit TIFFs with unity white balance and linear gamma so PIL can load them.
3. **Analyse** (`Lab2-Multiplexing/relighting.py` on laptop): loads the TIF stacks, demultiplexes using a Hadamard S-matrix, relights, and saves figures to `out/`.

## Running the scripts

**Pi camera HTTP server:**
```bash
python3 CodeBasics/stream.py
```

**Pi camera server over SSH (needs display for fullscreen illumination):**
```bash
DISPLAY=:0 python3 Lab2-Multiplexing/capture_relighting.py
DISPLAY=:0 python3 Lab2-Multiplexing/capture_relighting.py --preflight   # checks only, no capture
```

**Laptop camera viewer (connects to Pi):**
```bash
python3 CodeBasics/camview_capture.py --host <PI_IP> --port 8000
```

**Convert DNG → TIFF (update `SRC_DIR` in the script first):**
```bash
python3 tools/convert_dng_to_tif.py
```

**Lab 2 analysis (step through in a debugger — not meant to run end-to-end):**
```bash
python3 Lab2-Multiplexing/relighting.py
```

**View a single DNG file:**
```bash
python3 CodeBasics/view_raw.py path/to/frame.dng
```

## Key configuration to update before running

`Lab2-Multiplexing/relighting.py` has hardcoded paths and constants at the top:
- `DATA_PATH` — directory containing the captured TIF stacks (currently an absolute path)
- `BLACK_LEVEL` — set to `0.0` when using pre-converted TIFs (the DNG black level is baked in by `convert_dng_to_tif.py`)
- `N_SITES`, `N_AMBIENT`, `N_ALLON`, `N_FIRSTON` — must match what was captured

`tools/convert_dng_to_tif.py` also has a hardcoded `SRC_DIR`.

## Image conventions (imaging.py)

- All images are `float64` in nominal range `[0, 1]`, **linear in scene radiance** — never gamma-corrected in the data arrays.
- Image stacks have shape `(N, H, W, C)`, where `C=1` for monochrome/raw, `C=3` for colour.
- Display gamma is applied only inside `im.show()` / `im.show_stack()` — the data itself is never altered.
- Negative pixel values are **intentional** (noise excursions below black level) and must not be clamped except at display time.
- `im.load_stack(path, prefix, n_frames, black_level)` loads frames named `<prefix>N.<ext>` sorted by trailing number.

## Lab 2 TODO items

`relighting.py` is a skeleton with eight `# TODO` markers. Each produces a visible figure change when implemented correctly. The script is designed to be stepped through section by section in a debugger (e.g. VS Code's Python debugger), not run all at once.

## Dependencies

**On Raspberry Pi:**
- `picamera2`, `flask`

**On laptop (analysis):**
- `numpy`, `scipy`, `matplotlib`, `Pillow`
- `rawpy`, `imageio` (for DNG conversion)
- `opencv-python`, `requests`, `glfw`, `pyopengl`, `imgui-bundle` (for `camview_capture.py`)
- `rawpy` (for `view_raw.py`)

## macOS-specific note

`camview_capture.py` requests an OpenGL 3.2 Core Profile on macOS (required for `imgui_bundle`'s programmable backend). This is handled automatically; no user action needed.
