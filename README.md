# Event tracker

Event-camera-enhanced visual tracking for drone pursuit. Built for a bachelor's thesis on fusing Prophesee event cameras with standard RGB cameras to track fast-moving objects.

Combines a MOSSE correlation-filter tracker with event-based optical flow, a Kalman filter, and template-based re-identification so the tracker can follow fast, erratic targets (like drones) through occlusions, blur, and temporary loss.

## What's in here

Three tools that read from M3ED HDF5 data files:

| Binary | What it does |
|--------|-------------|
| `tracker` | **MOSSE tracker + event flow + re-ID.** Tracks a user-selected target using MOSSE on RGB frames, event-camera optical flow for inter-frame prediction, a Kalman filter for smoothing, and template-based re-identification for recovery after loss. |
| `calibrate` | **Offline camera calibration.** Finds the scale + offset mapping between the event camera and RGB camera by maximizing edge overlap across multiple frames. Saves the result to a YAML file for the tracker to load. |
| `align` | **Interactive alignment tool.** Visual tool for manually inspecting and fine-tuning the event-to-RGB camera mapping with live sliders and overlay. |

## Data

Uses [M3ED](https://m3ed.io/) format HDF5 files. The test dataset is `falcon_indoor_flight_1_data.h5` — a Falcon 250 drone flying indoors, captured with a Prophesee stereo event camera (1280×720) and an OVC RGB camera (1280×800 @ 25fps).

Put your `.h5` file in `data/`.

## Building

Dependencies: OpenCV 4.x (with `opencv_contrib` for legacy MOSSE tracker), HDF5, Eigen3.

```bash
cd build
cmake ..
make -j$(nproc)
```

Or build individual targets:

```bash
make tracker
make calibrate
make align
```

## Quick start

HDF5 with LZF compression needs the lzf library preloaded:

```bash
# 1. Calibrate the cameras (run once per dataset)
LD_PRELOAD=/lib/x86_64-linux-gnu/liblzf.so \
  ./build/calibrate data/falcon_indoor_flight_1_data.h5 -o calibration.yaml

# 2. Run the tracker
LD_PRELOAD=/lib/x86_64-linux-gnu/liblzf.so \
  ./build/tracker data/falcon_indoor_flight_1_data.h5 --calib calibration.yaml
```

## Calibration

The event camera and RGB camera have different intrinsics, so we need a scale + offset transform to map between them:

```
u_evt = sx * u_rgb + ox
v_evt = sy * v_rgb + oy
```

The `calibrate` tool finds these four parameters automatically by running a multi-pass grid search that maximizes edge overlap between Canny edges from the RGB frame and dilated event edges.

**Usage:**
```
./calibrate <data.h5> [-o calibration.yaml] [--side left|right]
                      [--frames 20] [--wide] [--preview]
```

| Flag | Description |
|------|-------------|
| `-o <path>` | Output YAML file (default: print to stdout) |
| `--frames N` | Number of frames to sample (default: 20, more = slower but more robust) |
| `--wide` | Use a wider search range for cameras with unknown intrinsics |
| `--preview` | Show a visual overlay of the final alignment before saving |
| `--side left\|right` | Which event camera to use (default: left) |

The tracker loads the calibration file with `--calib`. If no calibration is provided, it falls back to hardcoded M3ED intrinsics with a warning.

## Tracker

### Usage

1. Run the tracker — it starts playing the RGB video
2. Press **SPACE** to pause
3. Draw a bounding box around whatever you want to track
4. Press **SPACE** again to start tracking

Two windows open: the left shows the RGB frame with the MOSSE bounding box (green), the event-predicted box (yellow), and a flow arrow. The right shows the event camera's time-surface.

**Controls:**
- `SPACE` — pause / draw box / start tracking
- `R` — reset tracker
- `E` — toggle event heatmap overlay
- `T` — toggle trajectory trail
- `+`/`-` — adjust event time window (how many ms of events to use)
- `Q` — quit

**Options:**
```
./tracker <data.h5> [--calib calibration.yaml]
                    [--side left|right] [--start <seconds>] [--duration <seconds>]
```

### How it works

The tracking pipeline fuses three sources of information:

1. **MOSSE correlation filter** — runs on each RGB frame (~25fps). Fast and accurate when the target is visible, but drifts on blur or occlusion.

2. **Event-based optical flow** — between RGB frames, events from the Prophesee camera estimate motion inside the bounding box:
   - A time-surface records when each pixel last fired
   - A local plane fit on the 5×5 neighborhood gives the time-surface gradient
   - Normal flow: $\mathbf{v} = -\nabla T / |\nabla T|^2$
   - Per-event flows are averaged over the bbox and EMA-smoothed across frames

3. **Kalman filter + re-identification** — a 4-state Kalman filter (position + velocity) smooths the MOSSE output and predicts where the target should be. When MOSSE drifts (detected by comparing its output to the Kalman prediction), the tracker:
   - Re-seats MOSSE at the Kalman-predicted position (if the gap exceeds 15 px)
   - If the target is fully lost, enters recovery mode and searches for the target using template matching (NCC) against a gallery of recent appearance patches
   - The Kalman velocity decays while the target is lost, preventing runaway predictions

### Alignment tool

Use this if you want to visually inspect and manually tune the camera alignment. For automated calibration, use `calibrate` instead.

**Controls:**
- `A` — auto-align (grid search on current frame)
- `M` — multi-frame align (samples 5 frames, more robust)
- `S` — print the alignment parameters
- `N`/`P` — next / previous frame
- Sliders — manual fine-tuning of scale and offset
- `Q` — quit

**Options:**
```
./align <data.h5> [--side left|right] [--time <seconds>]
```

## Project structure

```
src/
  tracker_main.cpp      MOSSE + event flow + Kalman + re-ID tracker
  calibrate_main.cpp    Offline camera calibration tool
  align_main.cpp        Interactive camera alignment tool
  auto_calibrate.h      Shared calibration logic (grid search, YAML I/O)
  reidentifier.h/.cpp   Kalman filter + template gallery re-ID module
  event.h / event.cpp   Event data types + H5 reader
  rgb_reader.h / .cpp   RGB frame reader (from H5)
data/
  *.h5                  M3ED dataset files
build/
  tracker, calibrate, align
```
