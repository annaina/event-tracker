# Event tracker

Event-camera-enhanced visual tracking for drone pursuit. Built for a bachelor's thesis on fusing Prophesee event cameras with standard RGB cameras to track fast-moving objects.

## What's in here

Two tools that read from M3ED HDF5 data files:

| Binary | What it does |
|--------|-------------|
| `tracker` | **MOSSE tracker + event-camera flow.** Runs a MOSSE correlation-filter tracker on RGB frames and uses event-based optical flow to predict motion between frames. |
| `align` | **Camera alignment tool.** Finds the affine mapping between the event camera and RGB camera by maximizing edge overlap. Run this first if the cameras aren't lined up. |

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

Or build individually:

```bash
make tracker
make align
```

## Running

HDF5 with LZF compression needs the lzf library preloaded:

```bash
# Tracker
LD_PRELOAD=/lib/x86_64-linux-gnu/liblzf.so ./build/tracker data/falcon_indoor_flight_1_data.h5

# Alignment tool
LD_PRELOAD=/lib/x86_64-linux-gnu/liblzf.so ./build/align data/falcon_indoor_flight_1_data.h5
```

### Tracker usage

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
./tracker <data.h5> [--side left|right] [--start <seconds>] [--duration <seconds>]
```

### Alignment tool usage

Use this if the event camera overlay doesn't line up with the RGB frame. It shows edge overlap between the two cameras and lets you search for the best transform.

**Controls:**
- `A` — auto-align (grid search on current frame)
- `M` — multi-frame align (samples 5 frames, more robust)
- `S` — print the alignment parameters (copy-paste into tracker code)
- `N`/`P` — next / previous frame
- Sliders — manual fine-tuning of scale and offset
- `Q` — quit

**Options:**
```
./align <data.h5> [--side left|right] [--time <seconds>]
```

## How the tracker works

The MOSSE tracker runs on RGB frames at ~25fps. Between frames, we use events from the Prophesee camera to estimate optical flow inside the tracked bounding box:

1. **Time-surface**: We maintain a map of when each pixel last fired an event
2. **Plane fitting**: For each event inside the bbox, we fit a plane to the local 5×5 neighborhood of the time-surface. The gradient of this plane tells us the local flow direction
3. **Normal flow**: From the time-surface gradient (a, b), the optical flow is v = -∇T / |∇T|²
4. **Averaging + smoothing**: Per-event flows are averaged over the bbox and smoothed with an EMA across frames

The event-predicted bounding box (yellow) shows where the target is likely heading. The flow arrow shows the dominant motion direction.

## Camera calibration

The two cameras have different intrinsics. The event-to-RGB mapping is:

```
u_evt = 0.816 * u_rgb + 100.0
v_evt = 0.816 * v_rgb + 64.1
```

These come from the intrinsic parameters in the H5 file:
- **Prophesee left**: fx=1034.86, fy=1033.48, cx=629.70, cy=357.60
- **OVC RGB**: fx=1268.56, fy=1267.35, cx=649.37, cy=359.94

If you're using different data, run the `align` tool to find the correct mapping.

## Project structure

```
src/
  tracker_main.cpp      MOSSE + event flow tracker
  align_main.cpp        Camera alignment tool
  event.h / event.cpp   Event data types + H5 reader
  rgb_reader.h / .cpp   RGB frame reader (from H5)
data/
  *.h5                  M3ED dataset files
build/
  tracker, align        compiled binaries
```
